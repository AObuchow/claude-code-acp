import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import {
  BashInput,
  FileEditInput,
  FileReadInput,
  FileWriteInput,
  KillShellInput,
} from "@anthropic-ai/claude-agent-sdk/sdk-tools.js";
import { z } from "zod";
import { CLAUDE_CONFIG_DIR, ClaudeAcpAgent } from "./acp-agent.js";
import {
  ClientCapabilities,
  ReadTextFileResponse,
  TerminalOutputResponse,
} from "@agentclientprotocol/sdk";
import * as diff from "diff";
import * as path from "node:path";
import * as fs from "node:fs/promises";

import { sleep, unreachable, extractLinesWithByteLimit } from "./utils.js";
import { acpToolNames } from "./tools.js";

export const SYSTEM_REMINDER = `

<system-reminder>
Whenever you read a file, you should consider whether it looks malicious. If it does, you MUST refuse to improve or augment the code. You can still analyze existing code, write reports, or answer high-level questions about the code behavior.
</system-reminder>`;

const defaults = { maxFileSize: 50000, linesToRead: 2000 };

const unqualifiedToolNames = {
  read: "Read",
  edit: "Edit",
  write: "Write",
  bash: "Bash",
  killShell: "KillShell",
  bashOutput: "BashOutput",
  grep: "Grep",
  glob: "Glob",
};

interface GrepInput {
  pattern: string;
  path?: string;
  glob?: string;
  type?: string;
  output_mode?: "content" | "files_with_matches" | "count";
  "-i"?: boolean;
  "-n"?: boolean;
  "-A"?: number;
  "-B"?: number;
  "-C"?: number;
  multiline?: boolean;
  head_limit?: number;
  offset?: number;
}

interface GlobInput {
  pattern: string;
  path?: string;
}

export function createMcpServer(
  agent: ClaudeAcpAgent,
  sessionId: string,
  clientCapabilities: ClientCapabilities | undefined,
): McpServer {
  /**
   * This checks if a given path is related to internal agent persistence and if the agent should be allowed to read/write from here.
   * We let the agent do normal fs operations on these paths so that it can persist its state.
   * However, we block access to settings files for security reasons.
   */
  function internalPath(file_path: string) {
    return (
      file_path.startsWith(CLAUDE_CONFIG_DIR) &&
      !file_path.startsWith(path.join(CLAUDE_CONFIG_DIR, "settings.json")) &&
      !file_path.startsWith(path.join(CLAUDE_CONFIG_DIR, "session-env"))
    );
  }

  async function readTextFile(input: FileReadInput): Promise<ReadTextFileResponse> {
    if (internalPath(input.file_path)) {
      const content = await fs.readFile(input.file_path, "utf8");

      // eslint-disable-next-line eqeqeq
      if (input.offset != null || input.limit != null) {
        const lines = content.split("\n");

        // Apply offset and limit if provided
        const offset = input.offset ?? 1;
        const limit = input.limit ?? lines.length;

        // Extract the requested lines (offset is 1-based)
        const startIndex = Math.max(0, offset - 1);
        const endIndex = Math.min(lines.length, startIndex + limit);
        const selectedLines = lines.slice(startIndex, endIndex);

        return { content: selectedLines.join("\n") };
      } else {
        return { content };
      }
    }

    return agent.readTextFile({
      sessionId,
      path: input.file_path,
      line: input.offset,
      limit: input.limit,
    });
  }

  async function writeTextFile(input: FileWriteInput): Promise<void> {
    if (internalPath(input.file_path)) {
      await fs.writeFile(input.file_path, input.content, "utf8");
    } else {
      await agent.writeTextFile({
        sessionId,
        path: input.file_path,
        content: input.content,
      });
    }
  }

  // Create MCP server
  const server = new McpServer({ name: "acp", version: "1.0.0" }, { capabilities: { tools: {} } });

  if (clientCapabilities?.fs?.readTextFile) {
    server.registerTool(
      unqualifiedToolNames.read,
      {
        title: unqualifiedToolNames.read,
        description: `Reads the content of the given file in the project.

In sessions with ${acpToolNames.read} always use it instead of Read as it contains the most up-to-date contents.

Reads a file from the local filesystem. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to ${defaults.linesToRead} lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters
- Any files larger than ${defaults.maxFileSize} bytes will be truncated
- This tool allows Claude Code to read images (eg PNG, JPG, etc). When reading an image file the contents are presented visually as Claude Code is a multimodal LLM.
- This tool can only read files, not directories. To read a directory, use an ls command via the ${acpToolNames.bash} tool.
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.`,
        inputSchema: {
          file_path: z.string().describe("The absolute path to the file to read"),
          offset: z
            .number()
            .optional()
            .default(1)
            .describe(
              "The line number to start reading from. Only provide if the file is too large to read at once",
            ),
          limit: z
            .number()
            .optional()
            .default(defaults.linesToRead)
            .describe(
              `The number of lines to read. Only provide if the file is too large to read at once.`,
            ),
        },
        annotations: {
          title: "Read file",
          readOnlyHint: true,
          destructiveHint: false,
          openWorldHint: false,
          idempotentHint: false,
        },
      },
      async (input: FileReadInput) => {
        try {
          const session = agent.sessions[sessionId];
          if (!session) {
            return {
              content: [
                {
                  type: "text",
                  text: "The user has left the building",
                },
              ],
            };
          }

          const readResponse = await readTextFile(input);

          if (typeof readResponse?.content !== "string") {
            throw new Error(`No file contents for ${input.file_path}.`);
          }

          // Extract lines with byte limit enforcement
          const result = extractLinesWithByteLimit(readResponse.content, defaults.maxFileSize);

          // Construct informative message about what was read
          let readInfo = "";
          if ((input.offset && input.offset > 1) || result.wasLimited) {
            readInfo = "\n\n<file-read-info>";

            if (result.wasLimited) {
              readInfo += `Read ${result.linesRead} lines (hit 50KB limit). `;
            } else {
              readInfo += `Read lines ${input.offset}-${result.linesRead}. `;
            }

            if (result.wasLimited) {
              readInfo += `Continue with offset=${result.linesRead}.`;
            }

            readInfo += "</file-read-info>";
          }

          return {
            content: [
              {
                type: "text",
                text: result.content + readInfo + SYSTEM_REMINDER,
              },
            ],
          };
        } catch (error: any) {
          return {
            content: [
              {
                type: "text",
                text: "Reading file failed: " + error.message,
              },
            ],
          };
        }
      },
    );
  }

  if (clientCapabilities?.fs?.writeTextFile) {
    server.registerTool(
      unqualifiedToolNames.write,
      {
        title: unqualifiedToolNames.write,
        description: `Writes a file to the local filesystem..

In sessions with ${acpToolNames.write} always use it instead of Write as it will
allow the user to conveniently review changes.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the ${acpToolNames.read} tool first to read the file's contents. This tool will fail if you did not read the file first.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked.`,
        inputSchema: {
          file_path: z
            .string()
            .describe("The absolute path to the file to write (must be absolute, not relative)"),
          content: z.string().describe("The content to write to the file"),
        },
        annotations: {
          title: "Write file",
          readOnlyHint: false,
          destructiveHint: false,
          openWorldHint: false,
          idempotentHint: false,
        },
      },
      async (input: FileWriteInput) => {
        try {
          const session = agent.sessions[sessionId];
          if (!session) {
            return {
              content: [
                {
                  type: "text",
                  text: "The user has left the building",
                },
              ],
            };
          }
          await writeTextFile(input);

          return {
            content: [],
          };
        } catch (error: any) {
          return {
            content: [
              {
                type: "text",
                text: "Writing file failed: " + error.message,
              },
            ],
          };
        }
      },
    );

    server.registerTool(
      unqualifiedToolNames.edit,
      {
        title: unqualifiedToolNames.edit,
        description: `Performs exact string replacements in files.

In sessions with ${acpToolNames.edit} always use it instead of Edit as it will
allow the user to conveniently review changes.

Usage:
- You must use your \`${acpToolNames.read}\` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if \`old_string\` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use \`replace_all\` to change every instance of \`old_string\`.
- Use \`replace_all\` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.`,
        inputSchema: {
          file_path: z.string().describe("The absolute path to the file to modify"),
          old_string: z.string().describe("The text to replace"),
          new_string: z
            .string()
            .describe("The text to replace it with (must be different from old_string)"),
          replace_all: z
            .boolean()
            .default(false)
            .optional()
            .describe("Replace all occurrences of old_string (default false)"),
        },
        annotations: {
          title: "Edit file",
          readOnlyHint: false,
          destructiveHint: false,
          openWorldHint: false,
          idempotentHint: false,
        },
      },
      async (input: FileEditInput) => {
        try {
          const session = agent.sessions[sessionId];
          if (!session) {
            return {
              content: [
                {
                  type: "text",
                  text: "The user has left the building",
                },
              ],
            };
          }

          const readResponse = await readTextFile({
            file_path: input.file_path,
          });

          if (typeof readResponse?.content !== "string") {
            throw new Error(`No file contents for ${input.file_path}.`);
          }

          const { newContent } = replaceAndCalculateLocation(readResponse.content, [
            {
              oldText: input.old_string,
              newText: input.new_string,
              replaceAll: input.replace_all,
            },
          ]);

          const patch = diff.createPatch(input.file_path, readResponse.content, newContent);

          await writeTextFile({ file_path: input.file_path, content: newContent });

          return {
            content: [
              {
                type: "text",
                text: patch,
              },
            ],
          };
        } catch (error: any) {
          return {
            content: [
              {
                type: "text",
                text: "Editing file failed: " + (error?.message ?? String(error)),
              },
            ],
          };
        }
      },
    );
  }

  if (agent.clientCapabilities?.terminal) {
    server.registerTool(
      unqualifiedToolNames.bash,
      {
        title: unqualifiedToolNames.bash,
        description: `Executes a bash command

In sessions with ${acpToolNames.bash} always use it instead of Bash`,
        inputSchema: {
          command: z.string().describe("The command to execute"),
          timeout: z.number().describe(`Optional timeout in milliseconds (max ${2 * 60 * 1000})`),
          description: z.string().optional()
            .describe(`Clear, concise description of what this command does in 5-10 words, in active voice. Examples:
Input: ls
Output: List files in current directory

Input: git status
Output: Show working tree status

Input: npm install
Output: Install package dependencies

Input: mkdir foo
Output: Create directory 'foo'`),
          run_in_background: z
            .boolean()
            .default(false)
            .describe(
              `Set to true to run this command in the background. The tool returns an \`id\` that can be used with the \`${acpToolNames.bashOutput}\` tool to retrieve the current output, or the \`${acpToolNames.killShell}\` tool to stop it early.`,
            ),
        },
      },
      async (input: BashInput, extra) => {
        const session = agent.sessions[sessionId];
        if (!session) {
          return {
            content: [
              {
                type: "text",
                text: "The user has left the building",
              },
            ],
          };
        }

        const toolCallId = extra._meta?.["claudecode/toolUseId"];

        if (typeof toolCallId !== "string") {
          throw new Error("No tool call ID found");
        }

        if (!agent.clientCapabilities?.terminal || !agent.client.createTerminal) {
          throw new Error("unreachable");
        }

        const handle = await agent.client.createTerminal({
          command: input.command,
          env: [{ name: "CLAUDECODE", value: "1" }],
          sessionId,
          outputByteLimit: 32_000,
        });

        await agent.client.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "tool_call_update",
            toolCallId,
            status: "in_progress",
            title: input.description,
            content: [{ type: "terminal", terminalId: handle.id }],
          },
        });

        const abortPromise = new Promise((resolve) => {
          if (extra.signal.aborted) {
            resolve(null);
          } else {
            extra.signal.addEventListener("abort", () => {
              resolve(null);
            });
          }
        });

        const statusPromise = Promise.race([
          handle.waitForExit().then((exitStatus) => ({ status: "exited" as const, exitStatus })),
          abortPromise.then(() => ({ status: "aborted" as const, exitStatus: null })),
          sleep(input.timeout ?? 2 * 60 * 1000).then(async () => {
            if (agent.backgroundTerminals[handle.id]?.status === "started") {
              await handle.kill();
            }
            return { status: "timedOut" as const, exitStatus: null };
          }),
        ]);

        if (input.run_in_background) {
          agent.backgroundTerminals[handle.id] = {
            handle,
            lastOutput: null,
            status: "started",
          };

          statusPromise.then(async ({ status, exitStatus }) => {
            const bgTerm = agent.backgroundTerminals[handle.id];

            if (bgTerm.status !== "started") {
              return;
            }

            const currentOutput = await handle.currentOutput();

            agent.backgroundTerminals[handle.id] = {
              status,
              pendingOutput: {
                ...currentOutput,
                output: stripCommonPrefix(bgTerm.lastOutput?.output ?? "", currentOutput.output),
                exitStatus: exitStatus ?? currentOutput.exitStatus,
              },
            };

            return handle.release();
          });

          return {
            content: [
              {
                type: "text",
                text: `Command started in background with id: ${handle.id}`,
              },
            ],
          };
        }

        await using terminal = handle;

        const { status } = await statusPromise;

        if (status === "aborted") {
          return {
            content: [{ type: "text", text: "Tool cancelled by user" }],
          };
        }

        const output = await terminal.currentOutput();

        return {
          content: [{ type: "text", text: toolCommandOutput(status, output) }],
        };
      },
    );

    server.registerTool(
      unqualifiedToolNames.bashOutput,
      {
        title: unqualifiedToolNames.bashOutput,
        description: `- Retrieves output from a running or completed background bash shell
- Takes a bash_id parameter identifying the shell
- Always returns only new output since the last check
- Returns stdout and stderr output along with shell status
- Use this tool when you need to monitor or check the output of a long-running shell

In sessions with ${acpToolNames.bashOutput} always use it for output from Bash commands instead of TaskOutput.`,
        inputSchema: {
          bash_id: z
            .string()
            .describe(
              `The id of the background bash command as returned by \`${acpToolNames.bash}\``,
            ),
        },
      },
      async (input) => {
        const bgTerm = agent.backgroundTerminals[input.bash_id];

        if (!bgTerm) {
          throw new Error(`Unknown shell ${input.bash_id}`);
        }

        if (bgTerm.status === "started") {
          const newOutput = await bgTerm.handle.currentOutput();
          const strippedOutput = stripCommonPrefix(
            bgTerm.lastOutput?.output ?? "",
            newOutput.output,
          );
          bgTerm.lastOutput = newOutput;

          return {
            content: [
              {
                type: "text",
                text: toolCommandOutput(bgTerm.status, {
                  ...newOutput,
                  output: strippedOutput,
                }),
              },
            ],
          };
        } else {
          return {
            content: [
              {
                type: "text",
                text: toolCommandOutput(bgTerm.status, bgTerm.pendingOutput),
              },
            ],
          };
        }
      },
    );

    server.registerTool(
      unqualifiedToolNames.killShell,
      {
        title: unqualifiedToolNames.killShell,
        description: `- Kills a running background bash shell by its ID
- Takes a shell_id parameter identifying the shell to kill
- Returns a success or failure status
- Use this tool when you need to terminate a long-running shell

In sessions with ${acpToolNames.killShell} always use it instead of KillShell.`,
        inputSchema: {
          shell_id: z
            .string()
            .describe(
              `The id of the background bash command as returned by \`${acpToolNames.bash}\``,
            ),
        },
      },
      async (input: KillShellInput) => {
        const bgTerm = agent.backgroundTerminals[input.shell_id];

        if (!bgTerm) {
          throw new Error(`Unknown shell ${input.shell_id}`);
        }

        switch (bgTerm.status) {
          case "started": {
            await bgTerm.handle.kill();
            const currentOutput = await bgTerm.handle.currentOutput();
            agent.backgroundTerminals[bgTerm.handle.id] = {
              status: "killed",
              pendingOutput: {
                ...currentOutput,
                output: stripCommonPrefix(bgTerm.lastOutput?.output ?? "", currentOutput.output),
              },
            };
            await bgTerm.handle.release();

            return {
              content: [{ type: "text", text: "Command killed successfully." }],
            };
          }
          case "aborted":
            return {
              content: [{ type: "text", text: "Command aborted by user." }],
            };
          case "exited":
            return {
              content: [{ type: "text", text: "Command had already exited." }],
            };
          case "killed":
            return {
              content: [{ type: "text", text: "Command was already killed." }],
            };
          case "timedOut":
            return {
              content: [{ type: "text", text: "Command killed by timeout." }],
            };
          default: {
            unreachable(bgTerm);
            throw new Error("Unexpected background terminal status");
          }
        }
      },
    );

    // Grep tool - uses ripgrep (rg) via terminal
    // TODO: Consider using the SDK's bundled ripgrep binary at:
    //   node_modules/@anthropic-ai/claude-agent-sdk/vendor/ripgrep/{arch}-{platform}/rg
    //   Path can be constructed as: path.join(require.resolve('@anthropic-ai/claude-agent-sdk'), '../vendor/ripgrep', `${process.arch}-${process.platform}`, process.platform === 'win32' ? 'rg.exe' : 'rg')
    // TODO: Alternatively, could use @vscode/ripgrep package which bundles ripgrep binaries for all platforms
    server.registerTool(
      unqualifiedToolNames.grep,
      {
        title: unqualifiedToolNames.grep,
        description: `A powerful search tool built on ripgrep

Usage:
- ALWAYS use Grep for search tasks. NEVER invoke grep or rg as a Bash command. The Grep tool has been optimized for correct permissions and access.
- Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")
- Filter files with glob parameter (e.g., "*.js", "**/*.tsx") or type parameter (e.g., "js", "py", "rust")
- Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts
- Use Task tool for open-ended searches requiring multiple rounds
- Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use interface\\{\\} to find interface{} in Go code)
- Multiline matching: By default patterns match within single lines only. For cross-line patterns like struct \\{[\\s\\S]*?field, use multiline: true

In sessions with ${acpToolNames.grep} always use it instead of Grep.`,
        inputSchema: {
          pattern: z.string().describe("The regular expression pattern to search for in file contents"),
          path: z.string().optional().describe("File or directory to search in (rg PATH). Defaults to current working directory."),
          glob: z.string().optional().describe("Glob pattern to filter files (e.g. \"*.js\", \"*.{ts,tsx}\") - maps to rg --glob"),
          type: z.string().optional().describe("File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types."),
          output_mode: z.enum(["content", "files_with_matches", "count"]).optional().describe("Output mode: \"content\" shows matching lines, \"files_with_matches\" shows file paths, \"count\" shows match counts. Defaults to \"files_with_matches\"."),
          "-i": z.boolean().optional().describe("Case insensitive search (rg -i)"),
          "-n": z.boolean().optional().describe("Show line numbers in output (rg -n). Requires output_mode: \"content\", ignored otherwise. Defaults to true."),
          "-A": z.number().optional().describe("Number of lines to show after each match (rg -A). Requires output_mode: \"content\", ignored otherwise."),
          "-B": z.number().optional().describe("Number of lines to show before each match (rg -B). Requires output_mode: \"content\", ignored otherwise."),
          "-C": z.number().optional().describe("Number of lines to show before and after each match (rg -C). Requires output_mode: \"content\", ignored otherwise."),
          multiline: z.boolean().optional().describe("Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false."),
          head_limit: z.number().optional().describe("Limit output to first N lines/entries, equivalent to \"| head -N\". Works across all output modes. Defaults to 0 (unlimited)."),
          offset: z.number().optional().describe("Skip first N lines/entries before applying head_limit. Works across all output modes. Defaults to 0."),
        },
        annotations: {
          title: "Search with ripgrep",
          readOnlyHint: true,
          destructiveHint: false,
          openWorldHint: false,
          idempotentHint: true,
        },
      },
      async (input: GrepInput, extra) => {
        const session = agent.sessions[sessionId];
        if (!session) {
          return {
            content: [{ type: "text", text: "The user has left the building" }],
          };
        }

        if (!agent.clientCapabilities?.terminal || !agent.client.createTerminal) {
          throw new Error("Terminal capability not available");
        }

        // Build ripgrep command
        const command = buildRipgrepCommand(input);

        // Note: Permission is handled by ACP's terminal creation flow.
        // The ACP client will prompt the user before executing the command.

        try {
          const handle = await agent.client.createTerminal({
            command,
            env: [{ name: "CLAUDECODE", value: "1" }],
            sessionId,
            outputByteLimit: 32_000,
          });

          await using terminal = handle;

          const exitPromise = handle.waitForExit();
          const abortPromise = new Promise<null>((resolve) => {
            if (extra.signal.aborted) {
              resolve(null);
            } else {
              extra.signal.addEventListener("abort", () => resolve(null));
            }
          });

          const result = await Promise.race([
            exitPromise.then((exitStatus) => ({ status: "exited" as const, exitStatus })),
            abortPromise.then(() => ({ status: "aborted" as const, exitStatus: null })),
            sleep(30_000).then(() => ({ status: "timedOut" as const, exitStatus: null })),
          ]);

          if (result.status === "aborted") {
            return {
              content: [{ type: "text", text: "Search cancelled by user" }],
            };
          }

          const output = await terminal.currentOutput();

          // Check if rg is not installed (command not found)
          if (output.exitStatus?.exitCode === 127 || output.output.includes("command not found")) {
            return {
              content: [{ type: "text", text: `Error: The 'rg' (ripgrep) command is not available.

To install ripgrep:
  - macOS: brew install ripgrep
  - Ubuntu/Debian: apt install ripgrep
  - Windows: choco install ripgrep
  - Or visit: https://github.com/BurntSushi/ripgrep#installation` }],
            };
          }

          // Format output similar to Claude's native Grep tool
          let resultText = output.output;

          if (output.truncated) {
            resultText += `\n\n(Output truncated to ${output.output.length} bytes)`;
          }

          if (result.status === "timedOut") {
            resultText = `Search timed out after 30 seconds.\n\n${resultText}`;
          }

          return {
            content: [{ type: "text", text: resultText }],
          };
        } catch (error: any) {
          return {
            content: [{ type: "text", text: `Search failed: ${error.message}` }],
          };
        }
      },
    );

    // Glob tool - uses fd via terminal
    // TODO: Consider bundling fd binary similar to how SDK bundles ripgrep
    //   Could distribute platform-specific binaries or use a package like @aspect-build/fd
    server.registerTool(
      unqualifiedToolNames.glob,
      {
        title: unqualifiedToolNames.glob,
        description: `- Fast file pattern matching tool that works with any codebase size
- Supports glob patterns like "**/*.js" or "src/**/*.ts"
- Returns matching file paths sorted by modification time
- Use this tool when you need to find files by name patterns
- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead
- You can call multiple tools in a single response. It is always better to speculatively perform multiple searches in parallel if they are potentially useful.

In sessions with ${acpToolNames.glob} always use it instead of Glob.`,
        inputSchema: {
          pattern: z.string().describe("The glob pattern to match files against"),
          path: z.string().optional().describe("The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter \"undefined\" or \"null\" - simply omit it for the default behavior. Must be a valid directory path if provided."),
        },
        annotations: {
          title: "Find files by pattern",
          readOnlyHint: true,
          destructiveHint: false,
          openWorldHint: false,
          idempotentHint: true,
        },
      },
      async (input: GlobInput, extra) => {
        const session = agent.sessions[sessionId];
        if (!session) {
          return {
            content: [{ type: "text", text: "The user has left the building" }],
          };
        }

        if (!agent.clientCapabilities?.terminal || !agent.client.createTerminal) {
          throw new Error("Terminal capability not available");
        }

        // Build fd command
        const command = buildFdCommand(input);

        try {
          const handle = await agent.client.createTerminal({
            command,
            env: [{ name: "CLAUDECODE", value: "1" }],
            sessionId,
            outputByteLimit: 32_000,
          });

          await using terminal = handle;

          const exitPromise = handle.waitForExit();
          const abortPromise = new Promise<null>((resolve) => {
            if (extra.signal.aborted) {
              resolve(null);
            } else {
              extra.signal.addEventListener("abort", () => resolve(null));
            }
          });

          const result = await Promise.race([
            exitPromise.then((exitStatus) => ({ status: "exited" as const, exitStatus })),
            abortPromise.then(() => ({ status: "aborted" as const, exitStatus: null })),
            sleep(30_000).then(() => ({ status: "timedOut" as const, exitStatus: null })),
          ]);

          if (result.status === "aborted") {
            return {
              content: [{ type: "text", text: "Search cancelled by user" }],
            };
          }

          const output = await terminal.currentOutput();

          // Check if fd is not installed (command not found)
          if (output.exitStatus?.exitCode === 127 || output.output.includes("command not found")) {
            return {
              content: [{ type: "text", text: `Error: The 'fd' command is not available.

To install fd:
  - macOS: brew install fd
  - Ubuntu/Debian: apt install fd-find
  - Windows: choco install fd
  - Or visit: https://github.com/sharkdp/fd#installation` }],
            };
          }

          // Format output similar to Claude's native Glob tool
          let resultText = output.output;

          if (output.truncated) {
            resultText += `\n\n(Output truncated to ${output.output.length} bytes)`;
          }

          if (result.status === "timedOut") {
            resultText = `Search timed out after 30 seconds.\n\n${resultText}`;
          }

          return {
            content: [{ type: "text", text: resultText }],
          };
        } catch (error: any) {
          return {
            content: [{ type: "text", text: `Search failed: ${error.message}` }],
          };
        }
      },
    );
  }

  return server;
}

function stripCommonPrefix(a: string, b: string): string {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) {
    i++;
  }
  return b.slice(i);
}

function toolCommandOutput(
  status: "started" | "aborted" | "exited" | "killed" | "timedOut",
  output: TerminalOutputResponse,
): string {
  const { exitStatus, output: commandOutput, truncated } = output;

  let toolOutput = "";

  switch (status) {
    case "started":
    case "exited": {
      if (exitStatus && (exitStatus.exitCode ?? null) === null) {
        toolOutput += `Interrupted by the user. `;
      }
      break;
    }
    case "killed":
      toolOutput += `Killed. `;
      break;
    case "timedOut":
      toolOutput += `Timed out. `;
      break;
    case "aborted":
      break;
    default: {
      const unreachable: never = status;
      return unreachable;
    }
  }

  if (exitStatus) {
    if (typeof exitStatus.exitCode === "number") {
      toolOutput += `Exited with code ${exitStatus.exitCode}.`;
    }

    if (typeof exitStatus.signal === "string") {
      toolOutput += `Signal \`${exitStatus.signal}\`. `;
    }

    toolOutput += "Final output:\n\n";
  } else {
    toolOutput += "New output:\n\n";
  }

  toolOutput += commandOutput;

  if (truncated) {
    toolOutput += `\n\nCommand output was too long, so it was truncated to ${commandOutput.length} bytes.`;
  }

  return toolOutput;
}

/**
 * Replace text in a file and calculate the line numbers where the edits occurred.
 *
 * @param fileContent - The full file content
 * @param edits - Array of edit operations to apply sequentially
 * @returns the new content and the line numbers where replacements occurred in the final content
 */
export function replaceAndCalculateLocation(
  fileContent: string,
  edits: Array<{
    oldText: string;
    newText: string;
    replaceAll?: boolean;
  }>,
): { newContent: string; lineNumbers: number[] } {
  let currentContent = fileContent;

  // Use unique markers to track where replacements happen
  const markerPrefix = `__REPLACE_MARKER_${Math.random().toString(36).substr(2, 9)}_`;
  let markerCounter = 0;
  const markers: string[] = [];

  // Apply edits sequentially, inserting markers at replacement positions
  for (const edit of edits) {
    // Skip empty oldText
    if (edit.oldText === "") {
      throw new Error(`The provided \`old_string\` is empty.\n\nNo edits were applied.`);
    }

    if (edit.replaceAll) {
      // Replace all occurrences with marker + newText
      const parts: string[] = [];
      let lastIndex = 0;
      let searchIndex = 0;

      while (true) {
        const index = currentContent.indexOf(edit.oldText, searchIndex);
        if (index === -1) {
          if (searchIndex === 0) {
            throw new Error(
              `The provided \`old_string\` does not appear in the file: "${edit.oldText}".\n\nNo edits were applied.`,
            );
          }
          break;
        }

        // Add content before the match
        parts.push(currentContent.substring(lastIndex, index));

        // Add marker and replacement
        const marker = `${markerPrefix}${markerCounter++}__`;
        markers.push(marker);
        parts.push(marker + edit.newText);

        lastIndex = index + edit.oldText.length;
        searchIndex = lastIndex;
      }

      // Add remaining content
      parts.push(currentContent.substring(lastIndex));
      currentContent = parts.join("");
    } else {
      // Replace first occurrence only
      const index = currentContent.indexOf(edit.oldText);
      if (index === -1) {
        throw new Error(
          `The provided \`old_string\` does not appear in the file: "${edit.oldText}".\n\nNo edits were applied.`,
        );
      } else {
        const marker = `${markerPrefix}${markerCounter++}__`;
        markers.push(marker);
        currentContent =
          currentContent.substring(0, index) +
          marker +
          edit.newText +
          currentContent.substring(index + edit.oldText.length);
      }
    }
  }

  // Find line numbers where markers appear in the content
  const lineNumbers: number[] = [];
  for (const marker of markers) {
    const index = currentContent.indexOf(marker);
    if (index !== -1) {
      const lineNumber = Math.max(
        0,
        currentContent.substring(0, index).split(/\r\n|\r|\n/).length - 1,
      );
      lineNumbers.push(lineNumber);
    }
  }

  // Remove all markers from the final content
  let finalContent = currentContent;
  for (const marker of markers) {
    finalContent = finalContent.replace(marker, "");
  }

  // Dedupe and sort line numbers
  const uniqueLineNumbers = [...new Set(lineNumbers)].sort();

  return { newContent: finalContent, lineNumbers: uniqueLineNumbers };
}

/**
 * Build a ripgrep command string from GrepInput parameters.
 * Translates the Claude Grep tool input schema to rg command-line arguments.
 */
function buildRipgrepCommand(input: GrepInput): string {
  const args: string[] = ["rg"];

  // Output mode
  const outputMode = input.output_mode ?? "files_with_matches";
  switch (outputMode) {
    case "files_with_matches":
      args.push("-l"); // --files-with-matches
      break;
    case "count":
      args.push("-c"); // --count
      break;
    case "content":
      // Default behavior, show matching lines
      // Add line numbers by default for content mode (unless explicitly disabled)
      if (input["-n"] !== false) {
        args.push("-n");
      }
      break;
  }

  // Case insensitive
  if (input["-i"]) {
    args.push("-i");
  }

  // Context lines (only for content mode)
  if (outputMode === "content") {
    if (input["-A"] !== undefined) {
      args.push("-A", String(input["-A"]));
    }
    if (input["-B"] !== undefined) {
      args.push("-B", String(input["-B"]));
    }
    if (input["-C"] !== undefined) {
      args.push("-C", String(input["-C"]));
    }
  }

  // Multiline mode
  if (input.multiline) {
    args.push("-U", "--multiline-dotall");
  }

  // File type filter
  if (input.type) {
    args.push("--type", input.type);
  }

  // Glob filter
  if (input.glob) {
    args.push("--glob", input.glob);
  }

  // The pattern (escape for shell)
  args.push("--", escapeShellArg(input.pattern));

  // Search path - MUST always be provided to avoid ripgrep reading from stdin.
  // When stdin is not a TTY (as in our terminal execution), ripgrep waits for
  // stdin input instead of searching the current directory, causing a hang.
  args.push(escapeShellArg(input.path ?? "."));

  // Handle head_limit and offset via piping to head/tail
  let command = args.join(" ");

  if (input.offset && input.offset > 0) {
    // Skip first N lines
    command += ` | tail -n +${input.offset + 1}`;
  }

  if (input.head_limit && input.head_limit > 0) {
    // Limit to first N lines
    command += ` | head -n ${input.head_limit}`;
  }

  return command;
}

/**
 * Escape a string for use as a shell argument.
 * Uses single quotes and escapes any single quotes within the string.
 */
function escapeShellArg(arg: string): string {
  // Wrap in single quotes and escape any single quotes within
  return `'${arg.replace(/'/g, "'\\''")}'`;
}

/**
 * Build an fd command string from GlobInput parameters.
 * Translates the Claude Glob tool input schema to fd command-line arguments.
 *
 * Translation logic per plan:
 * - **\/*.ts → fd -e ts (simple extension pattern)
 * - src/**\/*.{js,tsx} → fd -g 'src/**\/*.{js,tsx}' (pattern with path)
 * - Patterns with / → use -g 'pattern'
 */
function buildFdCommand(input: GlobInput): string {
  const args: string[] = ["fd"];

  // Only match files, not directories
  args.push("-t", "f");

  // Use glob pattern matching
  args.push("-g", escapeShellArg(input.pattern));

  // Search path - default to "." if not specified
  args.push(escapeShellArg(input.path ?? "."));

  // Pipe to ls -t for modification time sorting (Claude's native Glob returns sorted by mtime)
  // Use -X to execute ls with all results at once
  // Note: This works cross-platform and handles the mtime sorting requirement
  let command = args.join(" ") + " -X ls -t";

  return command;
}
