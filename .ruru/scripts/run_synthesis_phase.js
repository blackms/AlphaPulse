#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const TOML = require('@iarna/toml');

// --- Argument Parsing ---
const argv = yargs(hideBin(process.argv))
  .option('library', {
    alias: 'l',
    description: 'The library name (e.g., python-cpython)',
    type: 'string',
    demandOption: true,
  })
  .option('mode', {
    alias: 'm',
    description: 'The target mode slug (e.g., dev-python)',
    type: 'string',
    demandOption: true,
  })
  .option('task-set', {
    alias: 't',
    description: 'Path to the TOML file defining synthesis tasks',
    type: 'string',
    demandOption: true,
  })
  .option('source-dir', {
    alias: 's',
    description: 'Path to the structured source KB directory',
    type: 'string',
  })
  .option('output-dir', {
    alias: 'o',
    description: 'Path for synthesized output',
    type: 'string',
  })
  .option('mcp-server', {
    description: 'Name of the MCP server providing the LLM tool',
    type: 'string',
    default: 'vertex-ai-mcp-server', // Defaulting as per common usage, can be overridden
  })
  .option('llm-tool', {
    description: 'Name of the LLM tool on the MCP server',
    type: 'string',
    default: 'answer_query_direct', // Defaulting, can be overridden
  })
  .help()
  .alias('help', 'h')
  .strict() // Report errors for unknown options
  .argv;

// --- Helper Functions ---

/**
 * Logs a message to the console with a timestamp.
 * @param {string} message - The message to log.
 * @param {'INFO' | 'WARN' | 'ERROR'} level - The log level.
 */
function log(message, level = 'INFO') {
  console.log(`[${new Date().toISOString()}] [${level}] ${message}`);
}

// --- Constants ---
const MAX_CONTEXT_LENGTH = 50000; // Max characters for combined context before truncation
const COORDINATOR_TASK_ID = 'TASK-CMD-20250425-202054'; // For traceability

// --- Main Synthesis Logic ---

async function runSynthesis() {
  log(`Starting KB Synthesis Phase for library: ${argv.library}, mode: ${argv.mode}. Coordinator: ${COORDINATOR_TASK_ID}`);

  // --- Validate and Prepare Paths ---
  const libraryName = argv.library;
  const modeSlug = argv.mode;
  const taskSetPath = path.resolve(argv.taskSet); // Ensure absolute path

  const sourceDir = path.resolve(argv.sourceDir || path.join('kb', libraryName));
  const outputDir = path.resolve(argv.outputDir || path.join('.ruru', 'modes', modeSlug, 'kb', libraryName, 'synthesized'));

  log(`Task Set Path: ${taskSetPath}`);
  log(`Source Directory: ${sourceDir}`);
  log(`Output Directory: ${outputDir}`);

  // --- Read Task Set ---
  let taskSet;
  try {
    const taskSetContent = fs.readFileSync(taskSetPath, 'utf-8');
    taskSet = TOML.parse(taskSetContent);
    log(`Successfully parsed task set file: ${taskSetPath}`);
  } catch (error) {
    log(`Failed to read or parse task set file: ${taskSetPath}. Error: ${error.message}`, 'ERROR');
    process.exit(1);
  }

  // --- Ensure Output Directory Exists ---
  try {
    fs.mkdirSync(outputDir, { recursive: true });
    log(`Ensured output directory exists: ${outputDir}`);
  } catch (error) {
    log(`Failed to create output directory: ${outputDir}. Error: ${error.message}`, 'ERROR');
    process.exit(1);
  }

  // --- Iterate Through Tasks ---
  if (!taskSet.task || !Array.isArray(taskSet.task)) {
      log(`No tasks found or 'task' is not an array in ${taskSetPath}`, 'ERROR');
      process.exit(1);
  }

  for (const task of taskSet.task) {
    const taskName = task.output_filename || 'Unnamed Task';
    log(`Processing task: ${taskName}`);

    if (!task.input_categories || !Array.isArray(task.input_categories)) {
        log(`Task '${taskName}' is missing 'input_categories' array. Skipping.`, 'WARN');
        continue;
    }
    if (!task.prompt_focus || typeof task.prompt_focus !== 'string') {
        log(`Task '${taskName}' is missing 'prompt_focus' string. Skipping.`, 'WARN');
        continue;
    }
     if (!task.output_filename || typeof task.output_filename !== 'string') {
        log(`Task '${taskName}' is missing 'output_filename' string. Skipping.`, 'WARN');
        continue;
    }

    // --- Find and Read Source Files ---
    let combinedContext = '';
    let filesFoundCount = 0;
    let contextTruncated = false;
    for (const category of task.input_categories) {
      const categoryDir = path.join(sourceDir, category);
      log(`Searching for source files in category: ${categoryDir}`);
      try {
        if (!fs.existsSync(categoryDir)) {
            log(`Source category directory does not exist: ${categoryDir}`, 'WARN');
            continue;
        }
        const files = fs.readdirSync(categoryDir);
        const mdFiles = files.filter(file => file.endsWith('.md'));

        if (mdFiles.length === 0) {
            log(`No .md files found in category directory: ${categoryDir}`, 'WARN');
            continue;
        }

        for (const file of mdFiles) {
          if (combinedContext.length >= MAX_CONTEXT_LENGTH) {
              if (!contextTruncated) {
                  log(`Combined context reached limit (${MAX_CONTEXT_LENGTH} chars). Further files for this task will be ignored.`, 'WARN');
                  contextTruncated = true;
              }
              break; // Stop reading files for this category if limit reached
          }

          const filePath = path.join(categoryDir, file);
          try {
            const content = fs.readFileSync(filePath, 'utf-8');
            const contentToAdd = `\n\n## Content from ${category}/${file}\n\n${content}`;

            if (combinedContext.length + contentToAdd.length > MAX_CONTEXT_LENGTH) {
                const remainingSpace = MAX_CONTEXT_LENGTH - combinedContext.length;
                combinedContext += contentToAdd.substring(0, remainingSpace) + '\n... [TRUNCATED]';
                log(`Read source file (truncated): ${filePath}`);
                if (!contextTruncated) {
                    log(`Combined context reached limit (${MAX_CONTEXT_LENGTH} chars) during file read.`, 'WARN');
                    contextTruncated = true;
                }
            } else {
                combinedContext += contentToAdd;
                log(`Read source file: ${filePath}`);
            }
            filesFoundCount++;

          } catch (readError) {
            log(`Failed to read source file: ${filePath}. Error: ${readError.message}`, 'WARN');
          }
        } // End file loop
      } catch (dirError) {
        log(`Error accessing category directory: ${categoryDir}. Error: ${dirError.message}`, 'WARN');
      }
      if (contextTruncated) break; // Stop processing categories if limit reached
    } // End category loop

    if (filesFoundCount === 0) {
        log(`No source files found for task '${taskName}'. Skipping LLM invocation.`, 'WARN');
        continue;
    }

    log(`Found ${filesFoundCount} source files for task '${taskName}'. Combined context length: ${combinedContext.length}${contextTruncated ? ' (Truncated)' : ''}`);

    // --- Prepare Prompt and Context ---
    const prompt = `${task.prompt_focus}\n\n${combinedContext}`;

    // --- Invoke LLM ---
    let synthesizedContent = '';
    log(`Invoking LLM tool '${argv.llmTool}' on server '${argv.mcpServer}' for task '${taskName}'...`);
    try {
      // NOTE: This script runs outside the Roo agent context, so it cannot directly use <use_mcp_tool>.
      // It needs to interact with the MCP server via its defined protocol (e.g., HTTP SSE or stdio).
      // The implementation below is a conceptual placeholder assuming direct interaction is needed.
      // A real implementation would require an MCP client library or direct HTTP/stdio communication setup.

      // Placeholder for actual MCP client interaction
      log('MCP interaction logic needs to be implemented here (e.g., using axios for HTTP or child_process for stdio).', 'WARN');
      // Simulating a successful call with mock data for now
      synthesizedContent = `## Mock Synthesized Content for ${taskName}\n\nBased on prompt: "${task.prompt_focus}"\n\nContext Length: ${combinedContext.length}${contextTruncated ? ' (Truncated)' : ''}\n\nThis is placeholder content. Implement actual MCP client logic.\n`;
      // Example using a hypothetical MCP client library (replace with actual implementation):
      // const mcpClient = require('./mcp-client'); // Hypothetical client
      // const response = await mcpClient.useTool(argv.mcpServer, argv.llmTool, { query: prompt });
      // synthesizedContent = response.result; // Assuming result structure

      log(`Received synthesized content for task '${taskName}' (Length: ${synthesizedContent.length})`);
    } catch (llmError) {
      log(`LLM invocation failed for task '${taskName}'. Error: ${llmError.message}`, 'ERROR');
      // Decide whether to continue with other tasks or exit
      continue; // Continue for now
    }

    // --- Write Output ---
    const outputFilePath = path.join(outputDir, task.output_filename);
    try {
      fs.writeFileSync(outputFilePath, synthesizedContent, 'utf-8');
      log(`Successfully wrote synthesized output to: ${outputFilePath}`);
    } catch (writeError) {
      log(`Failed to write output file: ${outputFilePath}. Error: ${writeError.message}`, 'ERROR');
      // Decide whether to continue
    }
  } // End task loop

  log('KB Synthesis Phase script finished.');
}

// --- Run the script ---
runSynthesis().catch(error => {
  log(`Unhandled error during script execution: ${error.message}`, 'ERROR');
  console.error(error.stack); // Print stack trace for debugging
  process.exit(1);
});