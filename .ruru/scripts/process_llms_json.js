const fs = require('fs');
const path = require('path');
// Removed yargs imports

// Node's built-in fetch is used, assuming Node v18+
// const fetch = require('node-fetch'); // Uncomment if using older Node versions and install node-fetch

const GITHUB_SRC_PREFIX = '/src/';
const DEFAULT_OUTPUT_DIR = './tmp/';
const DEFAULT_INPUT_FILE = 'llms.json';

/**
 * Extracts the relative source path from a GitHub URL.
 * e.g., "https://github.com/vuejs/docs/blob/main/src/guide/essentials/forms.md#_snippet_12" -> "guide/essentials/forms.md"
 * @param {string} codeId - The GitHub URL.
 * @returns {string|null} The relative path or null if extraction fails.
 */
function extractSourcePath(codeId) {
    try {
        const url = new URL(codeId);
        const pathname = url.pathname;
        const fragment = url.hash;

        const srcIndex = pathname.indexOf(GITHUB_SRC_PREFIX);
        if (srcIndex === -1) {
            console.warn(`[WARN] Could not find '${GITHUB_SRC_PREFIX}' in codeId: ${codeId}`);
            return null;
        }

        let relativePath = pathname.substring(srcIndex + GITHUB_SRC_PREFIX.length);

        // Clean potential fragment remnants from the path
        const hashIndex = relativePath.indexOf('#');
        if (hashIndex !== -1) {
            relativePath = relativePath.substring(0, hashIndex);
        }

        // Decode URI components in case of encoded characters in filenames
        return decodeURIComponent(relativePath);
    } catch (error) {
        console.error(`[ERROR] Error parsing codeId URL: ${codeId}`, error);
        return null;
    }
}

/**
 * Generates Markdown content for a group of snippets from the same source file.
 * @param {Array<object>} snippets - Array of snippet objects for the same source file.
 * @returns {string} The generated Markdown content.
 */
function generateMarkdown(snippets) {
    if (!snippets || snippets.length === 0) {
        return '';
    }

    // Use the pageTitle from the first snippet for the H1
    let markdownContent = `# ${snippets[0].pageTitle}\n\n`;

    snippets.forEach((snippet, index) => {
        markdownContent += `## ${snippet.codeTitle}\n\n`;
        markdownContent += `${snippet.codeDescription || 'No description provided.'}\n\n`; // Add fallback for description
        markdownContent += `Source: ${snippet.codeId}\n\n`;

        if (snippet.codeList && Array.isArray(snippet.codeList)) {
            snippet.codeList.forEach(codeBlock => {
                // Use 'text' as fallback language if none provided
                const language = codeBlock.language || 'text';
                markdownContent += `\`\`\`${language}\n`;
                markdownContent += `${codeBlock.code}\n`;
                markdownContent += `\`\`\`\n\n`;
            });
        } else {
            console.warn(`[WARN] Snippet has missing or invalid codeList: ${snippet.codeId}`);
            markdownContent += `\`\`\`text\nCode not available\n\`\`\`\n\n`;
        }


        if (index < snippets.length - 1) {
            markdownContent += '---\n\n';
        }
    });

    return markdownContent;
}

/**
 * Fetches JSON data from a URL.
 * @param {string} urlString - The URL to fetch data from.
 * @returns {Promise<Array<object>>} - A promise that resolves with the parsed JSON data.
 */
async function fetchJsonData(urlString) {
    console.log(`[INFO] Fetching JSON data from: ${urlString}...`);
    try {
        const response = await fetch(urlString);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        console.log(`[INFO] Successfully fetched and parsed JSON data.`);
        return data;
    } catch (error) {
        console.error(`[ERROR] Failed to fetch or parse JSON from ${urlString}:`, error);
        throw error; // Re-throw to be caught by the main function
    }
}

/**
 * Reads and parses JSON data from a local file.
 * @param {string} filePath - The path to the JSON file.
 * @returns {Array<object>} - The parsed JSON data.
 */
function readJsonDataFromFile(filePath) {
    console.log(`[INFO] Reading JSON data from file: ${filePath}...`);
    try {
        const rawData = fs.readFileSync(filePath, 'utf8');
        const data = JSON.parse(rawData);
        console.log(`[INFO] Successfully read and parsed JSON data from ${filePath}.`);
        return data;
    } catch (error) {
        console.error(`[ERROR] Failed to read or parse JSON from ${filePath}:`, error);
        throw error; // Re-throw to be caught by the main function
    }
}

/**
 * Parses command line arguments using process.argv.
 * @returns {object} - An object containing parsed arguments (baseUrl, inputFile, outputDir).
 */
function parseArgs() {
    const args = process.argv.slice(2); // Skip node executable and script path
    const parsedArgs = {
        baseUrl: null,
        inputFile: null,
        outputDir: DEFAULT_OUTPUT_DIR, // Default output directory
        tokens: null // Add tokens argument
    };

    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        const value = args[i + 1]; // Potential value for the argument

        if (arg === '--baseUrl' || arg === '-b') {
            if (value && !value.startsWith('--')) {
                parsedArgs.baseUrl = value;
                i++; // Skip the value in the next iteration
            } else {
                console.error(`[ERROR] Missing value for argument: ${arg}`);
                process.exit(1);
            }
        } else if (arg === '--inputFile' || arg === '-i') {
            if (value && !value.startsWith('--')) {
                parsedArgs.inputFile = value;
                i++; // Skip the value
            } else {
                console.error(`[ERROR] Missing value for argument: ${arg}`);
                process.exit(1);
            }
        } else if (arg === '--outputDir' || arg === '-o') {
            if (value && !value.startsWith('--')) {
                parsedArgs.outputDir = value;
                i++; // Skip the value
            } else {
                console.error(`[ERROR] Missing value for argument: ${arg}`);
                process.exit(1);
            }
        } else if (arg === '--tokens' || arg === '-t') {
            if (value && !value.startsWith('--')) {
                const numValue = parseInt(value, 10);
                if (isNaN(numValue)) {
                    console.error(`[ERROR] Invalid value for argument ${arg}: "${value}". Must be a number.`);
                    process.exit(1);
                }
                parsedArgs.tokens = numValue;
                i++; // Skip the value
            } else {
                console.error(`[ERROR] Missing value for argument: ${arg}`);
                process.exit(1);
            }
        } else if (arg === '--help' || arg === '-h') {
            console.log(`
Usage: node process_llms_json.js [options]

Options:
  -b, --baseUrl <url>      Base URL for the documentation (e.g., "https://context7.com/vercel/next.js").
  -i, --inputFile <path>   Path to a local JSON file to process. Overrides baseUrl.
  -o, --outputDir <path>   Base directory for output Markdown files and indexes. (default: "${DEFAULT_OUTPUT_DIR}")
  -t, --tokens <number>    Optional. Max tokens to request from baseUrl (appends ?tokens=<number>).
  -h, --help             Display this help message.

If neither --inputFile nor --baseUrl is provided, defaults to reading "${DEFAULT_INPUT_FILE}" in the current directory.
If both are provided, --inputFile takes precedence.
If --baseUrl is used without --tokens, the default behavior of the target server applies.
`);
            process.exit(0);
        } else {
            console.warn(`[WARN] Ignoring unknown argument: ${arg}`);
        }
    }

    // Apply logic previously in yargs.check
    if (!parsedArgs.inputFile && !parsedArgs.baseUrl) {
        console.log(`[INFO] No --inputFile or --baseUrl provided. Defaulting to reading "${DEFAULT_INPUT_FILE}" in current directory.`);
        // No need to explicitly set inputFile here, the main function handles the default read
    } else if (parsedArgs.inputFile && parsedArgs.baseUrl) {
         console.log('[INFO] Both --inputFile and --baseUrl provided. Prioritizing --inputFile.');
         // baseUrl will be ignored by the main function's logic
    }

    // Ensure outputDir is treated as a directory path and resolve to absolute path
    parsedArgs.outputDir = path.resolve(parsedArgs.outputDir);

    return parsedArgs;
}


/**
 * Main asynchronous function to process the JSON data and generate files.
 * @param {object} args - Parsed command-line arguments.
 */
async function processLlmsJson(args) {
    const outputBaseDir = args.outputDir;

    // Ensure the base output directory exists before any write operations
    fs.mkdirSync(outputBaseDir, { recursive: true });

    let llmsData;

    try {
        if (args.inputFile) {
            // Prioritize inputFile if provided
            llmsData = readJsonDataFromFile(args.inputFile);
        } else if (args.baseUrl) {
            // Use baseUrl if inputFile is not provided
            let fetchUrl = `${args.baseUrl.replace(/\/$/, '')}/llms.json`; // Ensure single slash
            if (args.tokens !== null) {
                fetchUrl += `?tokens=${args.tokens}`;
                console.log(`[INFO] Using specified token count: ${args.tokens}`);
            } else {
                 console.log(`[INFO] No --tokens specified, fetching default llms.json from base URL.`);
            }
            llmsData = await fetchJsonData(fetchUrl);
        } else {
            // Default to reading llms.json in the current directory
            llmsData = readJsonDataFromFile(DEFAULT_INPUT_FILE);
        }

        if (!Array.isArray(llmsData)) {
            throw new Error('Input JSON data is not an array.');
        }
        console.log(`[INFO] Processing ${llmsData.length} snippets.`);

    } catch (error) {
        console.error(`[FATAL] Could not load input JSON data. Exiting.`, error);
        process.exit(1);
    }

    const groupedSnippets = {};
    console.log('[INFO] Grouping snippets by source file...');
    llmsData.forEach(snippet => {
        // Basic validation of snippet structure
        if (!snippet || typeof snippet.codeId !== 'string' || typeof snippet.pageTitle !== 'string') {
             console.warn(`[WARN] Skipping invalid snippet object: ${JSON.stringify(snippet)}`);
             return;
        }

        const sourcePath = extractSourcePath(snippet.codeId);
        if (sourcePath) {
            if (!groupedSnippets[sourcePath]) {
                groupedSnippets[sourcePath] = [];
            }
            groupedSnippets[sourcePath].push(snippet);
        } else {
            console.warn(`[WARN] Skipping snippet due to invalid or unextractable codeId: ${snippet.codeId}`);
        }
    });
    console.log(`[INFO] Grouped snippets into ${Object.keys(groupedSnippets).length} unique source files.`);

    const rootIndexData = []; // To store info for the root _index.json

    console.log(`[INFO] Generating Markdown files and subdirectory indexes in ${outputBaseDir}...`);
    for (const sourcePath in groupedSnippets) {
        console.log(`[INFO] Processing ${sourcePath}...`);
        const snippets = groupedSnippets[sourcePath];
        const markdownContent = generateMarkdown(snippets);

        // Determine output paths
        const outputFileName = path.basename(sourcePath);
        // Ensure .md extension if missing (some paths might not have it)
        const markdownFileName = outputFileName.endsWith('.md') ? outputFileName : `${outputFileName}.md`;
        const outputDirName = path.dirname(sourcePath);
        const fullOutputDir = path.join(outputBaseDir, outputDirName);
        const fullOutputFilePath = path.join(fullOutputDir, markdownFileName);

        // Data for subdirectory index
        const pageTitle = snippets[0].pageTitle; // Assuming all snippets for a file share the same pageTitle
        const snippetTitles = snippets.map(s => s.codeTitle).filter(Boolean); // Get all codeTitles

        try {
            console.log(`[INFO] Ensuring directory exists: ${fullOutputDir}`);
            fs.mkdirSync(fullOutputDir, { recursive: true });

            console.log(`[INFO] Writing Markdown file: ${fullOutputFilePath}...`);
            fs.writeFileSync(fullOutputFilePath, markdownContent, 'utf8');

            // Add/Update entry for the subdirectory index
            const subIndexFilePath = path.join(fullOutputDir, '_index.json');
            let subIndexData = [];
            if (fs.existsSync(subIndexFilePath)) {
                try {
                    subIndexData = JSON.parse(fs.readFileSync(subIndexFilePath, 'utf8'));
                    if (!Array.isArray(subIndexData)) subIndexData = []; // Reset if invalid
                } catch (parseError) {
                    console.warn(`[WARN] Could not parse existing sub-index ${subIndexFilePath}. Overwriting.`, parseError);
                    subIndexData = [];
                }
            }

            // Check if this file is already in the index, update if necessary or add
            const existingFileIndex = subIndexData.findIndex(item => item.fileName === markdownFileName);
            const fileIndexEntry = {
                fileName: markdownFileName,
                pageTitle: pageTitle,
                snippetTitles: snippetTitles
            };

            if (existingFileIndex > -1) {
                subIndexData[existingFileIndex] = fileIndexEntry; // Update existing
            } else {
                subIndexData.push(fileIndexEntry); // Add new
            }

            console.log(`[INFO] Writing subdirectory index: ${subIndexFilePath}...`);
            fs.writeFileSync(subIndexFilePath, JSON.stringify(subIndexData, null, 2), 'utf8');

            // Prepare data for the root index
            const relativeSubIndexPath = path.relative(outputBaseDir, subIndexFilePath);
            const relativeSubDirPath = path.relative(outputBaseDir, fullOutputDir) || '.'; // Use '.' for root dir itself

            // Find if this directory is already in the root index
            const existingRootIndexEntry = rootIndexData.find(item => item.indexPath === relativeSubIndexPath);
            if (existingRootIndexEntry) {
                // Add page title if not already present
                if (!existingRootIndexEntry.containedPageTitles.includes(pageTitle)) {
                    existingRootIndexEntry.containedPageTitles.push(pageTitle);
                }
            } else {
                // Add new entry for this directory
                rootIndexData.push({
                    indexPath: relativeSubIndexPath,
                    directoryPath: relativeSubDirPath,
                    containedPageTitles: [pageTitle] // Start with the current page title
                });
            }

        } catch (error) {
            console.error(`[ERROR] Error creating directory, writing file, or updating index for ${sourcePath}:`, error);
            // Continue processing other files
        }
    }

    // Write the root index file
    const rootIndexFilePath = path.join(outputBaseDir, '_index.json');
    try {
        console.log(`[INFO] Writing root index: ${rootIndexFilePath}...`);
        // Sort root index by directory path for consistency
        rootIndexData.sort((a, b) => a.directoryPath.localeCompare(b.directoryPath));
        fs.writeFileSync(rootIndexFilePath, JSON.stringify(rootIndexData, null, 2), 'utf8');
    } catch (error) {
        console.error(`[ERROR] Failed to write root index file ${rootIndexFilePath}:`, error);
    }

    console.log('[INFO] Processing complete.');
}

// --- Main Execution ---
const args = parseArgs();

// Run the main processing function
processLlmsJson(args).catch(error => {
    console.error("[FATAL] Unhandled error during processing:", error);
    process.exit(1);
});