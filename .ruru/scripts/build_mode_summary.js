const fs = require('fs').promises;
const path = require('path');

const MODES_DIR = '.ruru/modes';
// Define output paths and filename
const OUTPUT_FILENAME = 'available-modes-summary.md';
const OUTPUT_PATH_KB = path.join('.ruru', 'modes', 'roo-commander', 'kb', OUTPUT_FILENAME);
const OUTPUT_PATH_RULES = path.join('.roo', 'rules', 'roo-commander', OUTPUT_FILENAME);
const DEFAULT_GROUPS = ["read", "edit", "browser", "command", "mcp"]; // Kept for potential future use, though not used in summary

let tomlParser = null;
let TomlError = Error; // Generic error if no parser found

// --- Attempt to import TOML parsers ---
try {
    const jToml = require('@ltd/j-toml');
    // j-toml parse returns an array for tables, we need the object directly
    tomlParser = (tomlString) => jToml.parse(tomlString, { joiner: '\n' });
    console.log("Using '@ltd/j-toml' for TOML parsing.");
} catch (e) {
    console.log("Info: '@ltd/j-toml' not found. Trying '@iarna/toml'.");
    try {
        const iarnaToml = require('@iarna/toml');
        tomlParser = iarnaToml.parse;
        TomlError = Error; // Use generic Error for @iarna/toml
        console.log("Using '@iarna/toml' for TOML parsing.");
    } catch (e2) {
        console.warn("Warning: Neither '@ltd/j-toml' nor '@iarna/toml' found. Will attempt regex fallback for TOML parsing (less reliable).");
        // No specific TomlError class available for regex fallback
    }
}

// --- Regex Fallback Functions ---
function extractTomlFrontmatterRegex(content) {
    const match = content.match(/^\+\+\+\s*([\s\S]*?)\s*\+\+\+/m);
    return match ? match[1] : null;
}

// Updated Regex parser to include description
function parseTomlWithRegex(tomlContent, filePath) {
    const data = {};
    let success = true;

    const idMatch = tomlContent.match(/^\s*id\s*=\s*"(.*?)"/m);
    if (idMatch) {
        data.id = idMatch[1];
    } else {
        console.warn(`  Warning: Regex failed to find 'id' in ${filePath}.`);
        success = false;
    }

    const nameMatch = tomlContent.match(/^\s*name\s*=\s*"(.*?)"/m);
    if (nameMatch) {
        data.name = nameMatch[1];
    } else {
        console.warn(`  Warning: Regex failed to find 'name' in ${filePath}.`);
        success = false;
    }

    // Regex for potentially multiline system_prompt
    const promptMatch = tomlContent.match(/^\s*system_prompt\s*=\s*"""([\s\S]*?)"""/m);
    if (promptMatch) {
        data.system_prompt = promptMatch[1];
    } else {
        // Try single line prompt match as fallback
        const singleLinePromptMatch = tomlContent.match(/^\s*system_prompt\s*=\s*"(.*?)"/m);
         if (singleLinePromptMatch) {
             data.system_prompt = singleLinePromptMatch[1];
         } else {
            console.warn(`  Warning: Regex failed to find 'system_prompt' (multiline or single) in ${filePath}.`);
            success = false;
         }
    }

    // Regex for summary (optional) - Changed from description
    const summaryMatch = tomlContent.match(/^\s*summary\s*=\s*"(.*?)"/m);
    if (summaryMatch) {
        data.summary = summaryMatch[1];
    } else {
        data.summary = "[No summary provided]"; // Default if not found
        console.log(`  Info: Regex did not find 'summary' in ${filePath}. Using default.`);
    }


    return success ? data : null;
}


// --- Main Build Function ---
// Changed function name
async function buildModeSummary() {
    let modes_data = [];

    try {
        console.log(`Scanning directory: ${MODES_DIR}`);
        const entries = await fs.readdir(MODES_DIR, { withFileTypes: true });

        const processingPromises = entries.map(async (entry) => {
            if (entry.isDirectory()) {
                const slug = entry.name;
                const mode_dir_path = path.join(MODES_DIR, slug);
                const mode_file_path = path.join(mode_dir_path, `${slug}.mode.md`);

                try {
                    await fs.stat(mode_file_path);
                    console.log(`  Processing file: ${mode_file_path}`);
                    const file_content = await fs.readFile(mode_file_path, 'utf-8');
                    const toml_content = extractTomlFrontmatterRegex(file_content);

                    if (!toml_content) {
                        console.warn(`  Warning: Could not find TOML frontmatter in ${mode_file_path}. Skipping.`);
                        return null;
                    }

                    let data = null;
                    let parseError = null;
                    if (tomlParser) {
                        try {
                            data = tomlParser(toml_content);
                        } catch (e) {
                            if (e instanceof TomlError || (e.name && e.name.includes('Toml'))) {
                                parseError = e;
                            } else {
                                throw e; // Re-throw unexpected errors
                            }
                        }
                    }

                    if (!data && !tomlParser) {
                         console.log(`  Info: Attempting regex fallback for ${mode_file_path}.`);
                         data = parseTomlWithRegex(toml_content, mode_file_path);
                         if (!data) parseError = new Error("Regex parsing failed to extract required fields.");
                    } else if (!data && parseError) {
                         console.warn(`  Warning: Failed to parse TOML in ${mode_file_path}: ${parseError}. Skipping.`);
                         return null;
                    }


                    if (!data) {
                         console.warn(`  Warning: Could not parse TOML data from ${mode_file_path} using any method. Skipping.`);
                         return null;
                    }


                    // Extract required fields + summary - Changed from description
                    const mode_slug_from_toml = data.id;
                    const mode_name = data.name;
                    const system_prompt = data.system_prompt;
                    // Extract summary, provide default if missing - Changed from description
                    const mode_summary = data.summary || "[No summary provided]";

                    // Validate required fields (excluding optional description)
                    const missing_fields = [];
                    if (!mode_slug_from_toml) missing_fields.push('id');
                    if (!mode_name) missing_fields.push('name');
                    if (!system_prompt) missing_fields.push('system_prompt');

                    if (missing_fields.length > 0) {
                        console.warn(`  Warning: Missing required field(s) ${missing_fields.join(', ')} in ${mode_file_path}. Skipping.`);
                        return null;
                    }

                    if (mode_slug_from_toml !== slug) {
                        console.warn(`  Warning: Slug mismatch! Directory is '${slug}' but TOML 'id' is '${mode_slug_from_toml}' in ${mode_file_path}. Using TOML 'id'.`);
                    }

                    // Return the successfully processed mode data including description
                    return {
                        slug: mode_slug_from_toml,
                        name: mode_name,
                        roleDefinition: system_prompt.trim(), // Keep for potential future use
                        summary: mode_summary.trim(), // Add summary - Changed from description
                        groups: DEFAULT_GROUPS // Keep for potential future use
                    };

                } catch (err) {
                    if (err.code === 'ENOENT') {
                         console.log(`  Info: Mode definition file ${mode_file_path} not found. Skipping directory '${slug}'.`);
                    } else {
                        console.error(`  Error processing directory ${slug} or file ${mode_file_path}:`, err);
                    }
                    return null;
                }
            } else {
                return null; // Not a directory
            }
        });

        const results = await Promise.all(processingPromises);
        modes_data = results.filter(result => result !== null);

    } catch (err) {
        console.error(`Error reading modes directory ${MODES_DIR}:`, err);
        return;
    }

    // Define level order (kept from original)
    const levelOrder = [
        'roo', // Special case for roo-commander
        'core',
        'manager',
        'lead',
        'agent',
        'spec',
        'framework',
        'design',
        'data',
        'infra',
        'edge',
        'test',
        'util',
        // Add other levels as needed, 'unknown' will go last
    ];

    // Function to get level and assign order index (kept from original)
    const getSortParams = (slug) => {
        if (slug === 'roo-commander') {
            return { levelIndex: 0, slug }; // Ensure roo-commander is first
        }
        const parts = slug.split('-');
        const level = parts[0];
        const levelIndex = levelOrder.indexOf(level);
        return {
            levelIndex: levelIndex === -1 ? levelOrder.length : levelIndex, // Put unknown levels last
            slug
        };
    };

    // Sort modes by level, then slug (kept from original)
    modes_data.sort((a, b) => {
        const paramsA = getSortParams(a.slug);
        const paramsB = getSortParams(b.slug);

        if (paramsA.levelIndex !== paramsB.levelIndex) {
            return paramsA.levelIndex - paramsB.levelIndex;
        }
        return paramsA.slug.localeCompare(paramsB.slug);
    });

    // --- Generate Markdown Output ---
    const today = new Date();
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, '0'); // Months are 0-indexed
    const day = String(today.getDate()).padStart(2, '0');
    const formattedDate = `${year}-${month}-${day}`;

    let markdownOutput = `+++
id = "kb-available-modes-summary"
title = "Available Modes Summary"
context_type = "summary"
target_audience = ["roo-commander"]
status = "generated"
last_generated = "${formattedDate}"
+++

# Available Modes Summary

This document provides a summary of available specialist modes for delegation.
`;

    // Group modes by level
    const modesByLevel = {};
    modes_data.forEach(mode => {
        const { levelIndex } = getSortParams(mode.slug);
        // Determine level name: 'roo', a name from levelOrder, or 'unknown'
        const levelName = levelIndex === 0 ? 'roo' : (levelIndex < levelOrder.length ? levelOrder[levelIndex] : 'unknown');

        if (!modesByLevel[levelName]) {
            modesByLevel[levelName] = [];
        }
        modesByLevel[levelName].push(mode);
    });

    // Function to capitalize first letter
    const capitalize = (s) => s ? s.charAt(0).toUpperCase() + s.slice(1) : '';

    // Define level titles based on levelOrder keys
    const levelTitles = {
        'roo': 'Roo Commander',
        'core': 'Core Modes',
        'manager': 'Manager Modes',
        'lead': 'Lead Modes',
        'agent': 'Agent Modes',
        'spec': 'Specialist Modes',
        'framework': 'Framework Modes',
        'design': 'Design Modes',
        'data': 'Data Modes',
        'infra': 'Infrastructure Modes',
        'edge': 'Edge Modes',
        'test': 'Testing Modes',
        'util': 'Utility Modes',
        'unknown': 'Other Modes' // Handle unknown level explicitly
    };


    // Add Roo Commander first if present
    if (modesByLevel['roo']) {
        markdownOutput += `\n## ${levelTitles['roo']}\n`;
        modesByLevel['roo'].forEach(mode => {
            // Ensure summary is handled if somehow null/undefined despite default - Changed from description
            const summary = mode.summary || "[No summary provided]";
            markdownOutput += `- **${mode.slug}** (${mode.name}): ${summary}\n`;
        });
        delete modesByLevel['roo']; // Remove from further processing
    }


    // Add other levels in the defined order
    levelOrder.forEach(levelKey => {
        if (levelKey !== 'roo' && modesByLevel[levelKey]) { // Skip 'roo' as it's handled
             const title = levelTitles[levelKey] || `${capitalize(levelKey)} Modes`; // Fallback title just in case
             markdownOutput += `\n## ${title}\n`;
             modesByLevel[levelKey].forEach(mode => {
                 const summary = mode.summary || "[No summary provided]"; // Changed from description
                 markdownOutput += `- **${mode.slug}** (${mode.name}): ${summary}\n`;
             });
             delete modesByLevel[levelKey]; // Remove from further processing
        }
    });

     // Add any remaining 'unknown' level modes at the end
     if (modesByLevel['unknown']) {
         markdownOutput += `\n## ${levelTitles['unknown']}\n`;
         modesByLevel['unknown'].forEach(mode => {
             const summary = mode.summary || "[No summary provided]"; // Changed from description
             markdownOutput += `- **${mode.slug}** (${mode.name}): ${summary}\n`;
         });
     }


    // Write the Markdown file to both locations
    try {
        console.log(`\nWriting mode summary to ${OUTPUT_PATH_KB}...`);
        await fs.writeFile(OUTPUT_PATH_KB, markdownOutput.trim() + '\n', 'utf-8'); // Ensure trailing newline
        console.log(`Successfully generated mode summary ${OUTPUT_PATH_KB}.`);

        console.log(`Writing mode summary copy to ${OUTPUT_PATH_RULES}...`);
        await fs.writeFile(OUTPUT_PATH_RULES, markdownOutput.trim() + '\n', 'utf-8'); // Ensure trailing newline
        console.log(`Successfully generated mode summary copy ${OUTPUT_PATH_RULES}.`);

        console.log(`\nGenerated ${modes_data.length} modes in total.`);
    } catch (err) {
        console.error(`Error writing output file(s):`, err);
    }
}

// --- Execute Main Function ---
// Changed function call
buildModeSummary();