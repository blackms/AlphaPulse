const fs = require('fs');
const path = require('path');
const TOML = require('@iarna/toml');

const modesDir = path.join(__dirname, '.ruru', 'modes');
const targetFilePath = path.join(__dirname, '.ruru', 'docs', 'standards', 'mode_selection_guide.md');
const sectionStartMarker = '## 3. Mode Details';
const sectionEndMarker = '## 4. Maintaining This Guide';

// New concise template
const conciseOutputTemplate = `
---

### \`{slug}\` ({emoji} {name})

*   **Core Purpose:** {corePurpose}
*   **Key Capabilities:** {keyCapabilities}
*   **Hierarchy & Collaboration:**
    *   **Typical Delegators:** {delegators}
    *   **Typical Reports To:** {reportsTo}
    *   **Frequent Collaborators:** {collaborators}

---
`;

// --- Helper Functions ---

/**
 * Recursively finds all files matching a pattern in a directory.
 * @param {string} dirPath - The directory path to search.
 * @param {RegExp} filePattern - The regex pattern to match filenames.
 * @param {string[]} [arrayOfFiles] - Array to accumulate file paths (used internally for recursion).
 * @returns {string[]} - An array of full file paths.
 */
function findFilesRecursive(dirPath, filePattern, arrayOfFiles = []) {
    try {
        const files = fs.readdirSync(dirPath);

        files.forEach(file => {
            const fullPath = path.join(dirPath, file);
            try {
                if (fs.statSync(fullPath).isDirectory()) {
                    findFilesRecursive(fullPath, filePattern, arrayOfFiles);
                } else if (filePattern.test(fullPath)) {
                    arrayOfFiles.push(fullPath);
                }
            } catch (statErr) {
                console.error(`Error stating file/dir ${fullPath}: ${statErr.message}`);
                // Continue with other files
            }
        });
    } catch (readDirErr) {
        console.error(`Error reading directory ${dirPath}: ${readDirErr.message}`);
        // Stop searching this branch
    }
    return arrayOfFiles;
}

/**
 * Extracts TOML frontmatter and Markdown body from TOML MD content.
 * @param {string} content - The file content.
 * @returns {{tomlString: string | null, markdownBody: string | null}}
 */
function extractTomlMd(content) {
    const delimiter = '+++';
    const firstDelimiterIndex = content.indexOf(delimiter);
    if (firstDelimiterIndex === -1) {
        return { tomlString: null, markdownBody: content }; // Assume no TOML if no delimiter
    }
    const secondDelimiterIndex = content.indexOf(delimiter, firstDelimiterIndex + delimiter.length);
    if (secondDelimiterIndex === -1) {
        // Malformed - treat as if no TOML or only TOML
        console.warn("Warning: Found opening '+++' but no closing '+++'. Treating as no valid TOML frontmatter.");
        return { tomlString: null, markdownBody: content };
    }

    const tomlString = content.substring(firstDelimiterIndex + delimiter.length, secondDelimiterIndex).trim();
    const markdownBody = content.substring(secondDelimiterIndex + delimiter.length).trim();

    return { tomlString, markdownBody };
}

/**
 * Infers hierarchy based on slug prefixes and common patterns. Very heuristic.
 * @param {string} slug - The mode slug.
 * @returns {{delegators: string[], reportsTo: string[], collaborators: string[]}}
 */
function inferHierarchy(slug) {
    const parts = slug.split('-');
    const category = parts[0];
    const role = parts.slice(1).join('-');

    let delegators = ['roo-commander', 'prime-coordinator'];
    let reportsTo = ['roo-commander'];
    let collaborators = []; // Default empty

    // General Rules based on category prefix
    if (category === 'lead') {
        delegators = ['roo-commander', 'manager-project', 'core-architect'];
        reportsTo = ['roo-commander', 'manager-project'];
        collaborators = ['core-architect', 'other lead-*', 'relevant spec-*', 'relevant agent-*'];
    } else if (category === 'spec' || category === 'framework' || category === 'design' || category === 'data' || category === 'infra' || category === 'test' || category === 'auth' || category === 'baas' || category === 'cms' || category === 'dev' || category === 'util' || category === 'edge') {
        // Find potential lead based on category (e.g., 'dev-react' -> 'lead-frontend')
        let potentialLead = 'lead-unknown';
        if (['framework', 'design', 'dev', 'util'].includes(category) || slug.includes('frontend') || slug.includes('react') || slug.includes('vue') || slug.includes('angular') || slug.includes('svelte') || slug.includes('nextjs') || slug.includes('remix') || slug.includes('astro')) potentialLead = 'lead-frontend';
        if (slug.includes('backend') || slug.includes('api') || slug.includes('node') || slug.includes('python') || slug.includes('django') || slug.includes('fastapi') || slug.includes('flask') || slug.includes('laravel')) potentialLead = 'lead-backend';
        if (category === 'data' || slug.includes('db') || slug.includes('sql') || slug.includes('mongo')) potentialLead = 'lead-db';
        if (category === 'infra' || slug.includes('docker') || slug.includes('compose') || slug.includes('aws') || slug.includes('azure') || slug.includes('gcp') || slug.includes('cicd')) potentialLead = 'lead-devops';
        if (category === 'test' || slug.includes('qa') || slug.includes('e2e')) potentialLead = 'lead-qa';
        if (category === 'design' && !potentialLead) potentialLead = 'lead-design'; // Catch-all for design
        if (slug.includes('security') || category === 'auth') potentialLead = 'lead-security';


        delegators = [potentialLead, 'manager-project', 'roo-commander'];
        reportsTo = [potentialLead, 'manager-project'];
        collaborators = [potentialLead, 'other spec-*', 'relevant agent-*', 'core-architect'];
    } else if (category === 'agent') {
        delegators = ['roo-commander', 'lead-*', 'manager-project'];
        reportsTo = ['roo-commander', 'relevant lead-*']; // Reporting might depend on context
        collaborators = ['other agent-*', 'spec-*', 'lead-*'];
    } else if (category === 'manager') {
        delegators = ['roo-commander'];
        reportsTo = ['roo-commander'];
        collaborators = ['lead-*', 'core-architect'];
    } else if (category === 'core' || category === 'cloud') { // Architects
        delegators = ['roo-commander'];
        reportsTo = ['roo-commander'];
        collaborators = ['lead-*', 'manager-project', 'other architects'];
    } else if (category === 'prime') {
        delegators = ['roo-commander']; // Special case
        reportsTo = ['roo-commander'];
        collaborators = ['roo-commander'];
    } else if (slug === 'roo-commander') {
        delegators = ['user'];
        reportsTo = ['user'];
        collaborators = ['prime-coordinator', 'lead-*', 'manager-*', 'core-architect'];
    }

    // Remove self from collaborators if present
    collaborators = collaborators.filter(c => c !== slug);
    // Basic formatting
    const formatSlugs = (arr) => arr.length > 0 ? arr.map(s => `\`${s}\``).join(', ') : 'N/A';

    return {
        delegators: formatSlugs([...new Set(delegators)]), // Unique slugs
        reportsTo: formatSlugs([...new Set(reportsTo)]),
        collaborators: formatSlugs([...new Set(collaborators)])
    };
}

/**
 * Generates placeholder text for capabilities and tasks.
 * @param {string} corePurpose - The core purpose string.
 * @returns {{keyCapabilities: string}}
 */
// Modified placeholder generation
function generateConcisePlaceholders(corePurpose, slug, tags = []) {
    // Infer 2-3 key capabilities based on purpose and tags
    let capabilities = new Set();

    // Add from tags first (often more specific) - Capitalize first letter
    tags.slice(0, 2).forEach(tag => {
        if (tag && typeof tag === 'string') {
             capabilities.add(tag.charAt(0).toUpperCase() + tag.slice(1));
        }
    });


    // Add based on core purpose if needed and space allows
    if (capabilities.size < 3 && corePurpose) {
        const lowerPurpose = corePurpose.toLowerCase();
        if (lowerPurpose.includes('implement') || lowerPurpose.includes('develop')) capabilities.add("Implementation");
        if (capabilities.size < 3 && lowerPurpose.includes('design')) capabilities.add("Design");
        if (capabilities.size < 3 && (lowerPurpose.includes('manage') || lowerPurpose.includes('coordinate'))) capabilities.add("Coordination");
        if (capabilities.size < 3 && (lowerPurpose.includes('test') || lowerPurpose.includes('quality'))) capabilities.add("Testing");
        if (capabilities.size < 3 && (lowerPurpose.includes('analyze') || lowerPurpose.includes('research'))) capabilities.add("Analysis");
        if (capabilities.size < 3 && (lowerPurpose.includes('deploy') || lowerPurpose.includes('infrastructure'))) capabilities.add("Infrastructure");
        if (capabilities.size < 3 && (lowerPurpose.includes('write') || lowerPurpose.includes('document'))) capabilities.add("Writing");
        if (capabilities.size < 3 && lowerPurpose.includes('refactor')) capabilities.add("Refactoring");
        if (capabilities.size < 3 && (lowerPurpose.includes('debug') || lowerPurpose.includes('fix'))) capabilities.add("Debugging");
    }

     // Fallback if still empty
    if (capabilities.size === 0) {
        capabilities.add("Core Task Execution");
    }

    // Limit to 3 and join
    const keyCapabilitiesString = Array.from(capabilities).slice(0, 3).join(', ');

    return {
        keyCapabilities: keyCapabilitiesString || "N/A" // Ensure not empty
    };
}


// --- Main Execution ---

let generatedSectionContent = `
## 3. Mode Details

*(Note: The detailed information below, especially under "Hierarchy & Collaboration", is automatically generated by \`build_mode_selection_guide_data.js\` parsing individual \`.mode.md\` files. Manual updates should be avoided.)*
`;

const modeFiles = findFilesRecursive(modesDir, /\.mode\.md$/);

if (modeFiles.length === 0) {
    console.error(`No *.mode.md files found in ${modesDir} or its subdirectories.`);
    process.exit(1);
}

modeFiles.sort(); // Sort for consistent output order

let hadError = false; // Flag to track if any file processing failed

// Read the target file content first
let targetFileContent;
try {
    targetFileContent = fs.readFileSync(targetFilePath, 'utf8');
} catch (readErr) {
    console.error(`Error reading target file ${targetFilePath}: ${readErr.message}`);
    process.exit(1);
}

// Process each mode file to generate the content for section 3
modeFiles.forEach(filePath => {
    // If an error occurred in a previous iteration, skip remaining files
    if (hadError) return;

    try {
        const content = fs.readFileSync(filePath, 'utf8');
        const { tomlString, markdownBody } = extractTomlMd(content);

        if (!tomlString) {
            console.warn(`Skipping ${filePath}: Could not find valid TOML frontmatter.`);
            return; // Skip this file
        }

        const parsedToml = TOML.parse(tomlString);
        const slug = parsedToml.slug || path.basename(filePath, '.mode.md'); // Fallback slug
        const name = parsedToml.name || slug;
        const emoji = parsedToml.emoji || '❓';
        // Keep raw tags for placeholder generation, don't format them here
        const rawTags = Array.isArray(parsedToml.tags) ? parsedToml.tags : [];

        const summary = parsedToml.summary || "No summary provided in TOML.";
        // Use summary directly as core purpose, removing trailing period if exists
        const corePurpose = summary.replace(/\.$/, '');

        const hierarchy = inferHierarchy(slug);
        // Use the new concise placeholder generator, passing raw tags
        const placeholders = generateConcisePlaceholders(summary, slug, rawTags);

        // Use the new concise template
        const formattedSection = conciseOutputTemplate
            .replace('{slug}', slug)
            .replace('{emoji}', emoji)
            .replace('{name}', name)
            .replace('{corePurpose}', corePurpose || "N/A")
            .replace('{keyCapabilities}', placeholders.keyCapabilities) // Use the new concise capabilities string
            // Removed: commonTasks, tags, selectionHints, whenNotToUse
            .replace('{delegators}', hierarchy.delegators)
            .replace('{reportsTo}', hierarchy.reportsTo)
            .replace('{collaborators}', hierarchy.collaborators);

        generatedSectionContent += formattedSection;

    } catch (parseErr) {
        console.error(`\n❌ Error processing file ${filePath}: ${parseErr.message}`);
        console.error("🛑 Aborting update due to error. Target file not modified.");
        process.exit(1); // Exit immediately on parsing error
    }
});

// This block is now technically unreachable due to the exit in the catch block,
// but kept for clarity in case the logic changes later.
// if (hadError) {
//     console.error("\n🛑 Aborting update due to errors processing .mode.md files. Target file not modified.");
//     process.exit(1);
// }

// Find the section to replace in the original target file content
const startIndex = targetFileContent.indexOf(sectionStartMarker);
const endIndex = targetFileContent.indexOf(sectionEndMarker);

if (startIndex === -1 || endIndex === -1 || startIndex >= endIndex) {
    console.error(`Error: Could not find section markers "${sectionStartMarker}" and "${sectionEndMarker}" in ${targetFilePath}.`);
    process.exit(1);
}

// Construct the new content
const contentBefore = targetFileContent.substring(0, startIndex);
const contentAfter = targetFileContent.substring(endIndex);
// Ensure proper spacing: content before + generated section + two newlines + content after
const newFileContent = contentBefore + generatedSectionContent.trim() + '\n\n' + contentAfter;

// Write the updated content back to the file
try {
    fs.writeFileSync(targetFilePath, newFileContent, 'utf8');
    console.log(`✅ Successfully updated ${targetFilePath}`);
} catch (writeErr) {
    console.error(`Error writing updated content to ${targetFilePath}: ${writeErr.message}`);
    process.exit(1);
}