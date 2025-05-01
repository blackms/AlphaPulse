#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync, exec } = require('child_process');
const process = require('process');
const yaml = require('js-yaml'); // Need js-yaml for frontmatter parsing

// --- Argument Parsing ---
let mode = 'process'; // 'process' or 'finalize'
let inputFile = '';
let libraryName = '';
let outputBaseArg = 'kb'; // Default output base directory

// Check for --finalize-index mode first
if (process.argv.includes('--finalize-index')) {
    mode = 'finalize';
    const finalizeIndex = process.argv.indexOf('--finalize-index');
    if (process.argv.length > finalizeIndex + 1) {
        libraryName = process.argv[finalizeIndex + 1];
    } else {
        console.error('Error: Missing library_name after --finalize-index');
        process.exit(1);
    }
    // For finalize, the optional path is the direct library KB dir
    if (process.argv.length > finalizeIndex + 2) {
        outputBaseArg = process.argv[finalizeIndex + 2]; // This will be the full path like .ruru/modes/X/kb/Y
    } else {
        outputBaseArg = path.join('kb', libraryName); // Default if not provided
    }

} else if (process.argv.length >= 4) { // Process mode (input file, library name, optional output base)
    mode = 'process';
    inputFile = process.argv[2];
    libraryName = process.argv[3];
    if (process.argv.length > 4) {
        outputBaseArg = process.argv[4]; // This is the base like .ruru/modes/X/kb
    }
} else {
    console.error('Usage:');
    console.error('  node create_kb_from_json.js <input.json> <library_name> [output_base_dir]');
    console.error('  node create_kb_from_json.js --finalize-index <library_name> [library_kb_dir]');
    process.exit(1);
}


if (mode === 'process' && !fs.existsSync(inputFile)) {
  console.error(`Error: Input file not found at ${inputFile}`);
  process.exit(1);
}

if (!libraryName || !/^[a-zA-Z0-9_-]+$/.test(libraryName)) {
    console.error(`Error: Invalid library name "${libraryName}". Use only alphanumeric characters, underscores, or hyphens.`);
    process.exit(1);
}
// --- End Argument Parsing ---


// Use the parsed argument for the output directory
// For 'process' mode, outputBaseArg is the base (e.g., .ruru/modes/X/kb), libraryName is added.
// For 'finalize' mode, outputBaseArg is the full library path (e.g., .ruru/modes/X/kb/Y)
const outputDir = mode === 'process' ? path.join(outputBaseArg, libraryName) : outputBaseArg;
const topLevelIndexFile = path.join(outputDir, 'index.json'); // Final top-level index for the library

// --- Configuration for Categorization ---
// ... (categoryKeywords definition remains the same) ...
const categoryKeywords = {
  'guide': ['guide', 'introduction', 'essentials', 'components', 'reusability', 'composition', 'reactivity', 'rendering', 'list', 'event', 'form', 'lifecycle', 'watchers', 'template', 'provide', 'inject', 'async', 'suspense', 'transitions', 'animation', 'scaling', 'routing', 'state management', 'testing', 'ssr', 'typescript', 'composition api', 'options api', 'built-ins', 'sfc', 'performance', 'security', 'deployment', 'best practices', 'style', 'start', 'usage', 'basic', 'core', 'concept', 'feature'],
  'api': ['api', 'reference', 'application', 'global', 'component instance', 'function', 'directive', 'special attribute', 'ref', 'reactive', 'computed', 'effect', 'defineprops', 'defineemits', 'defineslots', 'defineexpose', 'defineoptions', 'definemodel', 'createapp', 'nexttick', 'set', 'del', 'version', 'utility', 'interface', 'type', 'props', 'methods', 'events', 'slots'],
  'tutorial': ['tutorial', 'step', 'learn', 'example walkthrough'],
  'cookbook': ['cookbook', 'recipe', 'pattern'],
  'examples': ['example', 'demo'],
  'migration': ['migration', 'upgrade', 'v2', 'v3', 'v4', 'v5'], // Add more versions as needed
  'concepts': ['concept', 'mechanism', 'deep dive', 'architecture', 'philosophy'],
  'internals': ['internals', 'source', 'contribution', 'contributing'],
  'about': ['about', 'team', 'faq', 'community', 'release', 'changelog', 'sponsor', 'partner'],
  'config': ['config', 'configuration', 'options', 'settings', 'setup', 'install', 'installation'],
};

// Basic English stop words list - KEEP for fallback keyword generation if needed
// ... (stopWords definition remains the same) ...
const stopWords = new Set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could',
    'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s',
    'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m',
    'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t',
    'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t',
    'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
    'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too',
    'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t',
    'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s',
    'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself',
    'yourselves', 'use', 'using', 'get', 'set', 'add', 'new', 'via'
]);


function determineCategory(pageTitle, tags = []) { // Accept tags for potentially better categorization
  if (!pageTitle) return 'misc';
  const lowerTitle = pageTitle.toLowerCase();
  const lowerTags = tags.map(t => t.toLowerCase());

  // Prioritize tags if available
  for (const category in categoryKeywords) {
    if (categoryKeywords[category].some(keyword => lowerTags.includes(keyword))) {
      return category;
    }
  }
  // Fallback to title
  for (const category in categoryKeywords) {
    if (categoryKeywords[category].some(keyword => lowerTitle.includes(keyword))) {
      return category;
    }
  }
  const titleParts = pageTitle.split(':');
  if (titleParts.length > 1) {
      const potentialCat = titleParts[0].trim().toLowerCase();
      if (Object.keys(categoryKeywords).includes(potentialCat)) {
          return potentialCat;
      }
  }
  return 'misc'; // Default category
}
// --- End Configuration ---


// Function to sanitize filenames
// ... (sanitizeFilename remains the same) ...
function sanitizeFilename(name) {
  if (!name) return 'untitled_page.md';
  return name
    .toLowerCase()
    .replace(/[:\/\\?*|"<>&\s+]+/g, '_') // Replace spaces and invalid chars with underscore
    .replace(/_+/g, '_') // Collapse multiple underscores
    .replace(/^_|_$/g, '') // Trim leading/trailing underscores
    .substring(0, 100) + '.md';
}

// Function to parse YAML frontmatter
function parseFrontmatter(content) {
    const fmRegex = /^---\s*([\s\S]*?)\s*---/;
    const match = content.match(fmRegex);
    if (match && match[1]) {
        try {
            const frontmatter = yaml.load(match[1]);
            const contentWithoutFm = content.substring(match[0].length).trim();
            return { frontmatter: frontmatter || {}, content: contentWithoutFm };
        } catch (e) {
            console.warn(`Warning: Could not parse frontmatter. Error: ${e.message}`);
            return { frontmatter: {}, content }; // Return original content if parsing fails
        }
    }
    return { frontmatter: {}, content }; // No frontmatter found
}


// --- Main Logic ---
// ... (Main logic remains the same) ...
try {
  if (mode === 'process') {
    processInputFile();
  } else if (mode === 'finalize') {
    finalizeIndex();
  }
} catch (error) {
  console.error(`An error occurred during ${mode} for library "${libraryName}":`, error);
  process.exit(1);
}
// --- End Main Logic ---


// --- Function for Processing Input JSON ---
// This function largely remains the same, as it generates the initial markdown files.
// The key change is that it NO LONGER needs to collect keywords itself for the top-level index.
// It still updates the category index, but sorting and final keyword collection happen in finalizeIndex.
function processInputFile() {
  // Ensure base output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
    console.log(`Created base directory: ${outputDir}`);
  }

  // Read the input JSON file
  console.log(`Reading input file: ${inputFile}`);
  const rawData = fs.readFileSync(inputFile, 'utf-8');
  let snippets = [];
   try {
       snippets = JSON.parse(rawData);
       if (!Array.isArray(snippets)) {
           console.error(`Error: Input file ${inputFile} does not contain a JSON array.`);
           process.exit(1);
       }
   } catch (parseError) {
       console.error(`Error parsing JSON from ${inputFile}: ${parseError.message}`);
       process.exit(1);
   }
  console.log(`Found ${snippets.length} snippets in the input file.`);

  // Group snippets by pageTitle first
  const snippetsByPage = {};
  snippets.forEach(snippet => {
    const pageTitle = snippet.pageTitle || 'Untitled Page';
    if (!snippetsByPage[pageTitle]) {
      snippetsByPage[pageTitle] = [];
    }
    snippetsByPage[pageTitle].push(snippet);
  });

  let totalFilesCreated = 0;
  const pageTitles = Object.keys(snippetsByPage);
  console.log(`Found ${pageTitles.length} unique page titles to process from this input file.`);

  // Process each page group to create markdown files and update category indexes
  for (const pageTitle of pageTitles) {
    const pageSnippets = snippetsByPage[pageTitle];
    // Determine category based only on title initially for directory placement
    const category = determineCategory(pageTitle);
    const categoryDir = path.join(outputDir, category);
    const categoryIndexFile = path.join(categoryDir, 'index.json');

    // Ensure category directory exists
    if (!fs.existsSync(categoryDir)) {
      fs.mkdirSync(categoryDir, { recursive: true });
    }

    const filename = sanitizeFilename(pageTitle);
    const relativeFilePathInCategory = filename;
    const absoluteFilePath = path.join(categoryDir, filename);

    let snippetTitles = [];
    let pageMarkdownContent = '';

    pageSnippets.forEach((snippet, index) => {
      if (!snippet.codeTitle || !snippet.codeDescription || !snippet.codeList) {
        console.warn(`Skipping snippet within page "${pageTitle}" due to missing fields:`, snippet.codeId || `Index ${index}`);
        return;
      }
      snippetTitles.push(snippet.codeTitle);

      pageMarkdownContent += `## ${snippet.codeTitle}\n\n`;
      pageMarkdownContent += `**Description:** ${snippet.codeDescription}\n\n`;
      snippet.codeList.forEach(codeItem => {
        pageMarkdownContent += `\`\`\`${codeItem.language || ''}\n${codeItem.code}\n\`\`\`\n\n`;
      });
      pageMarkdownContent += `---\n\n`;
    });

    // Generate file summary (for the top of the markdown file)
    const fileSummaryDetailed = `This file covers topics related to "${pageTitle}". Key snippets include: ${snippetTitles.slice(0, 5).join(', ')}${snippetTitles.length > 5 ? '...' : ''}.`;
    // NOTE: We are NOT adding frontmatter here. That's done by the AI refinement step.
    // We still add the basic summary for now.
    const finalMarkdownContent = `# ${pageTitle}\n\n**Summary:** ${fileSummaryDetailed}\n\n---\n\n${pageMarkdownContent.trim()}`;

    // Generate concise summary for the category index
    const indexSummary = `${pageTitle} (Covers: ${snippetTitles.slice(0, 3).join(', ')}${snippetTitles.length > 3 ? '...' : ''})`;

    try {
      // Write the markdown file
      fs.writeFileSync(absoluteFilePath, finalMarkdownContent);
      totalFilesCreated++;

      // --- Update Category Index File (Append Logic) ---
      let categoryIndexData = [];
      if (fs.existsSync(categoryIndexFile)) {
        try {
          categoryIndexData = JSON.parse(fs.readFileSync(categoryIndexFile, 'utf-8'));
          if (!Array.isArray(categoryIndexData)) categoryIndexData = [];
        } catch (e) {
          console.warn(`Warning: Could not parse existing category index ${categoryIndexFile}. Overwriting.`);
          categoryIndexData = [];
        }
      }

      const existingEntryIndex = categoryIndexData.findIndex(entry => entry.filename === relativeFilePathInCategory);
      const newEntry = {
        filename: relativeFilePathInCategory,
        title: pageTitle,
        summary: indexSummary,
        snippet_count: snippetTitles.length
      };
      if (existingEntryIndex > -1) {
        categoryIndexData[existingEntryIndex] = newEntry;
      } else {
        categoryIndexData.push(newEntry);
      }

      fs.writeFileSync(categoryIndexFile, JSON.stringify(categoryIndexData, null, 2));
      // --- End Update Category Index File ---

    } catch (writeError) {
        console.error(`Error writing file ${absoluteFilePath} or its category index:`, writeError);
    }
  }
  console.log(`\nProcessed ${totalFilesCreated} markdown files from ${inputFile}. Category indexes updated.`);
}
// --- End Function for Processing Input JSON ---


// --- Function for Finalizing Top-Level Index ---
// *** UPDATED to read frontmatter tags ***
function finalizeIndex() {
  console.log(`Finalizing indexes for library "${libraryName}"...`);
  const topLevelIndexData = [];
  if (!fs.existsSync(outputDir)) {
      console.error(`Error: Library directory ${outputDir} not found for finalization.`);
      process.exit(1);
  }
  const categoryDirs = fs.readdirSync(outputDir, { withFileTypes: true })
                         .filter(dirent => dirent.isDirectory())
                         .map(dirent => dirent.name);

  console.log(`Found categories: ${categoryDirs.join(', ')}`);

  for (const category of categoryDirs) {
    const categoryDir = path.join(outputDir, category);
    const categoryIndexFile = path.join(categoryDir, 'index.json');
    const categoryKeywordsSet = new Set(); // Collect keywords per category from frontmatter

    if (fs.existsSync(categoryIndexFile)) {
      try {
        let categoryIndexData = JSON.parse(fs.readFileSync(categoryIndexFile, 'utf-8'));
        if (Array.isArray(categoryIndexData)) {
          // Sort the category index now
          categoryIndexData.sort((a, b) => a.title.localeCompare(b.title));
          fs.writeFileSync(categoryIndexFile, JSON.stringify(categoryIndexData, null, 2)); // Write sorted index back
          console.log(` - Sorted and finalized index for category "${category}" (${categoryIndexData.length} files)`);

          // Collect keywords from frontmatter tags within this category's files
          for (const fileEntry of categoryIndexData) {
              const mdFilePath = path.join(categoryDir, fileEntry.filename);
              if (fs.existsSync(mdFilePath)) {
                  const mdContent = fs.readFileSync(mdFilePath, 'utf-8');
                  const { frontmatter } = parseFrontmatter(mdContent);
                  if (frontmatter && Array.isArray(frontmatter.tags)) {
                      frontmatter.tags.forEach(tag => categoryKeywordsSet.add(tag.toLowerCase()));
                  }
                  // Optionally re-determine category based on tags for more accuracy
                  // const refinedCategory = determineCategory(frontmatter.title || fileEntry.title, frontmatter.tags || []);
                  // if (refinedCategory !== category) { /* Handle potential category change? Maybe move file? Complex. */ }
              }
          }

          topLevelIndexData.push({
            category: category,
            index_file: path.join(category, 'index.json').replace(/\\/g, '/'),
            file_count: categoryIndexData.length,
            keywords: Array.from(categoryKeywordsSet).slice(0, 20) // Use frontmatter tags, limit 20
          });
        } else {
          console.warn(`Warning: Category index ${categoryIndexFile} is not a valid array. Skipping category.`);
        }
      } catch (e) {
        console.warn(`Warning: Could not read, parse, or write category index ${categoryIndexFile}. Skipping category. Error: ${e.message}`);
      }
    } else {
      console.warn(`Warning: Category index not found at ${categoryIndexFile}. Skipping category.`);
    }
  }

  // Write the top-level index file
  console.log(`Writing final top-level index file: ${topLevelIndexFile}`);
  topLevelIndexData.sort((a, b) => a.category.localeCompare(b.category));
  fs.writeFileSync(topLevelIndexFile, JSON.stringify(topLevelIndexData, null, 2));
  console.log(`Top-level index for "${libraryName}" finalized using frontmatter tags.`);
}
// --- End Function for Finalizing Top-Level Index ---
