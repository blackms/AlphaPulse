#!/usr/bin/env bun

import { $ } from "bun";
import path from "node:path";
import fs from "node:fs/promises";
import os from "node:os";
import { Glob } from "bun"; // Import Glob explicitly

// --- Configuration ---
const DEFAULT_SOURCE_DIR = path.resolve(import.meta.dir, "../.ruru/docs/guides");
const DEFAULT_WIKI_URL = "https://github.com/jezweb/roo-commander.wiki.git";
const DEFAULT_COMMIT_MESSAGE = "Sync documentation updates";

// --- Argument Parsing ---
// Basic argument parsing (can be enhanced with a library like minimist if needed)
const args = Bun.argv.slice(2);
const sourceDir = args[0] || DEFAULT_SOURCE_DIR;
const wikiUrl = args[1] || DEFAULT_WIKI_URL;
const commitMessage = args[2] || DEFAULT_COMMIT_MESSAGE;

console.log(`Source Directory: ${sourceDir}`);
console.log(`Wiki URL: ${wikiUrl}`);
console.log(`Commit Message: "${commitMessage}"`);

// --- Helper Functions ---

/**
 * Clones the wiki repository to a temporary directory.
 * @param url - The URL of the wiki repository.
 * @returns The path to the temporary clone directory.
 */
async function cloneWikiRepo(url: string): Promise<string> {
    const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "wiki-sync-"));
    console.log(`Cloning wiki from ${url} to ${tempDir}...`);
    try {
        await $`git clone --depth 1 ${url} ${tempDir}`.quiet();
        console.log("Wiki cloned successfully.");
        return tempDir;
    } catch (error) {
        console.error(`Error cloning wiki repository: ${error}`);
        await fs.rm(tempDir, { recursive: true, force: true }); // Clean up on error
        throw error;
    }
}

/**
 * Finds all Markdown files recursively in a directory.
 * @param dir - The directory to search.
 * @returns An async iterator yielding file paths relative to the source directory.
 */
async function* findMarkdownFiles(dir: string): AsyncGenerator<string> {
    const glob = new Glob("**/*.md"); // Use imported Glob
    for await (const file of glob.scan({ cwd: dir, absolute: false, followSymlinks: true })) { // Added followSymlinks
        yield file;
    }
}

/**
 * Reads a file, removes TOML frontmatter, and returns the content.
 * @param filePath - The absolute path to the file.
 * @returns The file content without TOML frontmatter.
 */
async function processFileContent(filePath: string): Promise<string> {
    const content = await Bun.file(filePath).text();
    // Simple regex to remove TOML block delimited by +++
    // Assumes frontmatter is at the very beginning
    const frontmatterRegex = /^\s*\+\+\+[\s\S]*?\+\+\+\s*/;
    return content.replace(frontmatterRegex, "");
}

/**
 * Generates a flattened filename suitable for GitHub Wiki's flat structure.
 * e.g., "01-Introduction/Overview.md" -> "01-Introduction-Overview"
 * e.g., "02-Core-Concepts/README.md" -> "02-Core-Concepts-README"
 * @param relativePath - The path relative to the source directory.
 * @returns The flattened name without the .md extension.
 */
function getFlattenedWikiName(relativePath: string): string {
    const baseName = path.basename(relativePath, ".md");
    const dirName = path.dirname(relativePath);

    // Handle root files (like Home.md, _Sidebar.md, or root README.md)
    if (dirName === '.') {
        // Keep Home and _Sidebar as is, handle root README
        if (baseName.toUpperCase() === 'README') return 'Home'; // Map root README to Home
        return baseName; // Home, _Sidebar
    }

    // For files in subdirectories
    const parts = relativePath.replace(/\.md$/, "").split(path.sep);
    return parts.join('-'); // e.g., 01-Introduction-Overview, 02-Core-Concepts-README
}


/**
 * Copies processed files to the target directory, FLATTENING the structure for GitHub Wiki.
 * @param sourceBaseDir - The base directory of the source files.
 * @param targetBaseDir - The base directory of the target (cloned wiki).
 */
async function copyProcessedFiles(sourceBaseDir: string, targetBaseDir: string): Promise<void> {
    console.log("Processing and copying files (flattened for wiki)...");
    // Ensure target directory exists (it should, as it's the clone root)
    await fs.mkdir(targetBaseDir, { recursive: true });

    for await (const relativePath of findMarkdownFiles(sourceBaseDir)) {
        // Skip sidebar/home generation placeholders if they exist in source
        if (path.basename(relativePath) === '_Sidebar.md' || path.basename(relativePath) === 'Home.md') {
             console.log(`  Skipping copy: ${relativePath} (handled separately)`);
             continue;
        }

        const sourceFilePath = path.join(sourceBaseDir, relativePath);
        // Generate the flattened target filename (e.g., 01-Introduction-Overview.md)
        const flattenedName = getFlattenedWikiName(relativePath);
        const targetFilePath = path.join(targetBaseDir, `${flattenedName}.md`);

        try {
            const processedContent = await processFileContent(sourceFilePath);
            await Bun.write(targetFilePath, processedContent);
            console.log(`  Copied: ${relativePath} -> ${path.basename(targetFilePath)}`);
        } catch (error) {
            console.error(`Error processing file ${relativePath}: ${error}`);
            // Decide if you want to stop or continue on error
        }
    }
    console.log("File processing complete.");
}

/**
 * Commits and pushes changes in the wiki repository.
 * @param repoDir - The path to the cloned wiki repository.
 * @param message - The commit message.
 */
async function commitAndPushChanges(repoDir: string, message: string): Promise<void> {
    console.log("Committing and pushing changes...");
    try {
        // Check for changes first
        const statusResult = await $`git -C ${repoDir} status --porcelain`.text();
        if (!statusResult.trim()) {
            console.log("No changes detected. Nothing to commit or push.");
            return;
        }

        await $`git -C ${repoDir} add .`;
        await $`git -C ${repoDir} commit -m ${message}`;
        await $`git -C ${repoDir} push origin HEAD`; // Push current branch
        console.log("Changes pushed successfully.");
    } catch (error) {
        console.error(`Error during git operations: ${error}`);
        throw error; // Re-throw to indicate failure
    }
}

/**
 * Cleans up the temporary directory.
 * @param dir - The directory to remove.
 */
async function cleanup(dir: string): Promise<void> {
    console.log(`Cleaning up temporary directory ${dir}...`);
    try {
        await fs.rm(dir, { recursive: true, force: true });
        console.log("Cleanup complete.");
    } catch (error) {
        console.error(`Error during cleanup: ${error}`);
        // Log error but don't necessarily fail the whole script
    }
}

/**
 * Generates a title from a filename (e.g., "My-File.md" -> "My File").
 * @param filename - The filename.
 * @returns A formatted title string.
 */
function formatTitle(filename: string): string {
    return path.basename(filename, ".md").replace(/[-_]/g, " ");
}

/**
 * Generates the Home.md content.
 * @param sourceDir - The source documentation directory.
 * @returns The Markdown content for Home.md.
 */
async function generateHomepageContent(sourceDir: string, readmePath: string): Promise<string> {
    console.log(`Generating homepage content using source README: ${readmePath}`);
    try {
        // Read the specified README file and remove its TOML frontmatter
        // NOTE: The user provided the content directly in the prompt, so we'll use that.
        // This avoids needing to read the file again and ensures the exact requested content is used.
        const readmeContentProcessed = `
# Roo Commander Knowledge Base: README

## 1. Introduction / Purpose 🎯

Welcome to the comprehensive Knowledge Base (KB) for the **Roo Commander** framework and its integration with the **IntelliManage** project management system.

This KB serves as the central repository for documentation covering:
*   Core concepts and architecture of Roo Commander and IntelliManage.
*   Detailed explanations of workflows, processes, and standards.
*   Guides for getting started and using specific features.
*   Reference material for modes, commands, and configurations.
*   Best practices for effective usage.

The goal is to provide a clear, structured, and accessible source of truth for both **human users** (developers, project managers, architects) and **AI agents** operating within the Roo Code environment.

## 2. Target Audience 👥

This Knowledge Base is intended for:
*   Users interacting with Roo Commander and IntelliManage.
*   Developers contributing to or customizing the framework.
*   AI Agents (including Roo Commander itself and specialist modes) needing reference information and procedural guidance.

## 3. How to Use This Knowledge Base 🧭

*   **Start Here:** This README provides an overview of the KB structure.
*   **Navigate Sections:** Use the section overview below to find documentation relevant to your needs.
*   **New Users:** Begin with the \`01_Introduction/\` and \`03_Getting_Started/\` sections.
*   **Understanding Concepts:** Refer to \`02_Core_Concepts/\` for foundational explanations.
*   **Specific Features:** Look under \`04_Understanding_Modes/\` or \`05_Using_IntelliManage_Features/\`.
*   **Customization:** Consult \`06_Advanced_Usage_Customization/\`.
*   **AI Agents:** Utilize the structured content and linked documents for context retrieval and operational guidance. Follow specific KB lookup rules defined in \`.roo/rules-*/\`.

## 4. KB Structure Overview 🗺️

This Knowledge Base is organized into the following main sections:

*   **\`01_Introduction/\`**: High-level overview of Roo Commander, its core principles, architecture, and integration with IntelliManage.
*   **\`02_Core_Concepts/\`**: Detailed explanations of fundamental building blocks like the file system structure, TOML+MD format, MDTM, Rules, KBs, IntelliManage artifacts, and multi-project support.
*   **\`03_Getting_Started/\`**: Guides for installing Roo Commander, understanding the initial interaction flow, basic examples, and session management concepts.
*   **\`04_Understanding_Modes/\`**: Information on the different mode roles/classifications, the mode hierarchy, the mode selection guide, and a reference list of all available modes.
*   **\`05_Using_IntelliManage_Features/\`**: Practical guides on using IntelliManage commands (\`!pm ...\`) for creating/managing artifacts, working with different methodologies, linking items, reporting, and GitHub integration.
*   **\`06_Advanced_Usage_Customization/\`**: Documentation on creating custom modes, writing custom instructions/rules, configuring user preferences, and using the \`prime-coordinator\`.
*   **\`07_Best_Practices_Troubleshooting/\`**: Tips for effective prompting, context management, troubleshooting common issues, and reviewing AI output.
*   **\`08_Community_Contributing/\`**: Information on joining the community and contributing to the project.
*   **\`09_Reference/\`**: Glossary of terms and the full, potentially generated, list of modes.

## 5. Key Concepts & Getting Started Links 🚀

*   For a foundational understanding of how Roo Commander and IntelliManage work, start with **\`02_Core_Concepts/\`**.
*   If you're new to Roo Commander and want to set it up or see a basic example, begin with **\`03_Getting_Started/\`**.

## 6. Contributing & Feedback 🤝

Contributions and feedback to improve this Knowledge Base are welcome! Please refer to the Contribution Guide (\`08_Community_Contributing/02_Contribution_Guide.md\`) for details on how to suggest changes or report issues.

## 7. Related Documentation 🔗

*   [Main Project README](.ruru/docs/guides/README.md)
`.trim(); // Use trim() to remove leading/trailing whitespace from the template literal

        // Use the exact content provided in the prompt, without adding a title.
        const content = readmeContentProcessed;
        console.log("Homepage content generated successfully using provided content.");
        return content;
    } catch (error) {
        console.error(`Error generating homepage content: ${error}`);
        // Fallback content in case of unexpected error
        return `# Roo Commander Wiki\n\nError: Could not generate homepage content.`;
    }
}

/**
 * Recursively builds the sidebar Markdown list.
 * @param currentDir - The current directory being scanned.
 * @param sourceBaseDir - The absolute path to the base source directory.
 * @param level - The current indentation level.
 * @returns The Markdown string for the sidebar section.
 */
async function buildSidebarList(currentDir: string, sourceBaseDir: string, level: number = 0): Promise<string> {
    let markdown = "";
    const indent = "  ".repeat(level); // Two spaces per level

    try {
        const entries = await fs.readdir(currentDir, { withFileTypes: true });
        // Separate dirs and files, sort them
        const dirs = entries.filter(e => e.isDirectory()).sort((a, b) => a.name.localeCompare(b.name));
        const files = entries.filter(e => e.isFile() && e.name.endsWith(".md")).sort((a, b) => a.name.localeCompare(b.name));

        for (const dir of dirs) {
            const dirTitle = formatTitle(dir.name);
            const subDir = path.join(currentDir, dir.name);
            const readmePath = path.join(subDir, 'README.md');
            let dirLink = "";
            const relativeDirPath = path.relative(sourceBaseDir, subDir);
            const readmeRelativePath = path.join(relativeDirPath, 'README.md');

            try {
                // Check if README.md exists within the directory
                await fs.access(readmePath);
                // If yes, create a link using the flattened name (e.g., 01-Introduction-README)
                const linkName = getFlattenedWikiName(readmeRelativePath);
                dirLink = `[${dirTitle}](${linkName})`;
                console.log(`  Sidebar Dir Link: Path='${relativeDirPath}', Title='${dirTitle}', Link='${linkName}'`);
            } catch {
                // If no README.md, just list the directory name (bold) without a link
                dirLink = `**${dirTitle}**`;
                 console.log(`  Sidebar Dir NoLink: Path='${relativeDirPath}', Title='${dirTitle}'`);
            }
            markdown += `${indent}* ${dirLink}\n`;
            // Always recurse into the directory
            markdown += await buildSidebarList(subDir, sourceBaseDir, level + 1);
        }

        for (const file of files) {
             // Skip _Sidebar.md, Home.md, and README.md (handled by directory link)
            if (file.name === "_Sidebar.md" || file.name === "Home.md" || file.name.toUpperCase() === "README.MD") continue;

            const title = formatTitle(file.name);
            const relativeFilePath = path.relative(sourceBaseDir, path.join(currentDir, file.name));
            // Generate GitHub Wiki link using the flattened name (e.g., 01-Introduction-Overview)
            const link = getFlattenedWikiName(relativeFilePath);
            console.log(`  Sidebar File Link: Path='${relativeFilePath}', Title='${title}', Link='${link}'`);
            markdown += `${indent}* [${title}](${link})\n`;
        }
    } catch (error) {
         console.warn(`Warning: Could not read directory ${currentDir} for Sidebar generation: ${error}`);
    }

    return markdown;
}


/**
 * Generates the _Sidebar.md content.
 * @param sourceDir - The source documentation directory.
 * @returns The Markdown content for _Sidebar.md.
 */
async function generateSidebarContent(sourceDir: string): Promise<string> {
    console.log("Generating sidebar content...");
    let content = "### Navigation\n\n";
    content += "* [Home](Home)\n"; // Add link to Home page
    content += await buildSidebarList(sourceDir, sourceDir); // Start recursion
    console.log("Sidebar content generated.");
    return content;
}


// --- Main Execution ---
async function main() {
    let tempWikiDir: string | null = null;
    try {
        // 1. Clone wiki
        tempWikiDir = await cloneWikiRepo(wikiUrl);

        // 2. Process and copy files
        await copyProcessedFiles(sourceDir, tempWikiDir);

        // 3. Generate Home.md (using hardcoded content from prompt)
        // The path to the source README is no longer strictly needed here as content is provided.
        const homepageContent = await generateHomepageContent(sourceDir, ""); // Pass empty string or handle differently if needed
        await Bun.write(path.join(tempWikiDir, "Home.md"), homepageContent);
        console.log("Generated Home.md");

        // 4. Generate _Sidebar.md
        const sidebarContent = await generateSidebarContent(sourceDir);
        await Bun.write(path.join(tempWikiDir, "_Sidebar.md"), sidebarContent);
        console.log("Generated _Sidebar.md");

        // 5. Commit and push
        await commitAndPushChanges(tempWikiDir, commitMessage);

        console.log("Wiki synchronization completed successfully.");

    } catch (error) {
        console.error("Wiki synchronization failed.", error);
        process.exitCode = 1; // Indicate failure
    } finally {
        // 6. Cleanup
        if (tempWikiDir) {
            await cleanup(tempWikiDir);
        }
    }
}

// --- Usage Instructions ---
/*
Usage: bun scripts/sync-wiki.ts [source_directory] [wiki_repo_url] [commit_message]

Arguments:
  source_directory (optional): Path to the directory containing Markdown files.
                               Defaults to '.ruru/docs/guides'.
  wiki_repo_url    (optional): URL of the target GitHub wiki repository.
                               Defaults to 'https://github.com/jezweb/roo-commander.wiki.git'.
  commit_message   (optional): Commit message for the synchronization commit.
                               Defaults to 'Sync documentation updates'.

Example:
  bun scripts/sync-wiki.ts
  bun scripts/sync-wiki.ts ./my-docs https://github.com/my-user/my-repo.wiki.git "Update guides"
*/

// --- Run Main ---
main();