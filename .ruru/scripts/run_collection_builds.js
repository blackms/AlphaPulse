const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const COLLECTIONS_CONFIG_PATH = path.resolve(__dirname, 'build_collections.json');
const ROOMODES_PATH = path.resolve(__dirname, '..', '.roomodes'); // Project root
const BUILD_SCRIPT_PATH = path.resolve(__dirname, 'create_build.js');
const BUILD_OUTPUT_DIR = path.resolve(__dirname, '..', 'build');

const DEFAULT_ROOMODES_CONTENT = 'all';

/**
 * Reads and parses the build collections configuration file.
 * @returns {object|null} Parsed configuration object or null on error.
 */
function readCollectionsConfig() {
    console.log(`Reading collections config from: ${COLLECTIONS_CONFIG_PATH}`);
    try {
        if (!fs.existsSync(COLLECTIONS_CONFIG_PATH)) {
            console.error(`Error: Collections config file not found at ${COLLECTIONS_CONFIG_PATH}`);
            return null;
        }
        const rawData = fs.readFileSync(COLLECTIONS_CONFIG_PATH, 'utf8');
        return JSON.parse(rawData);
    } catch (error) {
        console.error(`Error reading or parsing ${COLLECTIONS_CONFIG_PATH}:`, error);
        return null;
    }
}

/**
 * Writes content to the .roomodes file.
 * @param {string} content - The content to write.
 * @returns {boolean} True on success, false on error.
 */
function writeRoomodes(content) {
    console.log(`Writing to ${ROOMODES_PATH}`);
    try {
        fs.writeFileSync(ROOMODES_PATH, content, 'utf8');
        return true;
    } catch (error) {
        console.error(`Error writing to ${ROOMODES_PATH}:`, error);
        return false;
    }
}

/**
 * Deletes the .roomodes file.
 * @returns {boolean} True on success, false on error (or if file doesn't exist).
 */
function deleteRoomodes() {
    console.log(`Attempting to delete ${ROOMODES_PATH}`);
    try {
        if (fs.existsSync(ROOMODES_PATH)) {
            fs.unlinkSync(ROOMODES_PATH);
            console.log(`.roomodes file deleted.`);
        } else {
            console.log(`.roomodes file does not exist, skipping deletion.`);
        }
        return true;
    } catch (error) {
        console.error(`Error deleting ${ROOMODES_PATH}:`, error);
        return false;
    }
}

/**
 * Executes the build script for a given collection.
 * @param {string} collectionName - The name of the collection.
 * @param {string[]} modeList - The list of modes for this collection (used for logging).
 */
function runBuildForCollection(collectionName, modeList) {
    // Define placeholders for positional arguments required by create_build.js
    const versionPlaceholder = `coll-${collectionName}`;
    const codenamePlaceholder = 'CollectionBuild';
    // Use path.resolve to ensure correct path construction relative to the project root
    const distReadmePlaceholder = '.ruru/templates/build/README.dist.md'; // Use path relative to project root
    const changelogPlaceholder = path.resolve(__dirname, '..', '.tmp', `CHANGELOG-${collectionName}.md`); // Placeholder path

    // Construct the command with positional arguments
    // Note: create_build.js calculates its own output path based on version/codename, so we don't pass the old 'outputPath'
    const command = `node "${BUILD_SCRIPT_PATH}" "${versionPlaceholder}" "${codenamePlaceholder}" "${distReadmePlaceholder}" "${changelogPlaceholder}"`;

    console.log(`\n--- Starting build for collection: ${collectionName} ---`);
    console.log(`Modes: ${modeList.join(', ')}`);
    console.log(`Executing: ${command}`);

    try {
        // Ensure build directory exists
        if (!fs.existsSync(BUILD_OUTPUT_DIR)) {
            console.log(`Creating build output directory: ${BUILD_OUTPUT_DIR}`);
            fs.mkdirSync(BUILD_OUTPUT_DIR, { recursive: true });
        }

        const output = execSync(command, { encoding: 'utf8', stdio: 'inherit' }); // Inherit stdio to see build output
        console.log(`\n✅ Build successful for collection: ${collectionName}`);
        // console.log(`   Output: ${outputPath}`); // outputPath is not defined in this scope
        return true;
    } catch (error) {
        console.error(`\n❌ Build failed for collection: ${collectionName}`);
        console.error(`   Error executing command: ${error.message}`);
        // execSync throws an error which includes stderr, so no need to log error.stderr separately
        return false;
    }
}

/**
 * Main function to run the collection builds.
 */
function main() {
    const collectionsConfig = readCollectionsConfig();
    if (!collectionsConfig || !collectionsConfig.collections) {
        console.error("Invalid or missing collections configuration. Exiting.");
        process.exit(1);
    }

    const collections = collectionsConfig.collections;
    let coreModes = [];

    // 1. Resolve Core Modes first
    if (collections.core) {
        console.log("Resolving 'core' modes...");
        coreModes = collections.core;
        console.log(`Core modes found: ${coreModes.join(', ')}`);
    } else {
        console.log("No 'core' collection defined.");
    }

    let allBuildsSuccessful = true;
    const collectionNames = Object.keys(collections);

    // 2. Iterate through collections
    for (const collectionName of collectionNames) {
        console.log(`\nProcessing collection: ${collectionName}`);
        let collectionModes = collections[collectionName];

        // Special case: "full" collection might not need a .roomodes file if it's just ["all"]
        const isSpecialFullCollection = collectionName === 'full' && Array.isArray(collectionModes) && collectionModes.length === 1 && collectionModes[0] === 'all';

        let finalModeList = [];

        // 3. Prepare Mode List
        if (Array.isArray(collectionModes)) {
            finalModeList = collectionModes.flatMap(mode => {
                if (mode === 'core') {
                    if (coreModes.length > 0) {
                        return coreModes;
                    } else {
                        console.warn(`Warning: Collection '${collectionName}' includes 'core', but no 'core' collection was defined. Skipping 'core'.`);
                        return [];
                    }
                }
                return mode;
            });
            // Remove duplicates
            finalModeList = [...new Set(finalModeList)];
        } else {
            console.error(`Error: Invalid mode definition for collection '${collectionName}'. Expected an array. Skipping.`);
            allBuildsSuccessful = false;
            continue; // Skip to the next collection
        }

        if (finalModeList.length === 0 && !isSpecialFullCollection) {
             console.warn(`Warning: Collection '${collectionName}' resulted in an empty mode list after resolving 'core'. Skipping build.`);
             continue;
        }

        // 4. Generate Temporary .roomodes (if needed)
        let roomodesWritten = true; // Assume success if not needed
        if (!isSpecialFullCollection) {
            const roomodesContent = finalModeList.join('\n');
            roomodesWritten = writeRoomodes(roomodesContent);
            if (!roomodesWritten) {
                console.error(`Failed to write .roomodes for collection '${collectionName}'. Skipping build.`);
                allBuildsSuccessful = false;
                continue; // Skip to the next collection
            }
        } else {
            console.log(`Collection '${collectionName}' is special 'full' type, potentially skipping .roomodes generation if build script handles 'all'.`);
            // Ensure .roomodes is clean for the 'full' build if the build script relies on it
            if (!writeRoomodes(DEFAULT_ROOMODES_CONTENT)) {
                 console.error(`Failed to write default .roomodes for collection '${collectionName}'. Skipping build.`);
                 allBuildsSuccessful = false;
                 continue;
            }
        }

        // 5. Execute Build Script
        const buildSuccess = runBuildForCollection(collectionName, finalModeList);
        if (!buildSuccess) {
            allBuildsSuccessful = false;
            // Decide if you want to stop on first failure or continue
            // For now, we continue to try building other collections
        }
    }

    // 6. Cleanup
    console.log("\n--- Build process finished. Cleaning up... ---");
    // Restore .roomodes to default state ("all")
    if (!writeRoomodes(DEFAULT_ROOMODES_CONTENT)) {
        console.error("Critical Error: Failed to restore .roomodes to default state!");
        // Don't exit, but report the error
        allBuildsSuccessful = false;
    } else {
        console.log(".roomodes restored to default ('all').");
    }


    console.log("\n--- Summary ---");
    if (allBuildsSuccessful) {
        console.log("✅ All collection builds attempted successfully.");
    } else {
        console.warn("⚠️ Some collection builds failed or were skipped due to errors. Please review the logs above.");
        process.exitCode = 1; // Indicate failure
    }
}

// Run the main function
main();