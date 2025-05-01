const fsPromises = require('fs').promises; // Use promises API for async operations
const fs = require('fs'); // Use standard API for streams
const path = require('path');
const archiver = require('archiver');
const os = require('os');

// --- Configuration ---
const includedFilesAndDirs = [
  '.ruru/modes',
  '.ruru/processes',
  '.roo',
  '.ruru/templates',
  '.ruru/workflows',
  '.ruru/docs', // Include the docs directory
  '.roomodes',
  'scripts',
  '.ruru/scripts'
];

const emptyDirsToCreate = [
  '.ruru/archive',
  '.ruru/context',
  '.ruru/decisions',
  '.ruru/ideas',
  '.ruru/logs',
  '.ruru/planning',
  '.ruru/reports',
  '.ruru/snippets',
  '.ruru/tasks'
];

const outputDir = '.builds';
// --- End Configuration ---

async function createBuild() {
  // --- Argument Parsing ---
  const args = process.argv.slice(2);
  if (args.length < 4) {
    console.error('Usage: node create_build.js <version> <codename> <distReadmePath> <changelogPath>');
    console.error('Example: node create_build.js v7.01 Wallaby .ruru/templates/build/README.dist.md .tmp/CHANGELOG.v7.01.md');
    process.exit(1);
  }
  const version = args[0]; // e.g., v7.01
  const codename = args[1]; // e.g., Wallaby
  const distReadmePath = args[2]; // Path to the prepared README for the zip
  const changelogPath = args[3]; // Path to the prepared CHANGELOG for the zip

  const archiveNameStem = `roo-commander-${version}-${codename}`;
  const archiveFileName = `${archiveNameStem}.zip`;
  const outputZipPath = path.join(outputDir, archiveFileName);
  const stagingDir = await fsPromises.mkdtemp(path.join(os.tmpdir(), `roo-build-${version}-`)); // Use fsPromises

  console.log(`Starting build for ${version} (${codename})...`);
  console.log(`Staging directory: ${stagingDir}`);
  console.log(`Output archive: ${outputZipPath}`);

  try {
    // --- Ensure Output Directory Exists ---
    await fsPromises.mkdir(outputDir, { recursive: true }); // Use fsPromises

    // --- Copy Included Files/Dirs ---
    console.log('Copying included files and directories...');
    for (const item of includedFilesAndDirs) {
      const sourcePath = path.resolve(__dirname, '..', '..', item); // Resolve from project root (go up two levels from .ruru/scripts)
      const destPath = path.join(stagingDir, item);
      try {
        const stats = await fsPromises.stat(sourcePath); // Use fsPromises
        if (stats.isDirectory()) {
          console.log(`  Copying directory: ${item}`);
          await fsPromises.cp(sourcePath, destPath, { recursive: true }); // Use fsPromises
        } else {
          console.log(`  Copying file: ${item}`);
          await fsPromises.copyFile(sourcePath, destPath); // Use fsPromises
        }
      } catch (err) {
        if (err.code === 'ENOENT') {
          console.warn(`  Warning: Source item not found, skipping: ${item}`);
        } else {
          throw err; // Re-throw other errors
        }
      }
    }

    // --- Create Empty Dirs ---
    console.log('Creating empty directories...');
    for (const dir of emptyDirsToCreate) {
      console.log(`  Creating empty dir: ${dir}`);
      await fsPromises.mkdir(path.join(stagingDir, dir), { recursive: true }); // Use fsPromises
    }

    // --- Copy Dist README and CHANGELOG ---
    console.log('Copying distribution README and CHANGELOG...');
    try {
      await fsPromises.copyFile(path.resolve(__dirname, '..', '..', distReadmePath), path.join(stagingDir, 'README.md')); // Resolve from project root
      console.log(`  Copied ${distReadmePath} to README.md`);
    } catch (err) {
      console.error(`  Error copying distribution README from ${distReadmePath}: ${err.message}`);
      throw err;
    }
    try {
      await fsPromises.copyFile(changelogPath, path.join(stagingDir, 'CHANGELOG.md')); // Use fsPromises
      console.log(`  Copied ${changelogPath} to CHANGELOG.md`);
    } catch (err) {
      if (err.code === 'ENOENT') {
        // Log a warning if the changelog file doesn't exist, as it might be generated later.
        console.warn(`  Warning: CHANGELOG file not found at ${changelogPath}, skipping copy. This might be expected if generated later.`);
      } else {
        // For other errors, log and re-throw.
        console.error(`  Error copying CHANGELOG from ${changelogPath}: ${err.message}`);
        throw err;
      }
    }


    // --- Create Zip Archive ---
    console.log(`Creating zip archive: ${archiveFileName}...`);
    const output = fs.createWriteStream(outputZipPath);
    const archive = archiver('zip', {
      zlib: { level: 9 } // Sets the compression level.
    });

    // Listen for all archive data to be written
    // 'close' event is fired only when a file descriptor is involved
    await new Promise((resolve, reject) => {
        output.on('close', () => {
          console.log(`Archive created successfully: ${archive.pointer()} total bytes`);
          resolve();
        });

        // This event is fired when the data source is drained no matter what was the data source.
        // It is not part of this library but rather from the NodeJS Stream API.
        // @see: https://nodejs.org/api/stream.html#stream_event_end
        output.on('end', () => {
          console.log('Data has been drained');
        });

        // Good practice to catch warnings (ie stat failures and other non-blocking errors)
        archive.on('warning', (err) => {
          if (err.code === 'ENOENT') {
            console.warn(`Archiver warning: ${err.message}`);
          } else {
            // Throw error
            reject(err);
          }
        });

        // Good practice to catch this error explicitly
        archive.on('error', (err) => {
          reject(err);
        });

        // Pipe archive data to the file
        archive.pipe(output);

        // Append files from staging directory, putting its contents at the root of archive
        archive.directory(stagingDir, false);

        // Finalize the archive (ie we are done appending files but streams have to finish yet)
        // 'close', 'end' or 'finish' may be fired right after calling this method so register to them beforehand
        archive.finalize();
    });

    console.log(`Build complete: ${outputZipPath}`);

  } catch (error) {
    console.error('Build failed:', error);
    process.exitCode = 1; // Indicate failure
  } finally {
    // --- Cleanup Staging Directory ---
    console.log(`Cleaning up staging directory: ${stagingDir}`);
    try {
      await fsPromises.rm(stagingDir, { recursive: true, force: true }); // Use fsPromises
      console.log('Cleanup complete.');
    } catch (cleanupError) {
      console.error(`Error during cleanup: ${cleanupError.message}`);
      // Don't override process exit code if build already failed
      if (!process.exitCode) {
        process.exitCode = 1;
      }
    }
  }
}

createBuild();