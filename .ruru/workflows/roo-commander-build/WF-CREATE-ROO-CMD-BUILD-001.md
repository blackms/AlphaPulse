+++
# --- Basic Metadata ---
id = "WF-CREATE-ROO-CMD-BUILD-001" # Keeping original ID
title = "Workflow: Roo Commander Release Build, Documentation Update & Publication" # Updated Title
status = "active"
created_date = "2025-04-20"
updated_date = "2025-04-28" # Updated date
version = "6.0" # Incremented for verification/finalization
tags = ["workflow", "build", "release", "distribution", "archive", "zip", "roo-commander", "git", "github", "release-notes", "publication", "documentation", "readme", "changelog", "commit"] # Ensured tags are present

# --- Ownership & Context ---
owner = "Roo Commander"
related_docs = [
  ".ruru/scripts/create_build.js",
  ".ruru/scripts/create_kilocode_build.js",
  ".ruru/scripts/run_collection_builds.js",
  ".ruru/workflows/roo-commander-build/WF-RELEASE-NOTES-HYBRID-001.md",
  ".roo/rules/07-git-commit-standard-simplified.md",
  # Add links to GitHub API docs or gh CLI docs if used
]
related_templates = [
    # ".ruru/templates/toml-md/NN_release_notes.md"
]

# --- Workflow Specific Fields ---
objective = "Creates Roo Commander distribution builds, generates release notes based on Conventional Commits, updates documentation files (README.md, CHANGELOG.md), commits these changes, tags the release in Git, and optionally creates a GitHub release with artifacts." # Ensured objective reflects changes
scope = "Applies when creating and publishing an official release build of Roo Commander, including documentation updates." # Updated scope
roles = [
  "Coordinator (Roo Commander)",
  "Executor (Terminal via `execute_command`)",
  "Git Specialist (`dev-git`)",
  "File Reader/Writer (`prime-txt`, `read_file`, `write_to_file`, `apply_diff`, `list_files`)", # Consolidated file roles
  "GitHub Interactor (GitHub MCP or `gh` CLI via Executor)"
] # Ensured roles are present
trigger = "Manual initiation by the Coordinator for a new official release."

# --- Input Parameters ---
[parameters]
target_tag = { type = "string", description = "The Git tag for the current release (e.g., v1.2.0). MUST be provided.", required = true }
previous_tag = { type = "string", description = "The Git tag for the previous release to compare against for release notes. If omitted, uses the latest tag before target_tag.", required = false }
build_output_dir = { type = "string", description = "Directory to save the build artifacts.", default = "./build/release/" }
notes_output_dir = { type = "string", description = "Directory to save the generated release notes Markdown file.", default = "./build/release/" }
create_github_release = { type = "boolean", description = "Attempt to create a GitHub Release.", default = true }
github_owner = { type = "string", description = "GitHub repository owner (required if create_github_release is true).", required = false }
github_repo = { type = "string", description = "GitHub repository name (required if create_github_release is true).", required = false }
mark_as_draft = { type = "boolean", description = "Mark the GitHub Release as a draft.", default = false }
mark_as_prerelease = { type = "boolean", description = "Mark the GitHub Release as a pre-release.", default = false }
release_date = { type = "string", description = "Release date (e.g., YYYY-MM-DD) for CHANGELOG. Defaults to current date if omitted.", required = false } # Added for CHANGELOG

# --- Output Variables ---
# These variables are set during the workflow execution
[output_variables]
previous_tag_resolved = { type = "string", description = "The actual previous tag used for comparison (either provided or determined)." }
release_notes_body = { type = "string", description = "The generated Markdown content for the release notes." }
notes_file_path = { type = "string", description = "The full path to the saved release notes file." }
changelog_exists_and_updated = { type = "boolean", description = "Flag indicating if CHANGELOG.md existed and was updated.", default = false }

success_criteria = [
  "The `create_build.js` script executes successfully, placing output in `{{build_output_dir}}`.",
  "The `create_kilocode_build.js` script executes successfully, placing output in `{{build_output_dir}}`.",
  "The `run_collection_builds.js` script executes successfully, placing output in `{{build_output_dir}}`.",
  "Expected build artifacts are verified to exist within `{{build_output_dir}}`.",
  "Previous tag is determined successfully (stored in `previous_tag_resolved`).",
  "Git history is queried successfully between tags.",
  "Release notes content is generated based on Conventional Commits (stored in `release_notes_body`).",
  "Release notes Markdown file is saved successfully to `{{notes_output_dir}}/NOTES-{{target_tag}}.md` (path stored in `notes_file_path`).",
  "README.md is updated successfully with the release tag.", # Added/Ensured
  "CHANGELOG.md is updated successfully by prepending release notes (if the file existed, flag `changelog_exists_and_updated` is true).", # Added/Ensured
  "Release notes and updated documentation files are committed successfully.", # Added/Ensured
  "The `target_tag` is pushed successfully to the remote Git repository (`origin`).",
  "If `create_github_release` is true, the GitHub Release is created successfully with notes and artifacts attached.",
] # Updated criteria
failure_criteria = [
  "Any build script fails or produces errors.",
  "Build artifacts are not found in `{{build_output_dir}}` during verification.",
  "Failed to determine the previous Git tag.",
  "Failed to query Git history.",
  "Failed to parse commits or generate release notes content.",
  "Failed to save the release notes file.",
  "Failed to update README.md.", # Added/Ensured
  "Failed to update CHANGELOG.md (if attempted).", # Added/Ensured
  "Failed to commit release notes and updated documentation files.", # Added/Ensured
  "Failed to push the `target_tag` to the remote Git repository.",
  "If `create_github_release` is true, failed to create the GitHub Release or attach artifacts."
] # Updated criteria

# --- Integration ---
acqa_applicable = false
pal_validated = false
validation_notes = "Workflow needs testing with build scripts, Git interactions, documentation updates, and optional GitHub release creation." # Updated

# --- AI Interaction Hints (Optional) ---
# context_type = "workflow_definition"
+++

# Workflow: Roo Commander Release Build, Documentation Update & Publication

## 1. Objective 🎯
*   To create the official Roo Commander distribution builds (Complete, Kilocode, Collections).
*   To generate Markdown release notes based on Conventional Commits between the previous and target release tags.
*   **To update `README.md` and `CHANGELOG.md` with release information.**
*   **To commit the generated notes and documentation updates.**
*   To tag the release commit in Git and push the tag.
*   To optionally create a corresponding GitHub Release, attaching the build artifacts and release notes.

## 2. Scope ↔️
*   This workflow is triggered manually to perform a full release process. It modifies local documentation files, creates commits, pushes a tag, and potentially interacts with GitHub Releases.

## 3. Roles & Responsibilities 👤
*   **Coordinator (Roo Commander):** Initiates the workflow, validates inputs, orchestrates steps, delegates tasks, manages state (output variables).
*   **Executor (Terminal):** Runs build scripts and potentially the `gh` CLI via `execute_command`.
*   **Git Specialist (`dev-git`):** Handles Git operations like finding tags, querying logs, staging, committing, and pushing tags.
*   **File Reader/Writer:** Reads/writes/updates local files (`README.md`, `CHANGELOG.md`, notes file) using appropriate tools (`read_file`, `write_to_file`, `apply_diff`, `list_files`).
*   **GitHub Interactor (GitHub MCP or `gh` CLI via Executor):** Creates the GitHub Release.

## 4. Preconditions🚦
*   Necessary tools (`node`, `git`, standard shell commands, potentially `gh` CLI) are available.
*   Build scripts (`.ruru/scripts/create_build.js`, etc.) exist, are functional, and support outputting to `{{build_output_dir}}`.
*   The Git repository is in a clean state (no uncommitted changes conflicting with `README.md` or `CHANGELOG.md`), ready for tagging.
*   Conventional Commit standards are followed in the commit history.
*   If `create_github_release` is true, necessary credentials/authentication for GitHub (MCP or `gh` CLI) are configured.
*   `README.md` exists at the repository root.

## 5. Reference Documents & Tools 📚🛠️
*   `.ruru/scripts/create_build.js`, `.ruru/scripts/create_kilocode_build.js`, `.ruru/scripts/run_collection_builds.js`
*   `.ruru/workflows/roo-commander-build/WF-RELEASE-NOTES-HYBRID-001.md` (Source for notes logic)
*   `.roo/rules/07-git-commit-standard-simplified.md`
*   `execute_command`: Tool to run shell commands, Node.js scripts, `gh` CLI.
*   `dev-git` mode: For Git operations.
*   `read_file`, `write_to_file`, `apply_diff`, `list_files`: For file operations.
*   GitHub MCP / `gh` CLI: For GitHub Release creation.

## 6. Workflow Steps 🪜
*Use 📣 to indicate steps that should report progress back.*
*Use ❓ for decision points/optional steps.*
*Use ❗ for critical failure points.*

*   **Step 1: Initialization & Input Validation (Coordinator)**
    *   [ ] 📣 Verify required parameter `target_tag` is provided and looks like a valid tag (e.g., `vX.Y.Z`).
    *   [ ] ❗ If `create_github_release` is true, verify `github_owner` and `github_repo` are provided. Fail if not.
    *   [ ] ❗ If `create_github_release` is true, verify the configured GitHub interaction method (MCP or `gh` CLI) is available and likely authenticated. Fail if not.
    *   [ ] Ensure output directories (`{{build_output_dir}}`, `{{notes_output_dir}}`) exist or can be created (e.g., using `execute_command` with `mkdir -p {{build_output_dir}} {{notes_output_dir}}`).
    *   [ ] Determine `release_date_resolved` (use `parameters.release_date` if provided, otherwise get current date YYYY-MM-DD).

*   **Step 2: Execute Complete Build (Coordinator delegates to Executor via `execute_command`)**
    *   **Description:** Run the script to create the complete build in the release directory.
    *   **Tool:** `execute_command`
    *   **Command Example:** `node .ruru/scripts/create_build.js --output {{build_output_dir}}`
    *   **Validation:** Check for exit code 0 and success messages.
    *   **Error Handling:** ❗ If the script fails, analyze output. Report failure.

*   **Step 3: Execute Kilocode Build (Coordinator delegates to Executor via `execute_command`)**
    *   **Description:** Run the script to create the Kilocode build in the release directory.
    *   **Tool:** `execute_command`
    *   **Command Example:** `node .ruru/scripts/create_kilocode_build.js --output {{build_output_dir}}`
    *   **Validation:** Check for exit code 0 and success messages.
    *   **Error Handling:** ❗ If the script fails, analyze output. Report failure.

*   **Step 4: Execute Collection Builds (Coordinator delegates to Executor via `execute_command`)**
    *   **Description:** Run the script to create the collection builds in the release directory.
    *   **Tool:** `execute_command`
    *   **Command Example:** `node .ruru/scripts/run_collection_builds.js --output {{build_output_dir}}`
    *   **Validation:** Check for exit code 0 and success messages.
    *   **Error Handling:** ❗ If the script fails, analyze output. Report failure.

*   **Step 5: Verify Build Artifacts (Coordinator delegates to Executor via `execute_command` or uses `list_files`)**
    *   **Description:** Check if the expected primary build artifacts exist in the release directory.
    *   **Tool:** `execute_command` or `list_files`
    *   **Command Example (`execute_command` - Linux/macOS):**
        ```bash
        echo "Verifying builds in {{build_output_dir}}..." && \
        ls {{build_output_dir}}/*.zip && \
        # Add checks for other expected artifact types if necessary
        echo "Verification successful."
        # Adjust *.zip pattern based on actual expected output filenames/types.
        ```
    *   **Validation:** Check for command success (exit code 0 for `execute_command`, presence of files in `list_files` result).
    *   **Error Handling:** ❗ If verification fails, report missing artifacts. Halt the workflow.

*   **Step 6: Determine Previous Tag (Coordinator delegates to `dev-git`)**
    *   **Description:** If `previous_tag` parameter was not provided, find the most recent tag before `target_tag`.
    *   **Tool:** `new_task` -> `dev-git`
    *   **Input:** `target_tag`
    *   **Instruction:** "Find the latest Git tag chronologically before `{{target_tag}}`." (e.g., using `git describe --tags --abbrev=0 {{target_tag}}^` or `git tag --sort=-creatordate | grep -B1 {{target_tag}} | head -n1`)
    *   **Output:** Store the determined tag in `output_variables.previous_tag_resolved`. Use the provided `parameters.previous_tag` if it was given.
    *   **Validation:** Check if `dev-git` returned a valid tag.
    *   **Error Handling:** ❗ If no previous tag can be found (e.g., first release), report and potentially halt or adapt notes generation.

*   **Step 7: Query Git History (Coordinator delegates to `dev-git`)**
    *   **Description:** Get commit history between the previous and target tags.
    *   **Tool:** `new_task` -> `dev-git`
    *   **Input:** `output_variables.previous_tag_resolved`, `parameters.target_tag`
    *   **Instruction:** `Execute git log --pretty=format:"%H ||| %s ||| %b%n---COMMIT-END---%n" {{output_variables.previous_tag_resolved}}..{{parameters.target_tag}}`
    *   **Output:** Store the raw `git log` output.
    *   **Validation:** Check if `dev-git` returned output.
    *   **Error Handling:** ❗ Handle errors from `dev-git` (e.g., invalid tags).

*   **Step 8: Parse Commits & Generate Notes Content (Coordinator or dedicated parser mode)**
    *   **Description:** Process the raw `git log` output, filter by Conventional Commit types, and format into Markdown.
    *   *(Self-execution or delegate to a text-processing specialist mode if complex)*
    *   **Input:** Raw `git log` output from Step 7.
    *   **Logic:** (As described in previous version) Parse commits, group by type, format into Markdown sections.
    *   **Output:** Store Markdown string in `output_variables.release_notes_body`.
    *   **Validation:** Check if `release_notes_body` is non-empty (unless no relevant commits were found).
    *   **Error Handling:** Report issues during parsing.

*   **Step 9: Save Release Notes File (Coordinator delegates to File Writer)**
    *   **Description:** Create the local Markdown file containing the generated notes.
    *   **Tool:** `write_to_file`
    *   **Input:** `output_variables.release_notes_body`, `parameters.target_tag`, `parameters.notes_output_dir`
    *   **Path:** `{{parameters.notes_output_dir}}/NOTES-{{parameters.target_tag}}.md`
    *   **Content:** Combine appropriate TOML frontmatter (optional) and the `release_notes_body`.
    *   **Output:** Store the full path in `output_variables.notes_file_path`.
    *   **Validation:** Check for successful file write confirmation.
    *   **Error Handling:** ❗ Report file writing errors.
    *   **Action:** 📣 Report success and the path to the created file: `{{output_variables.notes_file_path}}`.

*   **Step 10: Update README.md (Coordinator delegates to File Writer)**
    *   **Description:** Update the root `README.md` to indicate the latest release tag.
    *   **Tool:** `read_file`, then `apply_diff` or `write_to_file`.
    *   **Input:** `parameters.target_tag`, Path: `./README.md`.
    *   **Logic:**
        *   Read `./README.md`.
        *   Find a line like `Latest Release: vX.Y.Z` or a specific marker (e.g., `<!-- LATEST_RELEASE -->`).
        *   Prepare a diff to replace the old tag/marker line with `Latest Release: {{parameters.target_tag}}` (or similar). If no marker, decide on an insertion point.
        *   Use `apply_diff` or `write_to_file` with the modified content.
    *   **Validation:** Check for successful file modification confirmation.
    *   **Error Handling:** ❗ Report errors reading or modifying `README.md`.
    *   **Action:** 📣 Report success of README update.

*   **Step 11: Update CHANGELOG.md (Coordinator delegates to File Writer)**
    *   **Description:** Check if `CHANGELOG.md` exists. If yes, prepend the generated release notes content to it.
    *   **Tool:** `list_files`, `read_file`, `write_to_file`.
    *   **Input:** `parameters.target_tag`, `output_variables.notes_file_path`, `release_date_resolved`. Path: `./CHANGELOG.md`.
    *   **Logic:**
        *   Use `list_files` for `./`. Check if `CHANGELOG.md` is present.
        *   If yes:
            *   Read content from `output_variables.notes_file_path`. Store as `notes_content`.
            *   Read content from `./CHANGELOG.md`. Store as `changelog_content`.
            *   Construct `new_entry = "\n\n## [{{parameters.target_tag}}] - {{release_date_resolved}}\n\n" + notes_content`.
            *   Construct `updated_changelog = new_entry + changelog_content`.
            *   Use `write_to_file` to write `updated_changelog` to `./CHANGELOG.md`.
            *   Set `output_variables.changelog_exists_and_updated = true`.
        *   If no:
            *   Set `output_variables.changelog_exists_and_updated = false`.
            *   Log that CHANGELOG.md was not found and skipped.
    *   **Validation:** Check for successful file write confirmation (if attempted).
    *   **Error Handling:** ❗ Report errors reading notes file or reading/writing `CHANGELOG.md`.
    *   **Action:** 📣 Report success or skip based on file existence.

*   **Step 12: Commit Release Files (Coordinator delegates to `dev-git`)**
    *   **Description:** Stage and commit the updated documentation files and the generated release notes.
    *   **Tool:** `new_task` -> `dev-git`
    *   **Input:** `parameters.target_tag`, `output_variables.notes_file_path`, `output_variables.changelog_exists_and_updated`. Paths: `./README.md`, `./CHANGELOG.md`.
    *   **Instruction:**
        1.  `Execute git add ./README.md {{output_variables.notes_file_path}}`
        2.  If `output_variables.changelog_exists_and_updated` is true: `Execute git add ./CHANGELOG.md`
        3.  `Execute git commit -m "docs: Update documentation for release {{parameters.target_tag}}"`
    *   **Validation:** Check for success confirmation from `dev-git` for both `add` and `commit`.
    *   **Error Handling:** ❗ Report errors during staging or commit (e.g., nothing to commit, conflicts).
    *   **Action:** 📣 Report success of commit.

*   **Step 13: Push Git Tag (Coordinator delegates to `dev-git`)**
    *   **Description:** Push the specified `target_tag` to the remote repository (`origin`). Assumes the tag exists locally on the commit created in the previous step (or the commit before that if no docs were changed). (Consider adding `git tag -a {{target_tag}} -m "Release {{target_tag}}"` before push if tag doesn't exist).
    *   **Tool:** `new_task` -> `dev-git`
    *   **Input:** `parameters.target_tag`
    *   **Instruction:** `Execute git push origin {{parameters.target_tag}}`
    *   **Validation:** Check for success confirmation from `dev-git`.
    *   **Error Handling:** ❗ Report errors during tag push (e.g., tag already exists remotely, authentication issues).

*   **Step 14: ❓ Optional - Create GitHub Release (Coordinator delegates to GitHub Interactor)**
    *   **Description:** If `create_github_release` is true, create a release on GitHub.
    *   **Condition:** `parameters.create_github_release == true`
    *   **Tool:** `use_mcp_tool` (GitHub MCP) or `execute_command` (`gh` CLI)
    *   **Input:** `parameters.github_owner`, `parameters.github_repo`, `parameters.target_tag`, `output_variables.notes_file_path`, `parameters.mark_as_draft`, `parameters.mark_as_prerelease`, list of artifact paths from `{{parameters.build_output_dir}}`.
    *   **Instruction (gh CLI Example):**
        ```bash
        gh release create {{parameters.target_tag}} \
          --repo {{parameters.github_owner}}/{{parameters.github_repo}} \
          --title "Version {{parameters.target_tag}}" \
          --notes-file {{output_variables.notes_file_path}} \
          {{#if parameters.mark_as_draft}}--draft{{/if}} \
          {{#if parameters.mark_as_prerelease}}--prerelease{{/if}} \
          {{parameters.build_output_dir}}/*.zip # Adjust glob pattern for artifacts
        ```
    *   **Instruction (MCP):** Use the appropriate MCP tool (e.g., `create_release`) with corresponding arguments.
    *   **Validation:** Check for success confirmation from the tool/MCP.
    *   **Error Handling:** ❗ Report errors (authentication, API limits, tag mismatch, artifact upload failure).
    *   **Action:** 📣 Report success (with link to release/draft) or failure of GitHub release creation.

## 7. Postconditions ✅
*   Build artifacts are successfully generated in `{{parameters.build_output_dir}}`.
*   Release notes Markdown file is created at `{{output_variables.notes_file_path}}`.
*   `README.md` is updated.
*   `CHANGELOG.md` is updated (if it existed).
*   A Git commit containing these documentation updates is created.
*   The Git tag `{{parameters.target_tag}}` is pushed to the remote repository (`origin`).
*   If requested, a GitHub Release is created (or attempted).

## 8. Error Handling & Escalation (Overall) ⚠️
*   If build scripts fail, analyze output. Check dependencies, paths. Halt.
*   If verification fails, investigate build scripts. Halt.
*   If Git operations fail (finding tags, logging, staging, committing, pushing tag), check Git state, tags, authentication. Halt or adapt.
*   If notes generation fails, check commit history format or parsing logic. Halt or proceed without notes.
*   If file reading/writing/modification fails, check permissions, paths. Halt.
*   If GitHub release creation fails, check parameters, authentication, API limits, artifact paths. Report failure but potentially consider the rest of the workflow successful if builds and tagging worked.
*   Escalate to the user if steps fail and cannot be resolved.

## 9. PAL Validation Record 🧪
*   (Requires testing after implementation)

## 10. Revision History 📜
*   v6.0 (2025-04-28): Verified inclusion and order of documentation update (README, CHANGELOG) and commit steps. Updated version number.
*   v5.0 (2025-04-28): Added steps 10 (Update README), 11 (Update CHANGELOG), 12 (Commit Docs). Renumbered subsequent steps. Updated metadata, objective, roles, criteria, parameters (`release_date`), and added output variables (`notes_file_path`, `changelog_exists_and_updated`). Refined descriptions for clarity.
*   v4.0 (2025-04-28): Transformed into a full release workflow. Removed temporary directory logic. Changed build output to `./build/release/`. Integrated release notes generation (tag determination, git log, parsing, file saving). Added Git tag push step. Added optional GitHub Release creation step. Updated all metadata, parameters, roles, steps, and descriptions accordingly. (Note: This was v3.0 in previous file, corrected here).
*   v2.0 (2025-04-28): Refocused workflow on creating builds in temporary directories only. Removed Git, GitHub release, CHANGELOG, and build log steps. Added setup, verification, and cleanup steps for temporary directories. Updated metadata, objective, scope, roles, criteria, and descriptions accordingly.
*   v1.2 (2025-04-21): Inserted Git add, commit, push steps (5-7) before build (8) and release (9). Moved build log update to Step 4. Renumbered steps accordingly. Added `git` to tools list and updated descriptions/examples. Corrected build script example command and archive name format. Updated objective, preconditions, and postconditions.
*   v1.1 (2025-04-20): Added Step 6 for GitHub Release creation using `gh` CLI. Updated roles, criteria, and tools list.
*   v1.0 (2025-04-20): Initial draft incorporating versioning, build log, CHANGELOG, distribution README, and suggestion for an automated build script.