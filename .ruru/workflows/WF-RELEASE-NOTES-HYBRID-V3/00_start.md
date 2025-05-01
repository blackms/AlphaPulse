+++
# --- Step Metadata ---
step_id = "WF-RELEASE-NOTES-HYBRID-V3-00-START" # Updated ID for V3
title = "Step 00: Initialization, Config Reading & Version Determination" # Updated title
description = """
Validates inputs (action, output_dir, push options), reads GitHub config from '.ruru/config/project.toml',
determines the previous tag and next version based on existing release notes,
and prepares context for subsequent steps based on the specified action ('add' or 'finalize').
"""

# --- Flow Control ---
depends_on = [] # Start step has no dependencies
# Next step depends on the action. This needs conditional logic, handled by the orchestrator based on 'determined_action'.
# Placeholder - actual routing happens based on output. We might add specific next_step_add/next_step_finalize later if needed.
next_step = "01_generate_incremental_notes.md" # Default path for 'add' action
error_step = "99_finish.md" # Default error handling to finish step for now

# --- Execution ---
delegate_to = "" # Orchestrator or initial mode handles validation logic

# --- Interface ---
inputs = [
    "workflow_input: action (String, Optional, Default: 'add')",
    "workflow_input: output_dir (String, Optional, Default: '.ruru/docs/release-notes/')",
    "workflow_input: push_to_github (Boolean, Optional, Default: false)",
    "workflow_input: mark_as_draft (Boolean, Optional, Default: true)",
    "workflow_input: mark_as_prerelease (Boolean, Optional, Default: false)"
]
outputs = [
    "determined_action: The validated action ('add' or 'finalize').",
    "config_github_owner: GitHub owner read from config.",
    "config_github_repo: GitHub repo read from config.",
    "determined_previous_tag: The latest release tag found (e.g., v7.1.4).",
    "determined_next_version: The calculated next version tag (e.g., v7.1.5).",
    "validated_output_dir: The validated base output directory path.",
    "incremental_notes_dir: The path for storing incremental notes (e.g., .ruru/docs/release-notes/next/).",
    "validated_push_flag: The validated boolean flag for pushing to GitHub (relevant for 'finalize').",
    "validated_github_options: Object containing draft and prerelease flags (relevant for 'finalize').",
    "github_mcp_available: Boolean indicating if GitHub MCP is available (checked only if pushing during 'finalize')."
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # Placeholder
template_schema_doc = ".ruru/templates/toml-md/24_workflow_step_start.md"
+++

# Step 00: Initialization, Config Reading & Version Determination

## Actions

1.  **Validate Action:**
    *   Read the `action` input parameter.
    *   If not provided or invalid (not "add" or "finalize"), default to `"add"`.
    *   Store the validated action as `determined_action`.
2.  **Validate Output Directory:**
    *   Read the `output_dir` input parameter or use the default `.ruru/docs/release-notes/`.
    *   Ensure it's a valid path format. Store as `validated_output_dir`.
    *   Define the directory for incremental notes: `incremental_notes_dir = validated_output_dir + "/next"`.
    *   Ensure `incremental_notes_dir` exists. If not, create it (e.g., using `mkdir -p`).
3.  **Read Project Configuration:**
    *   Read the file `.ruru/config/project.toml`.
    *   Parse the TOML content.
    *   Extract `github.owner` and `github.repo`.
    *   If the file or keys are missing, proceed to `{{error_step}}`.
    *   Store the values as `config_github_owner` and `config_github_repo`.
4.  **Determine Versions:**
    *   List files in `validated_output_dir` matching the pattern `v*.md`.
    *   Parse the version numbers from the filenames (e.g., "v7.1.4.md" -> 7.1.4).
    *   Find the highest semantic version among the files. This is `determined_previous_tag` (e.g., "v7.1.4"). If no files found, potentially default to an initial tag like "v0.0.0" or require manual input (error for now).
    *   Increment the patch number of `determined_previous_tag` to calculate `determined_next_version` (e.g., "v7.1.5").
5.  **Validate GitHub Push Parameters (Conditional):**
    *   Store `push_to_github` flag (or default `false`) as `validated_push_flag`.
    *   Store `mark_as_draft` and `mark_as_prerelease` flags in `validated_github_options`.
    *   **If `determined_action` is "finalize" AND `validated_push_flag` is true:**
        *   Check for the availability and connectivity of the GitHub MCP server. Store result (true/false) in `github_mcp_available`. If not available, proceed to `{{error_step}}`.
    *   **Else:** Set `github_mcp_available` to `null` or indicate it's not applicable.

## Acceptance Criteria

*   `determined_action` is either "add" or "finalize".
*   `validated_output_dir` and `incremental_notes_dir` contain valid paths, and `incremental_notes_dir` exists.
*   `config_github_owner` and `config_github_repo` contain values read from the config file.
*   `determined_previous_tag` contains the latest version tag found (or an initial default).
*   `determined_next_version` contains the calculated next version tag.
*   `validated_push_flag` and `validated_github_options` are set.
*   If `determined_action` is "finalize" and `validated_push_flag` is true, `github_mcp_available` is true.

## Error Handling

*   If `.ruru/config/project.toml` cannot be read or `github.owner`/`github.repo` are missing, proceed to `{{error_step}}`.
*   If no previous release notes (`v*.md`) are found in `validated_output_dir` and no initial tag logic is defined, proceed to `{{error_step}}`.
*   If `determined_action` is "finalize", `validated_push_flag` is true, and the GitHub MCP server is unavailable, proceed to `{{error_step}}`.