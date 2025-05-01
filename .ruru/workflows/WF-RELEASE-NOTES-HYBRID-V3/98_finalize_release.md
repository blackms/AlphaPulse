+++
# --- Step Metadata ---
step_id = "WF-RELEASE-NOTES-HYBRID-V3-98-FINALIZE" # Unique ID for this step
title = "Step 98: Finalize Release Notes & Optionally Push" # Title of this specific step
description = """
Aggregates all incremental notes from the 'next/' directory into a single, versioned release notes file.
Optionally uses the GitHub MCP to create/update a GitHub Release based on the aggregated notes and determined version.
Cleans up the incremental notes directory. This step only runs if the action is 'finalize'.
"""

# --- Flow Control ---
depends_on = ["WF-RELEASE-NOTES-HYBRID-V3-00-START"] # Depends on the initialization step
next_step = "99_finish.md" # Proceeds to finish after finalizing
error_step = "99_finish.md" # Default error handling to finish step

# --- Execution ---
# Needs modes for file manipulation, potentially formatting, and GitHub interaction.
# Let's use 'util-writer' for aggregation/formatting and note the MCP usage.
delegate_to = "util-writer" # Placeholder - orchestrator might need to coordinate with GitHub MCP directly

# --- Interface ---
inputs = [ # Data/artifacts needed from previous steps
    "Output from step {{depends_on[0]}}: determined_action", # Needed to confirm action is 'finalize'
    "Output from step {{depends_on[0]}}: determined_next_version", # e.g., v7.1.5
    "Output from step {{depends_on[0]}}: validated_output_dir", # e.g., .ruru/docs/release-notes/
    "Output from step {{depends_on[0]}}: incremental_notes_dir", # e.g., .ruru/docs/release-notes/next/
    "Output from step {{depends_on[0]}}: config_github_owner",
    "Output from step {{depends_on[0]}}: config_github_repo",
    "Output from step {{depends_on[0]}}: validated_push_flag",
    "Output from step {{depends_on[0]}}: validated_github_options", # Contains draft/prerelease flags
    "Output from step {{depends_on[0]}}: github_mcp_available"
]
outputs = [ # Data/artifacts produced by this step
    "final_release_notes_path: Path to the aggregated release notes file (e.g., .ruru/docs/release-notes/v7.1.5.md).",
    "github_release_url: URL of the created/updated GitHub Release (if pushed).",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # Placeholder
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # Link to the template definition
+++

# Step 98: Finalize Release Notes & Optionally Push

## Precondition
*   This step assumes the `determined_action` from the start step is `"finalize"`.

## Actions

1.  **List Incremental Notes:** Get a list of all `.md` files within the `{{incremental_notes_dir}}`.
2.  **Read & Concatenate:** Read the content of each incremental note file. Concatenate the Markdown bodies together (potentially adding separators or headers if needed).
3.  **Format Final Notes:**
    *   Load the standard release notes template (`.ruru/templates/toml-md/13_release_notes.md`).
    *   Populate the template's TOML frontmatter with `version = "{{determined_next_version}}"` and other relevant metadata (e.g., `date`).
    *   Insert the concatenated Markdown content into the body of the template.
4.  **Determine Final Filename:** Construct the filename for the final notes using the determined version (e.g., `{{determined_next_version}}.md`).
5.  **Construct Final Path:** Combine `{{validated_output_dir}}` and the final filename.
6.  **Save Final Notes:** Write the fully formatted release notes content to the final path. Store this path as `final_release_notes_path`.
7.  **Push to GitHub (Conditional):**
    *   **If `{{validated_push_flag}}` is true AND `{{github_mcp_available}}` is true:**
        *   Prepare arguments for the GitHub MCP `create_pull_request` (or potentially a dedicated `create_release` if available - check MCP schema) tool:
            *   `owner`: `{{config_github_owner}}`
            *   `repo`: `{{config_github_repo}}`
            *   `tag_name`: `{{determined_next_version}}`
            *   `name`: "Release {{determined_next_version}}" (or similar title)
            *   `body`: The aggregated Markdown content from step 3.
            *   `draft`: Value from `{{validated_github_options.mark_as_draft}}`.
            *   `prerelease`: Value from `{{validated_github_options.mark_as_prerelease}}`.
        *   Execute the GitHub MCP tool.
        *   Store the resulting release URL (if provided by the tool) as `github_release_url`.
        *   Handle potential errors from the MCP call.
8.  **Cleanup:** Delete the `{{incremental_notes_dir}}` and its contents.

## Acceptance Criteria

*   The final release notes file exists at `{{final_release_notes_path}}`.
*   The file contains aggregated content from the incremental notes, formatted correctly.
*   If `{{validated_push_flag}}` was true, a GitHub Release was attempted, and `github_release_url` might contain a URL.
*   The `{{incremental_notes_dir}}` directory has been removed.

## Error Handling

*   If reading/listing files from `{{incremental_notes_dir}}` fails, proceed to `{{error_step}}`.
*   If writing the final release notes file fails, proceed to `{{error_step}}`.
*   If the GitHub MCP call fails (when `validated_push_flag` is true), log the error but potentially continue to cleanup. Decide if this should halt the workflow (`error_step`) or just be noted in the final report. (Let's default to `error_step` for now).
*   If cleanup fails, log the error but proceed to `{{next_step}}`.