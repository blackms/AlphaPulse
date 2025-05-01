+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-11-UPDATE-CHANGELOG" # (String, Required) Unique ID for this step.
title = "Step 11: Update CHANGELOG.md" # (String, Required) Title of this specific step.
description = """
(String, Required) Updates the main project CHANGELOG.md file by prepending
the generated release notes content from the previous steps.
"""

# --- Flow Control ---
depends_on = ["WF-CREATE-ROO-CMD-BUILD-V1-10-UPDATE-README"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "12_commit_docs.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_changelog_update_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "prime-txt" # (String, Optional) Mode responsible for executing the core logic of this step.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-08-GENERATE-RELEASE-NOTES: release_notes_markdown", # Content to prepend
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-10-UPDATE-README: readme_update_status", # Prerequisite check
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "changelog_update_status: Success or Failure.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 11: Update CHANGELOG.md

## Actions

1.  **Receive Context:** Get `release_notes_markdown` and `readme_update_status` from previous steps.
2.  **Check Prerequisite:** If `readme_update_status` is Failure, immediately skip to the error step (`EE_handle_changelog_update_error.md`).
3.  **Read CHANGELOG:** Load the content of the main `CHANGELOG.md` file.
4.  **Prepend Notes:** Construct the new content by placing the `release_notes_markdown` at the beginning of the existing `CHANGELOG.md` content (ensure proper spacing/newlines between the new and old content).
5.  **Write CHANGELOG:** Save the combined content back to `CHANGELOG.md`.
6.  **Set Status:** Set `changelog_update_status` to Success if the file is updated successfully, otherwise Failure.
7.  **Prepare Output:** Provide the `changelog_update_status`.

## Acceptance Criteria

*   Step proceeds only if `readme_update_status` was Success.
*   `CHANGELOG.md` is read successfully.
*   The new release notes content is correctly prepended to the existing content.
*   `CHANGELOG.md` is saved with the updated content.
*   `changelog_update_status` is set accurately.

## Error Handling

*   If prerequisite `readme_update_status` is Failure, proceed directly to `EE_handle_changelog_update_error.md`.
*   If reading, modifying, or writing `CHANGELOG.md` fails, set `changelog_update_status` to Failure and proceed to `EE_handle_changelog_update_error.md`.