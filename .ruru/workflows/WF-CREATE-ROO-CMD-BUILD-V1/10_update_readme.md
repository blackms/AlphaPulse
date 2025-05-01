+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-10-UPDATE-README" # (String, Required) Unique ID for this step.
title = "Step 10: Update README.md" # (String, Required) Title of this specific step.
description = """
(String, Required) Updates the main project README.md file to include a link
to the newly created release notes file.
"""

# --- Flow Control ---
depends_on = ["WF-CREATE-ROO-CMD-BUILD-V1-09-SAVE-RELEASE-NOTES"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "11_update_changelog.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_readme_update_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "prime-txt" # (String, Optional) Mode responsible for executing the core logic of this step.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-09-SAVE-RELEASE-NOTES: release_notes_file_path",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-09-SAVE-RELEASE-NOTES: save_notes_status", # Prerequisite check
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "readme_update_status: Success or Failure.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 10: Update README.md

## Actions

1.  **Receive Context:** Get `release_notes_file_path` and `save_notes_status` from the previous step.
2.  **Check Prerequisite:** If `save_notes_status` is Failure, immediately skip to the error step (`EE_handle_readme_update_error.md`).
3.  **Read README:** Load the content of the main `README.md` file.
4.  **Find Insertion Point:** Locate the appropriate section in the README (e.g., a "Releases" or "Changelog" section) where the link should be added.
5.  **Generate Link:** Create a Markdown link to the `release_notes_file_path`.
6.  **Insert Link:** Add the generated link to the README content at the identified insertion point. Ensure formatting is consistent.
7.  **Write README:** Save the modified content back to `README.md`.
8.  **Set Status:** Set `readme_update_status` to Success if the file is updated successfully, otherwise Failure.
9.  **Prepare Output:** Provide the `readme_update_status`.

## Acceptance Criteria

*   Step proceeds only if `save_notes_status` was Success.
*   `README.md` is read successfully.
*   The correct insertion point is found.
*   A valid Markdown link to the new release notes is inserted.
*   `README.md` is saved with the updated content.
*   `readme_update_status` is set accurately.

## Error Handling

*   If prerequisite `save_notes_status` is Failure, proceed directly to `EE_handle_readme_update_error.md`.
*   If reading, modifying, or writing `README.md` fails, set `readme_update_status` to Failure and proceed to `EE_handle_readme_update_error.md`.