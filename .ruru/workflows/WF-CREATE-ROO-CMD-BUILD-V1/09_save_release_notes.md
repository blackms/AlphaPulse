+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-09-SAVE-RELEASE-NOTES" # (String, Required) Unique ID for this step.
title = "Step 09: Save Release Notes" # (String, Required) Title of this specific step.
description = """
(String, Required) Saves the generated Markdown release notes content to a
designated file path, typically within the `.ruru/docs/release-notes/` directory.
"""

# --- Flow Control ---
depends_on = ["WF-CREATE-ROO-CMD-BUILD-V1-08-GENERATE-RELEASE-NOTES"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "10_update_readme.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_notes_save_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "prime-txt" # (String, Optional) Mode responsible for executing the core logic of this step.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-08-GENERATE-RELEASE-NOTES: release_notes_markdown",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-08-GENERATE-RELEASE-NOTES: generate_notes_status", # Prerequisite check
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-01-VALIDATE-PARAMS: validated_build_params", # For determining filename/path
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "release_notes_file_path: The path where the release notes were saved.",
    "save_notes_status: Success or Failure.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 09: Save Release Notes

## Actions

1.  **Receive Context:** Get `release_notes_markdown`, `generate_notes_status`, and `validated_build_params` from previous steps.
2.  **Check Prerequisite:** If `generate_notes_status` is Failure, immediately skip to the error step (`EE_handle_notes_save_error.md`).
3.  **Determine File Path:** Construct the target file path for the release notes. This usually involves the new version number from `validated_build_params` and a standard location like `.ruru/docs/release-notes/vX.Y.Z.md` or `.ruru/docs/release-notes/next/update_{timestamp}.md`.
4.  **Write File:** Use the `write_to_file` tool to save the `release_notes_markdown` content to the determined path.
5.  **Set Status:** Set `save_notes_status` to Success if the file is written successfully, otherwise Failure.
6.  **Prepare Output:** Provide the `release_notes_file_path` and the `save_notes_status`.

## Acceptance Criteria

*   Step proceeds only if `generate_notes_status` was Success.
*   A valid file path for the release notes is determined.
*   The release notes Markdown content is successfully written to the file.
*   `release_notes_file_path` is populated.
*   `save_notes_status` is set accurately.

## Error Handling

*   If prerequisite `generate_notes_status` is Failure, proceed directly to `EE_handle_notes_save_error.md`.
*   If writing the file fails (e.g., permissions, invalid path), set `save_notes_status` to Failure and proceed to `EE_handle_notes_save_error.md`.