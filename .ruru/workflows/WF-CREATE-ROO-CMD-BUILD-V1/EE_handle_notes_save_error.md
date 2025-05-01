+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-EE-HANDLE-NOTES-SAVE-ERROR" # (String, Required) Unique ID for this error handling step.
title = "Error Handler: Save Release Notes Failed" # (String, Required) Title of this specific step.
description = """
(String, Required) Handles errors encountered during the 'Save Release Notes' step (09).
Logs the error and prepares for workflow termination.
"""

# --- Flow Control ---
# This step is triggered by the 'error_step' field in '09_save_release_notes.md'.
depends_on = []
next_step = ""
# error_step = ""

# --- Execution ---
delegate_to = "lead-devops" # Or prime-txt

# --- Interface ---
inputs = [
    "Input from step WF-CREATE-ROO-CMD-BUILD-V1-09-SAVE-RELEASE-NOTES: save_notes_status", # Failure status
    "Input from step WF-CREATE-ROO-CMD-BUILD-V1-08-GENERATE-RELEASE-NOTES: release_notes_markdown", # Content that failed to save
    # Potentially the target file path determined in step 09
]
outputs = [
    "workflow_result: Summary object indicating Failure, e.g., { overall_status: 'Failure', failed_step: '09', reason: 'Failed to save release notes file', ... }.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}"
template_schema_doc = ".ruru/templates/toml-md/24_workflow_step_error.md"
+++

# Error Handler: Save Release Notes Failed

## Actions

1.  **Receive Context:** Get error details from the failed 'Save Release Notes' step (09).
2.  **Log Error:** Record the failure, indicating issues writing the release notes file (e.g., permissions, path error).
3.  **Prepare Failure Output:** Construct the `workflow_result` object indicating Failure, specifying `failed_step` '09'.
4.  **Terminate Workflow:** Signal workflow termination with a failure status.

## Acceptance Criteria

*   Error context is received.
*   The file saving failure is logged.
*   A `workflow_result` object indicating Failure is generated.
*   Workflow terminates.

## Error Handling

*   Terminal error handling step.