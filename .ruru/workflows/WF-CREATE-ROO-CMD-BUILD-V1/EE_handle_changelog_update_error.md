+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-EE-HANDLE-CHANGELOG-UPDATE-ERROR" # (String, Required) Unique ID for this error handling step.
title = "Error Handler: CHANGELOG Update Failed" # (String, Required) Title of this specific step.
description = """
(String, Required) Handles errors encountered during the 'Update CHANGELOG.md' step (11).
Logs the error and prepares for workflow termination.
"""

# --- Flow Control ---
# This step is triggered by the 'error_step' field in '11_update_changelog.md'.
depends_on = []
next_step = ""
# error_step = ""

# --- Execution ---
delegate_to = "lead-devops" # Or prime-txt

# --- Interface ---
inputs = [
    "Input from step WF-CREATE-ROO-CMD-BUILD-V1-11-UPDATE-CHANGELOG: changelog_update_status", # Failure status
    "Input from step WF-CREATE-ROO-CMD-BUILD-V1-08-GENERATE-RELEASE-NOTES: release_notes_markdown", # Content that failed to prepend
]
outputs = [
    "workflow_result: Summary object indicating Failure, e.g., { overall_status: 'Failure', failed_step: '11', reason: 'Failed to update CHANGELOG.md', ... }.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}"
template_schema_doc = ".ruru/templates/toml-md/24_workflow_step_error.md"
+++

# Error Handler: CHANGELOG Update Failed

## Actions

1.  **Receive Context:** Get error details from the failed 'Update CHANGELOG.md' step (11).
2.  **Log Error:** Record the failure, indicating issues reading, modifying, or writing `CHANGELOG.md`.
3.  **Prepare Failure Output:** Construct the `workflow_result` object indicating Failure, specifying `failed_step` '11'.
4.  **Terminate Workflow:** Signal workflow termination with a failure status.

## Acceptance Criteria

*   Error context is received.
*   The CHANGELOG update failure is logged.
*   A `workflow_result` object indicating Failure is generated.
*   Workflow terminates.

## Error Handling

*   Terminal error handling step.