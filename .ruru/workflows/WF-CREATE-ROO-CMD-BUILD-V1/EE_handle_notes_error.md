+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-EE-HANDLE-NOTES-ERROR" # (String, Required) Unique ID for this error handling step.
title = "Error Handler: Release Notes Generation Failed" # (String, Required) Title of this specific step.
description = """
(String, Required) Handles errors encountered during the 'Generate Release Notes' step (08).
Logs the error and prepares for workflow termination.
"""

# --- Flow Control ---
# This step is triggered by the 'error_step' field in '08_generate_release_notes.md'.
depends_on = []
next_step = ""
# error_step = ""

# --- Execution ---
delegate_to = "lead-devops" # Or technical-writer

# --- Interface ---
inputs = [
    "Input from step WF-CREATE-ROO-CMD-BUILD-V1-08-GENERATE-RELEASE-NOTES: generate_notes_status", # Failure status
    "Input from step WF-CREATE-ROO-CMD-BUILD-V1-07-QUERY-GIT-HISTORY: commit_history_raw", # Context
]
outputs = [
    "workflow_result: Summary object indicating Failure, e.g., { overall_status: 'Failure', failed_step: '08', reason: 'Failed to generate release notes from commit history', ... }.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}"
template_schema_doc = ".ruru/templates/toml-md/24_workflow_step_error.md"
+++

# Error Handler: Release Notes Generation Failed

## Actions

1.  **Receive Context:** Get error details from the failed 'Generate Release Notes' step (08).
2.  **Log Error:** Record the failure, indicating issues parsing commit history or formatting notes.
3.  **Prepare Failure Output:** Construct the `workflow_result` object indicating Failure, specifying `failed_step` '08'.
4.  **Terminate Workflow:** Signal workflow termination with a failure status.

## Acceptance Criteria

*   Error context is received.
*   The notes generation failure is logged.
*   A `workflow_result` object indicating Failure is generated.
*   Workflow terminates.

## Error Handling

*   Terminal error handling step.