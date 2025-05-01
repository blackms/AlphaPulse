+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-EE-HANDLE-GIT-TAG-ERROR" # (String, Required) Unique ID for this error handling step.
title = "Error Handler: Determine Previous Git Tag Failed" # (String, Required) Title of this specific step.
description = """
(String, Required) Handles errors encountered during the 'Determine Previous Git Tag' step (06).
Logs the error and prepares for workflow termination.
"""

# --- Flow Control ---
# This step is triggered by the 'error_step' field in '06_determine_prev_tag.md'.
depends_on = []
next_step = ""
# error_step = ""

# --- Execution ---
delegate_to = "lead-devops" # Or dev-git

# --- Interface ---
inputs = [
    "Input from step WF-CREATE-ROO-CMD-BUILD-V1-06-DETERMINE-PREV-TAG: determine_tag_status", # Failure status
    # Potentially Git command output/error
]
outputs = [
    "workflow_result: Summary object indicating Failure, e.g., { overall_status: 'Failure', failed_step: '06', reason: 'Failed to determine previous Git tag', ... }.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}"
template_schema_doc = ".ruru/templates/toml-md/24_workflow_step_error.md"
+++

# Error Handler: Determine Previous Git Tag Failed

## Actions

1.  **Receive Context:** Get error details from the failed 'Determine Previous Git Tag' step (06).
2.  **Log Error:** Record the failure, indicating issues fetching tags or using `git describe`.
3.  **Prepare Failure Output:** Construct the `workflow_result` object indicating Failure, specifying `failed_step` '06'.
4.  **Terminate Workflow:** Signal workflow termination with a failure status.

## Acceptance Criteria

*   Error context is received.
*   The Git tag determination failure is logged.
*   A `workflow_result` object indicating Failure is generated.
*   Workflow terminates.

## Error Handling

*   Terminal error handling step.