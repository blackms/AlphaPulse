+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-EE-HANDLE-GIT-TAG-PUSH-ERROR" # (String, Required) Unique ID for this error handling step.
title = "Error Handler: Git Tag Push Failed" # (String, Required) Title of this specific step.
description = """
(String, Required) Handles errors encountered during the 'Push Git Tag' step (13).
Logs the error and prepares for workflow termination.
"""

# --- Flow Control ---
# This step is triggered by the 'error_step' field in '13_push_tag.md'.
depends_on = []
next_step = ""
# error_step = ""

# --- Execution ---
delegate_to = "lead-devops" # Or dev-git

# --- Interface ---
inputs = [
    "Input from step WF-CREATE-ROO-CMD-BUILD-V1-13-PUSH-TAG: tag_push_status", # Failure status
    "Input from step WF-CREATE-ROO-CMD-BUILD-V1-01-VALIDATE-PARAMS: validated_build_params", # Contains tag name
    # Potentially Git command output/error
]
outputs = [
    "workflow_result: Summary object indicating Failure, e.g., { overall_status: 'Failure', failed_step: '13', reason: 'Failed to push Git tag to remote', ... }.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}"
template_schema_doc = ".ruru/templates/toml-md/24_workflow_step_error.md"
+++

# Error Handler: Git Tag Push Failed

## Actions

1.  **Receive Context:** Get error details from the failed 'Push Git Tag' step (13).
2.  **Log Error:** Record the failure, indicating issues creating the tag locally (`git tag`) or pushing it to the remote (`git push origin <tag>`).
3.  **Prepare Failure Output:** Construct the `workflow_result` object indicating Failure, specifying `failed_step` '13'.
4.  **Terminate Workflow:** Signal workflow termination with a failure status.

## Acceptance Criteria

*   Error context is received.
*   The Git tag push failure is logged.
*   A `workflow_result` object indicating Failure is generated.
*   Workflow terminates.

## Error Handling

*   Terminal error handling step.