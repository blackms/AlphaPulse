+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-EE-HANDLE-GITHUB-RELEASE-ERROR" # (String, Required) Unique ID for this error handling step.
title = "Error Handler: GitHub Release Creation Failed" # (String, Required) Title of this specific step.
description = """
(String, Required) Handles errors encountered during the 'Create GitHub Release' step (14).
Logs the error and prepares for workflow termination. Note: This might be triggered by prerequisite failures too.
"""

# --- Flow Control ---
# This step is triggered by the 'error_step' field in '14_create_github_release.md'.
depends_on = []
next_step = ""
# error_step = ""

# --- Execution ---
delegate_to = "lead-devops"

# --- Interface ---
inputs = [
    "Input from step WF-CREATE-ROO-CMD-BUILD-V1-14-CREATE-GITHUB-RELEASE: github_release_status", # Failure status
    "Input from step WF-CREATE-ROO-CMD-BUILD-V1-13-PUSH-TAG: tag_push_status", # Check if prerequisite failed
    # Potentially API error messages, tag name, artifact path
]
outputs = [
    "workflow_result: Summary object indicating Failure, e.g., { overall_status: 'Failure', failed_step: '14', reason: 'Failed to create GitHub release', ... }.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}"
template_schema_doc = ".ruru/templates/toml-md/24_workflow_step_error.md"
+++

# Error Handler: GitHub Release Creation Failed

## Actions

1.  **Receive Context:** Get error details from the failed 'Create GitHub Release' step (14) or prerequisite steps.
2.  **Log Error:** Record the failure, indicating issues with the GitHub API/CLI, artifact upload, or prerequisite failures (like tag push).
3.  **Prepare Failure Output:** Construct the `workflow_result` object indicating Failure, specifying `failed_step` '14'.
4.  **Terminate Workflow:** Signal workflow termination with a failure status.

## Acceptance Criteria

*   Error context is received.
*   The GitHub release failure is logged.
*   A `workflow_result` object indicating Failure is generated.
*   Workflow terminates.

## Error Handling

*   Terminal error handling step.