+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-EE-HANDLE-GIT-COMMIT-ERROR" # (String, Required) Unique ID for this error handling step.
title = "Error Handler: Git Commit Failed" # (String, Required) Title of this specific step.
description = """
(String, Required) Handles errors encountered during the 'Commit Documentation Changes' step (12).
Logs the error and prepares for workflow termination.
"""

# --- Flow Control ---
# This step is triggered by the 'error_step' field in '12_commit_docs.md'.
depends_on = []
next_step = ""
# error_step = ""

# --- Execution ---
delegate_to = "lead-devops" # Or dev-git

# --- Interface ---
inputs = [
    "Input from step WF-CREATE-ROO-CMD-BUILD-V1-12-COMMIT-DOCS: commit_status", # Failure status
    # Potentially paths to files that failed staging/commit
]
outputs = [
    "workflow_result: Summary object indicating Failure, e.g., { overall_status: 'Failure', failed_step: '12', reason: 'Failed to commit documentation changes to Git', ... }.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}"
template_schema_doc = ".ruru/templates/toml-md/24_workflow_step_error.md"
+++

# Error Handler: Git Commit Failed

## Actions

1.  **Receive Context:** Get error details from the failed 'Commit Documentation Changes' step (12).
2.  **Log Error:** Record the failure, indicating issues staging files (`git add`) or committing (`git commit`).
3.  **Prepare Failure Output:** Construct the `workflow_result` object indicating Failure, specifying `failed_step` '12'.
4.  **Terminate Workflow:** Signal workflow termination with a failure status.

## Acceptance Criteria

*   Error context is received.
*   The Git commit failure is logged.
*   A `workflow_result` object indicating Failure is generated.
*   Workflow terminates.

## Error Handling

*   Terminal error handling step.