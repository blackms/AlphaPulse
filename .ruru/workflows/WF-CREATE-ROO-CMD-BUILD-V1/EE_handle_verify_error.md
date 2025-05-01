+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-EE-HANDLE-VERIFY-ERROR" # (String, Required) Unique ID for this error handling step.
title = "Error Handler: Artifact Verification Failed" # (String, Required) Title of this specific step.
description = """
(String, Required) Handles errors encountered during the 'Verify Build Artifacts' step (04).
Logs the error and prepares for workflow termination.
"""

# --- Flow Control ---
# This step is triggered by the 'error_step' field in '04_verify_artifacts.md'.
depends_on = [] # Error handlers typically don't depend on successful completion of other steps.
next_step = "" # (String, Required) Always empty for error handlers leading to termination.
# error_step = "" # (String, Optional) Typically not defined for final error handlers.

# --- Execution ---
delegate_to = "lead-devops" # (String, Optional) Mode responsible for logging/reporting the error.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Might include error details or context from the failed step.
    "Input from step WF-CREATE-ROO-CMD-BUILD-V1-04-VERIFY-ARTIFACTS: artifacts_verified_status", # The failure status
    "Input from step WF-CREATE-ROO-CMD-BUILD-V1-03-RUN-BUILDS: raw_artifacts_path", # Context for the error
    # Potentially other context like build logs
]
outputs = [ # (Array of Strings, Optional) Outputs indicating failure.
    "workflow_result: Summary object indicating Failure, e.g., { overall_status: 'Failure', failed_step: '04', reason: 'Artifact verification failed', ... }.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/24_workflow_step_error.md" # (String, Required) Link to this template definition.
+++

# Error Handler: Artifact Verification Failed

## Actions

1.  **Receive Context:** Get error details and context from the failed 'Verify Build Artifacts' step (04).
2.  **Log Error:** Record the failure details, indicating that expected build artifacts were missing or invalid in the `raw_artifacts_path`.
3.  **Prepare Failure Output:** Construct the `workflow_result` object, setting `overall_status` to 'Failure', specifying the `failed_step` as '04', and providing a `reason`.
4.  **Terminate Workflow:** Signal workflow termination with a failure status.

## Acceptance Criteria

*   Error context is received.
*   The verification failure is logged appropriately.
*   A `workflow_result` object indicating Failure is generated.
*   Workflow terminates.

## Error Handling

*   This is a terminal error handling step. Further errors here might require manual intervention or a meta-error handler if designed.