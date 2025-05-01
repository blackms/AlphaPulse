+++
# --- Step Metadata ---
step_id = "WF-MODE-DELETE-V1-EE-HANDLE-ERROR" # (String, Required) Unique ID for this step.
title = "Step EE: Handle Workflow Error" # (String, Required) Title of this specific step.
description = """
(String, Required) Handles errors encountered during the mode deletion workflow.
Logs the error details and reports failure. Also handles user cancellation.
"""

# --- Flow Control ---
depends_on = [] # (Array of Strings, Required) Can be triggered from any step via 'error_step'.
next_step = "" # (String, Required) Always empty for an error handling step.
error_step = "" # (String, Optional) No further error step.

# --- Execution ---
delegate_to = "prime-coordinator" # (String, Optional) Coordinator handles error reporting.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed.
    "error_details: Description of the error or cancellation reason.",
    "failed_step_id: The ID of the step that failed or where cancellation occurred.",
    "mode_slug: (If available) The target mode slug.",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "workflow_result: Summary of workflow failure or cancellation.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Using standard template as base.
+++

# Step EE: Handle Workflow Error

## Actions

(Coordinator `{{delegate_to}}` performs these actions)

1.  **Log Error/Cancellation:** Log the details received in the `error_details` input, including the `failed_step_id` and `mode_slug` if available. Clearly indicate if it was an error or a user cancellation. (Follow Rule `08-logging-procedure-simplified.md`).
2.  **Prepare Failure/Cancellation Report:** Construct the `workflow_result` output message.
    *   If an error occurred: "Workflow WF-MODE-DELETE-V1 failed at step `{{failed_step_id}}` while processing mode `{{mode_slug}}`. Error: {{error_details}}."
    *   If user cancelled: "Workflow WF-MODE-DELETE-V1 was cancelled by the user at step `{{failed_step_id}}` while processing mode `{{mode_slug}}`."
3.  **Report Outcome:** Use `<attempt_completion>` to report the `workflow_result` (failure/cancellation) back to the User or initiating process.

## Acceptance Criteria

*   The error or cancellation details are logged.
*   A failure or cancellation message (`workflow_result`) is generated.
*   Workflow failure/cancellation is reported via `<attempt_completion>`.

## Error Handling

*   If logging fails, attempt to report the original error/cancellation anyway.