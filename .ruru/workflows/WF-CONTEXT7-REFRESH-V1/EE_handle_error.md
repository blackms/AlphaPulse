+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-REFRESH-V1-EE-HANDLE-ERROR" # (String, Required) Unique ID for this step.
title = "Step EE: Handle Workflow Error" # (String, Required) Title of this specific step.
description = """
(String, Required) Handles errors encountered during the workflow execution.
Logs the error details and reports failure.
"""

# --- Flow Control ---
depends_on = [] # (Array of Strings, Required) Can be triggered from any step via 'error_step'.
next_step = "" # (String, Required) Always empty for an error handling step.
error_step = "" # (String, Optional) No further error step.

# --- Execution ---
delegate_to = "roo-commander" # (String, Optional) Coordinator handles error reporting.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed.
    "error_details: Description of the error encountered.",
    "failed_step_id: The ID of the step that failed.",
    "mode_slug: (If available) The target mode slug.",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "workflow_result: Summary of workflow failure.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Using standard template as base.
+++

# Step EE: Handle Workflow Error

## Actions

(Coordinator `{{delegate_to}}` performs these actions)

1.  **Log Error:** Log the details of the error received in the `error_details` input, including the `failed_step_id`. (Follow Rule `08-logging-procedure-simplified.md`).
2.  **Prepare Failure Report:** Construct the `workflow_result` output message indicating failure.
    *   Example: "Workflow WF-CONTEXT7-REFRESH-V1 failed at step `{{failed_step_id}}` for mode `{{mode_slug}}`. Error: {{error_details}}."
3.  **Report Failure:** Use `<attempt_completion>` to report the `workflow_result` (failure) back to the User or initiating process.

## Acceptance Criteria

*   The error details are logged.
*   A failure message (`workflow_result`) is generated.
*   Workflow failure is reported via `<attempt_completion>`.

## Error Handling

*   If logging fails, attempt to report the original error anyway.
