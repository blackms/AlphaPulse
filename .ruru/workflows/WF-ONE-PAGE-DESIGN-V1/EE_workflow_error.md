+++
# --- Step Metadata ---
step_id = "WF-ONE-PAGE-DESIGN-V1-EE-WORKFLOW-ERROR"
title = "Step EE: Handle Workflow Error"
description = """
Handles errors encountered during the workflow execution. Logs the error
and reports failure. Specific recovery logic is not implemented in this placeholder.
"""

# --- Flow Control ---
depends_on = [] # Triggered by 'error_step' field in other steps
next_step = "" # Error step usually terminates or goes to a final notification step
error_step = "" # Avoid error loops

# --- Execution ---
delegate_to = "" # Orchestrator likely handles this

# --- Interface ---
inputs = [
    "error_details: Information about the error that occurred (from the failing step).",
]
outputs = [
    "workflow_result: Summary indicating failure and including error details.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}"
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # Based on standard step
+++

# Step EE: Handle Workflow Error

## Actions

1.  **Receive Error Details:** Obtain the `error_details` from the step that triggered the error.
2.  **Log Error:** Record the error details in appropriate logs.
3.  **Prepare Failure Report:** Format a `workflow_result` output indicating failure and including the `error_details`.
4.  **Terminate Workflow:** Signal workflow termination due to error.

## Acceptance Criteria

*   Error details are received.
*   Error is logged.
*   A failure `workflow_result` is prepared.

## Error Handling

*   If logging fails, attempt to report the original error anyway.