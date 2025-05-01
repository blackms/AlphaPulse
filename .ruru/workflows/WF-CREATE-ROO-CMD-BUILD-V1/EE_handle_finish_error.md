+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-EE-FINISH" # (String, Required) Unique ID for this error step.
title = "Error Handler: Finish Step Failure" # (String, Required) Title of this specific step.
description = """
(String, Required) Handles errors occurring during the '99_finish' step
(e.g., failure during cleanup, aggregation, or final reporting).
Placeholder: Logic needs to be defined (e.g., logging, notification).
"""

# --- Flow Control ---
# Error steps typically terminate the workflow or attempt specific recovery.
depends_on = [] # (Array of Strings, Required) Usually empty or references the step that triggered it.
next_step = "" # (String, Required) Usually empty for termination.
error_step = "" # (String, Optional) Avoid recursive error handling unless carefully designed.

# --- Execution ---
delegate_to = "lead-devops" # (String, Optional) Mode responsible for executing error handling logic.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data needed, e.g., error message, context from failing step.
    "ErrorContext: Details about the failure from step 99.",
    "AggregatedResults: Data collected before the finish step failed (if available).",
]
outputs = [ # (Array of Strings, Optional) Usually focuses on logging or notification status.
    "error_handling_report: Confirmation of actions taken.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/27_workflow_step_error.md" # (String, Required) Hypothetical link to an error step template definition.
+++

# Error Handler: Finish Step Failure

## Actions

1.  **Log Error:** Record detailed error information from `ErrorContext`. Log whatever `AggregatedResults` were available.
2.  **Notify:** Alert the relevant coordinator or role (e.g., `lead-devops`) about the failure during finalization.
3.  **Cleanup:** Attempt critical cleanup if possible, but be cautious as the state might be inconsistent.
4.  **Set Final Status:** Ensure the overall workflow status reflects failure.

*(Placeholder - Requires implementation)*