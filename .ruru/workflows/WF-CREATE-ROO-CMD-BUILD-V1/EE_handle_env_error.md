+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-EE-ENV" # (String, Required) Unique ID for this error step.
title = "Error Handler: Environment Setup Failure" # (String, Required) Title of this specific step.
description = """
(String, Required) Handles errors occurring during the '02_setup_environment' step
(e.g., source checkout failure, dependency installation failure).
Placeholder: Logic needs to be defined (e.g., logging, notification, cleanup).
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
    "ErrorContext: Details about the failure from step 02.",
    "validated_build_params: Parameters used during the failed setup.",
]
outputs = [ # (Array of Strings, Optional) Usually focuses on logging or notification status.
    "error_handling_report: Confirmation of actions taken.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/27_workflow_step_error.md" # (String, Required) Hypothetical link to an error step template definition.
+++

# Error Handler: Environment Setup Failure

## Actions

1.  **Log Error:** Record detailed error information from the `ErrorContext` input (e.g., checkout error, dependency install log).
2.  **Notify:** Alert the relevant coordinator or role (e.g., `lead-devops`).
3.  **Cleanup:** Perform cleanup (e.g., remove partially checked out code, incomplete dependencies).
4.  **Set Final Status:** Ensure the overall workflow status reflects failure.

*(Placeholder - Requires implementation)*