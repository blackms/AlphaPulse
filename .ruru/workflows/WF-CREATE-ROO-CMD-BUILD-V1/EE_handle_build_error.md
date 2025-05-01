+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-EE-BUILD" # (String, Required) Unique ID for this error step.
title = "Error Handler: Build Command Failure" # (String, Required) Title of this specific step.
description = """
(String, Required) Handles errors occurring during the '03_run_build' step
(e.g., compilation errors, non-zero exit code from build command).
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
    "ErrorContext: Details about the failure from step 03.",
    "build_log: Path to the build log file containing error details.",
    "build_directory_path: Path where the build was attempted.",
]
outputs = [ # (Array of Strings, Optional) Usually focuses on logging or notification status.
    "error_handling_report: Confirmation of actions taken.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/27_workflow_step_error.md" # (String, Required) Hypothetical link to an error step template definition.
+++

# Error Handler: Build Command Failure

## Actions

1.  **Log Error:** Record detailed error information from `ErrorContext` and reference the `build_log`.
2.  **Notify:** Alert the relevant coordinator or role (e.g., `lead-devops`), providing the build log path.
3.  **Cleanup:** Perform cleanup within the `build_directory_path` if necessary.
4.  **Set Final Status:** Ensure the overall workflow status reflects failure.

*(Placeholder - Requires implementation)*