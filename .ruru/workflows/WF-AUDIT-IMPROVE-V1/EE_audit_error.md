+++
# --- Basic Metadata (Auto-Generated) ---
id = "WF-AUDIT-IMPROVE-V1_EE_audit_error" # Unique ID for this step
step_number = -1 # Indicates an error step, using EE prefix convention
# --- Workflow Step Configuration ---
title = "Handle Workflow Audit Error"
description = "Handles errors encountered during the workflow audit process, logs the error, and terminates the workflow."
step_type = "error" # start, standard, finish, error
# --- Execution Details ---
# mode = "self" # Optional: Mode responsible for executing this step (defaults to workflow executor)
# delegate_to = "" # Optional: Delegate execution to another mode
# --- Input/Output ---
inputs = ["error_message", "failed_step_id", "audit_report_path", "iteration_count?"] # List of input variables required by this step (error context, iteration_count might not always be available)
outputs = ["audit_result", "final_report_path", "iterations_performed"] # List of output variables produced by this step (indicating failure)
# --- Control Flow ---
# next_step = "" # No next step for an error step that terminates
# error_step = "" # Cannot have an error step for an error step
# --- Logging & Auditing ---
log_level = "error" # Ensure errors are logged prominently
# --- Tags & Categorization ---
tags = ["error-handling", "failure", "termination", "logging"]
# --- Notes ---
# - This step is triggered when any preceding step transitions to its error_step.
# - It ensures the workflow terminates gracefully with a failure status.
+++

# Step EE: Handle Workflow Audit Error

## Description

This step is executed if any error occurs during the "Workflow Audit & Improvement" process. It logs the details of the error, attempts to save any partial information if relevant, sets the final workflow outputs to indicate failure, and terminates the workflow execution.

## Actions

1.  **Log Error:** Record the error details, including the `error_message`, the `failed_step_id`, and the `iteration_count` (if available) where the error occurred. Use a high log level (e.g., ERROR).
2.  **Attempt Partial Report (Optional):** If feasible and useful, attempt to compile and save a partial audit report to `audit_report_path` containing any findings gathered before the error occurred. This might involve reading temporary files if they exist. Handle potential errors during this reporting attempt gracefully.
3.  **Set Failure Outputs:**
    *   Set `audit_result` to a failure status (e.g., "Failed during audit"). Include the failed step ID if possible.
    *   Set `final_report_path` to the `audit_report_path` (even if only partially written or empty).
    *   Set `iterations_performed` to the current `iteration_count`.
4.  **Terminate Workflow:** Ensure no further steps are executed.

## Inputs

*   `error_message` (from the step that failed)
*   `failed_step_id` (ID of the step that failed)
*   `audit_report_path` (path for potential partial report)
*   `iteration_count` (optional, current count when error occurred)

## Outputs

*   `audit_result`: Failure status string (e.g., "Failed during audit at step [failed_step_id]").
*   `final_report_path`: Path to the (potentially partial) audit report file.
*   `iterations_performed`: Number of iterations completed before failure.

## Next Step

*   None (Workflow terminates).