+++
# --- Basic Metadata (Auto-Generated) ---
id = "WF-AUDIT-IMPROVE-V1_99_finish" # Unique ID for this step
step_number = 8 # Step sequence number
# --- Workflow Step Configuration ---
title = "Finalize Audit and Report"
description = "Generates the final audit report, consolidating all findings, changes, and simulation results, and concludes the workflow."
step_type = "finish" # start, standard, finish, error
# --- Execution Details ---
# mode = "self" # Optional: Mode responsible for executing this step (defaults to workflow executor)
# delegate_to = "" # Optional: Delegate execution to another mode
# --- Input/Output ---
inputs = [
    "target_workflow_dir",
    "consolidated_findings_path", # Path to initial findings (might need re-reading if not passed through)
    "approved_changes_path", # Path to applied changes (might need re-reading if not passed through)
    "simulation_results",
    "iteration_count",
    "audit_report_path"
] # List of input variables required by this step
outputs = ["audit_result", "final_report_path", "iterations_performed"] # List of output variables produced by this step
# --- Control Flow ---
# next_step = "" # No next step for a finish step
error_step = "EE_audit_error.md" # Optional: Step file to execute on error (though less likely needed here)
# --- Logging & Auditing ---
# log_level = "info" # Default logging level for this step
# --- Tags & Categorization ---
tags = ["finish", "reporting", "summary", "cleanup"]
# --- Notes ---
# - Compiles information from previous steps into a final report.
# - Cleans up temporary files if necessary.
# - Sets the final workflow output variables.
+++

# Step 99: Finalize Audit and Report

## Description

This is the final step of the "Workflow Audit & Improvement" process. It gathers all the results generated throughout the workflow – initial findings, approved/applied changes, and static simulation results – compiles them into a comprehensive final audit report, saves the report, and sets the final output variables for the workflow.

## Actions

1.  **Read Input Data:** Load the content from:
    *   `consolidated_findings_path` (contains the initial findings presented for approval).
    *   `approved_changes_path` (contains the changes that were actually applied).
    *   `simulation_results` (contains the results of the static flow check).
2.  **Compile Final Report:** Create a comprehensive Markdown report that includes:
    *   The target workflow directory (`target_workflow_dir`).
    *   The number of improvement iterations performed (`iteration_count`).
    *   A summary of the initial findings (from `consolidated_findings_path`).
    *   A list or summary of the changes that were approved and applied (from `approved_changes_path`). If no changes were applied, state this clearly.
    *   The results of the static flow simulation (`simulation_results`).
    *   A final assessment or status (e.g., "Audit complete, N changes applied.", "Audit complete, no changes needed.").
3.  **Save Final Report:** Use `write_to_file` to save the compiled report to the path specified in `audit_report_path`.
4.  **Cleanup (Optional):** Delete temporary files like `consolidated_findings_path` and `approved_changes_path` if desired.
5.  **Set Final Outputs:**
    *   Set `audit_result` based on whether changes were made (e.g., "Completed with improvements", "Completed, no changes needed").
    *   Set `final_report_path` to the value of `audit_report_path`.
    *   Set `iterations_performed` to the final value of `iteration_count`.
6.  **Log Completion:** Record the successful completion of the workflow audit and the location of the final report.

## Inputs

*   `target_workflow_dir` (from step 00)
*   `consolidated_findings_path` (path from step 03)
*   `approved_changes_path` (path from step 04)
*   `simulation_results` (from step 06)
*   `iteration_count` (final count after loops)
*   `audit_report_path` (from step 00)

## Outputs

*   `audit_result`: Summary status string (e.g., "Completed with improvements").
*   `final_report_path`: Path to the final audit report file.
*   `iterations_performed`: Total number of improvement iterations performed.

## Next Step

*   None (Workflow finishes).

## Error Handling

*   If reading input files fails, transition to `EE_audit_error.md`.
*   If writing the final report fails, transition to `EE_audit_error.md`.