+++
# --- Basic Metadata (Auto-Generated) ---
id = "WF-AUDIT-IMPROVE-V1_00_start" # Unique ID for this step
step_number = 0 # Step sequence number
# --- Workflow Step Configuration ---
title = "Initialize Workflow Audit"
description = "Starts the workflow audit process, validates input, and sets initial state."
step_type = "start" # start, standard, finish, error
# --- Execution Details ---
# mode = "self" # Optional: Mode responsible for executing this step (defaults to workflow executor)
# --- Input/Output ---
inputs = ["target_workflow_dir", "max_iterations?", "audit_report_path?", "consolidated_findings_path?", "approved_changes_path?"] # List of input variables required by this step (optional indicated by ?)
outputs = ["target_workflow_dir", "iteration_count", "max_iterations", "audit_report_path", "consolidated_findings_path", "approved_changes_path"] # List of output variables produced by this step
# --- Control Flow ---
next_step = "01_gather_context.md" # The next step file to execute
error_step = "EE_audit_error.md" # Step file to execute on error
# --- Logging & Auditing ---
# log_level = "info" # Default logging level for this step
# --- Tags & Categorization ---
tags = ["init", "validation", "setup"]
# --- Notes ---
# - Validates the presence of the required 'target_workflow_dir' input.
# - Initializes iteration counter and other state variables.
+++

# Step 00: Initialize Workflow Audit

## Description

This step initiates the "Workflow Audit & Improvement" process. It validates the essential input (`target_workflow_dir`) and sets up the initial state variables required for the subsequent steps, including the iteration counter and paths for temporary/output files.

## Actions

1.  **Validate Input:** Check if the `target_workflow_dir` input variable is provided and potentially if the directory exists (basic check). If validation fails, transition to the `error_step`.
2.  **Initialize State:**
    *   Set `iteration_count` to 0 (or retrieve from input if resuming).
    *   Confirm `max_iterations` (default or from input).
    *   Confirm paths for `audit_report_path`, `consolidated_findings_path`, `approved_changes_path` (defaults or from input). Ensure necessary parent directories exist or can be created.
3.  **Log Initialization:** Record the start of the workflow audit, including the target directory.

## Inputs

*   `target_workflow_dir` (from workflow inputs)
*   `max_iterations` (optional, from workflow inputs)
*   `audit_report_path` (optional, from workflow inputs)
*   `consolidated_findings_path` (optional, from workflow inputs)
*   `approved_changes_path` (optional, from workflow inputs)

## Outputs

*   `target_workflow_dir` (validated)
*   `iteration_count` (initialized to 0)
*   `max_iterations` (confirmed)
*   `audit_report_path` (confirmed)
*   `consolidated_findings_path` (confirmed)
*   `approved_changes_path` (confirmed)

## Next Step

*   Proceed to `01_gather_context.md`.

## Error Handling

*   If `target_workflow_dir` is missing or invalid, transition to `EE_audit_error.md`.
*   If file path setup fails, transition to `EE_audit_error.md`.