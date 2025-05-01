+++
# --- Basic Metadata (Auto-Generated) ---
id = "WF-AUDIT-IMPROVE-V1_02a_check_structure" # Unique ID for this step
step_number = 2 # Step sequence number
# --- Workflow Step Configuration ---
title = "Analyze Workflow Structure & Format"
description = "Delegates structural validation (file names, TOML+MD format, required fields based on rules/templates) to util-workflow-manager."
step_type = "standard" # start, standard, finish, error
# --- Execution Details ---
# mode = "self" # Optional: Mode responsible for executing this step (defaults to workflow executor)
delegate_to = "util-workflow-manager" # Delegate execution to another mode
# --- Input/Output ---
inputs = ["target_workflow_dir", "workflow_files_content", "rules_content", "templates_content"] # List of input variables required by this step
outputs = ["structure_analysis_results"] # List of output variables produced by this step (e.g., a report or structured data)
# --- Control Flow ---
next_step = "02b_review_logic.md" # The next step file to execute
error_step = "EE_audit_error.md" # Optional: Step file to execute on error
# --- Logging & Auditing ---
# log_level = "info" # Default logging level for this step
# --- Tags & Categorization ---
tags = ["analysis", "validation", "structure", "format", "toml", "delegation", "compliance"]
# --- Notes ---
# - Assumes util-workflow-manager can accept workflow content, rules, and templates as input.
# - Expected output is a structured report detailing compliance issues and suggestions.
+++

# Step 02a: Analyze Workflow Structure & Format

## Description

This step focuses on validating the structural integrity and format compliance of the target workflow definition files. It delegates this task to the `util-workflow-manager` mode, providing the necessary context gathered in the previous step.

## Actions

1.  **Prepare Delegation Context:** Package the `workflow_files_content`, relevant `rules_content` (especially `01-standard-toml-md-format.md`), and `templates_content` into a format suitable for the `util-workflow-manager`. Include the `target_workflow_dir` path for reference.
2.  **Delegate Task:** Use `new_task` to delegate the structural analysis to `util-workflow-manager`. The message should clearly instruct the mode to:
    *   Validate file naming conventions (`NN_*.md`, `EE_*.md`, `README.md`).
    *   Verify TOML+MD format compliance (presence and syntax of `+++` blocks) for all `.md` files.
    *   Check for the presence and basic validity of required TOML fields in each step file, comparing against the provided standard templates (`24_*.md`, `25_*.md`, `26_*.md`) and rules.
    *   Report findings as structured data (e.g., JSON or Markdown list) detailing any violations or deviations found, along with the file path and specific issue.
3.  **Receive Results:** Store the analysis results received from `util-workflow-manager` in the `structure_analysis_results` output variable.
4.  **Log Delegation & Results:** Record the delegation action and the receipt of the analysis results.

## Inputs

*   `target_workflow_dir` (from step 00)
*   `workflow_files_content` (from step 01)
*   `rules_content` (from step 01)
*   `templates_content` (from step 01)

## Outputs

*   `structure_analysis_results`: Structured data (e.g., JSON string or Markdown report) containing the findings from the structural and format validation.

## Next Step

*   Proceed to `02b_review_logic.md`.

## Error Handling

*   If delegation to `util-workflow-manager` fails or the mode returns an error, transition to `EE_audit_error.md`.