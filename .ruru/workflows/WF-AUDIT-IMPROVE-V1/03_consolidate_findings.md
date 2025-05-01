+++
# --- Basic Metadata (Auto-Generated) ---
id = "WF-AUDIT-IMPROVE-V1_03_consolidate_findings" # Unique ID for this step
step_number = 4 # Step sequence number
# --- Workflow Step Configuration ---
title = "Consolidate Audit Findings"
description = "Aggregates the results from structural analysis and logic/quality review into a single structured report."
step_type = "standard" # start, standard, finish, error
# --- Execution Details ---
# mode = "self" # Optional: Mode responsible for executing this step (defaults to workflow executor)
# delegate_to = "" # Optional: Delegate execution to another mode
# --- Input/Output ---
inputs = ["structure_analysis_results", "logic_analysis_results", "consolidated_findings_path"] # List of input variables required by this step
outputs = ["consolidated_findings_path"] # List of output variables produced by this step (path to the saved consolidated report)
# --- Control Flow ---
next_step = "04_request_approval.md" # The next step file to execute
error_step = "EE_audit_error.md" # Optional: Step file to execute on error
# --- Logging & Auditing ---
# log_level = "info" # Default logging level for this step
# --- Tags & Categorization ---
tags = ["aggregation", "consolidation", "reporting", "analysis-results"]
# --- Notes ---
# - Merges findings from 02a and 02b.
# - Output format could be JSON or a structured Markdown file.
# - Ensures actionable suggestions are clearly presented.
+++

# Step 03: Consolidate Audit Findings

## Description

This step takes the outputs from the structural analysis (`02a`) and the logic/quality review (`02b`) and combines them into a single, coherent report. This consolidated report will contain all identified issues and actionable suggestions for improvement, ready for presentation to the user/coordinator for approval.

## Actions

1.  **Parse Analysis Results:** Process the `structure_analysis_results` and `logic_analysis_results` variables, which contain the findings from the previous steps.
2.  **Combine Findings:** Merge the findings from both analyses into a unified structure. This could involve:
    *   Grouping findings by file or by type of issue (e.g., formatting, logic, efficiency).
    *   Clearly listing each issue and the corresponding suggestion(s).
    *   Prioritizing suggestions if possible.
    *   Formatting the combined findings into a structured format (e.g., JSON object, detailed Markdown).
3.  **Save Consolidated Report:** Use `write_to_file` to save the combined findings to the path specified in the `consolidated_findings_path` variable.
4.  **Log Consolidation:** Record that the findings have been consolidated and saved.

## Inputs

*   `structure_analysis_results` (from step 02a)
*   `logic_analysis_results` (from step 02b)
*   `consolidated_findings_path` (from step 00)

## Outputs

*   `consolidated_findings_path`: The path to the file containing the consolidated audit findings and suggestions.

## Next Step

*   Proceed to `04_request_approval.md`.

## Error Handling

*   If parsing or combining results fails, transition to `EE_audit_error.md`.
*   If writing the consolidated report fails, transition to `EE_audit_error.md`.