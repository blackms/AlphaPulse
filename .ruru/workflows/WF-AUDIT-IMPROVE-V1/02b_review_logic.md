+++
# --- Basic Metadata (Auto-Generated) ---
id = "WF-AUDIT-IMPROVE-V1_02b_review_logic" # Unique ID for this step
step_number = 3 # Step sequence number
# --- Workflow Step Configuration ---
title = "Review Workflow Logic & Quality"
description = "Delegates review of workflow logic, efficiency, error handling, and best practices to util-reviewer, providing relevant context."
step_type = "standard" # start, standard, finish, error
# --- Execution Details ---
# mode = "self" # Optional: Mode responsible for executing this step (defaults to workflow executor)
delegate_to = "util-reviewer" # Delegate execution to another mode (could also be agent-research + Vertex AI)
# --- Input/Output ---
inputs = ["target_workflow_dir", "workflow_files_content", "rules_content", "templates_content", "available_modes"] # List of input variables required by this step (available_modes needed for delegate_to validation)
outputs = ["logic_analysis_results"] # List of output variables produced by this step (e.g., a report or structured data)
# --- Control Flow ---
next_step = "03_consolidate_findings.md" # The next step file to execute
error_step = "EE_audit_error.md" # Optional: Step file to execute on error
# --- Logging & Auditing ---
# log_level = "info" # Default logging level for this step
# --- Tags & Categorization ---
tags = ["analysis", "review", "logic", "efficiency", "best-practice", "delegation", "quality"]
# --- Notes ---
# - Provides workflow files, rules, templates, and available modes list to the reviewer.
# - Instructs reviewer to check flow control, dependencies, clarity, efficiency, error handling, and best practices.
# - References the example review output: .ruru/docs/vertex/answers-direct/20250430093530-WF_ONE_PAGE_DESIGN_V1_review.md
# - Consider using agent-research + Vertex AI (e.g., answer_query_direct) as an alternative or supplement if util-reviewer lacks depth.
+++

# Step 02b: Review Workflow Logic & Quality

## Description

This step performs a deeper analysis of the target workflow's logic, efficiency, error handling, and adherence to best practices. It delegates this review to the `util-reviewer` mode, providing the workflow definition files and relevant contextual information (rules, templates, available modes).

## Actions

1.  **Prepare Delegation Context:** Package the `workflow_files_content`, relevant `rules_content`, `templates_content`, and the list of `available_modes` (for validating `delegate_to` fields) into a format suitable for the `util-reviewer`. Include the `target_workflow_dir` path for reference.
2.  **Delegate Task:** Use `new_task` to delegate the logic and quality review to `util-reviewer`. The message should clearly instruct the mode to:
    *   Analyze the step sequence and control flow (`next_step`, `conditional_next_steps`, `error_step`).
    *   Evaluate step dependencies (`depends_on`) and input/output consistency between steps.
    *   Assess the clarity and actionability of step descriptions and actions.
    *   Identify potential inefficiencies, redundancies, or overly complex logic.
    *   Review error handling strategy (`error_step` usage).
    *   Check adherence to general workflow design best practices and relevant project rules.
    *   Validate that all `delegate_to` fields reference modes present in the `available_modes` list.
    *   Format the review output similar in depth and structure to the example: `.ruru/docs/vertex/answers-direct/20250430093530-WF_ONE_PAGE_DESIGN_V1_review.md`.
    *   Provide specific, actionable suggestions for improvement.
3.  **Receive Results:** Store the analysis results received from `util-reviewer` in the `logic_analysis_results` output variable.
4.  **Log Delegation & Results:** Record the delegation action and the receipt of the analysis results.

## Inputs

*   `target_workflow_dir` (from step 00)
*   `workflow_files_content` (from step 01)
*   `rules_content` (from step 01)
*   `templates_content` (from step 01)
*   `available_modes` (List of available mode slugs, assumed available in workflow context)

## Outputs

*   `logic_analysis_results`: Structured data or Markdown report containing the findings and suggestions from the logic and quality review.

## Next Step

*   Proceed to `03_consolidate_findings.md`.

## Error Handling

*   If delegation to `util-reviewer` fails or the mode returns an error, transition to `EE_audit_error.md`.