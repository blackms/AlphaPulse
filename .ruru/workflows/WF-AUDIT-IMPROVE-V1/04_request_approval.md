+++
# --- Basic Metadata (Auto-Generated) ---
id = "WF-AUDIT-IMPROVE-V1_04_request_approval" # Unique ID for this step
step_number = 5 # Step sequence number
# --- Workflow Step Configuration ---
title = "Request Approval for Improvements"
description = "Presents consolidated findings and suggestions to the user/coordinator for approval and determines if changes should be applied."
step_type = "standard" # start, standard, finish, error
# --- Execution Details ---
# mode = "self" # This step likely requires interaction, potentially handled by the workflow executor or a coordinator mode.
# delegate_to = "" # Optional: Delegate execution to another mode
# --- Input/Output ---
inputs = ["consolidated_findings_path", "iteration_count", "max_iterations", "approved_changes_path"] # List of input variables required by this step
outputs = ["approved_changes_path", "proceed_with_changes"] # Path to approved changes file and a boolean flag
# --- Control Flow ---
# next_step = "" # Determined conditionally below
error_step = "EE_audit_error.md" # Optional: Step file to execute on error
conditional_next_steps = [
    { condition = "{{ proceed_with_changes == true and iteration_count < max_iterations }}", next_step = "05_apply_changes.md" },
    # Default case (no changes approved OR max iterations reached)
    { condition = "default", next_step = "06_simulate_flow.md" }
]
# --- Logging & Auditing ---
# log_level = "info" # Default logging level for this step
# --- Tags & Categorization ---
tags = ["approval", "interaction", "decision", "iteration-control"]
# --- Notes ---
# - This step requires interaction with the user/coordinator.
# - The mechanism might involve `ask_followup_question` or presenting the findings file for manual review/edit.
# - The output `approved_changes_path` should contain only the changes the user agreed to implement.
# - Checks iteration count against max_iterations.
+++

# Step 04: Request Approval for Improvements

## Description

This step presents the consolidated findings and actionable suggestions (generated in step 03) to the initiating user or coordinator. It seeks approval on which, if any, of the suggested improvements should be applied to the target workflow definition. It also checks if the maximum number of improvement iterations has been reached.

## Actions

1.  **Read Findings:** Load the consolidated findings from the file specified by `consolidated_findings_path`.
2.  **Present Findings & Request Approval:** Display the findings and suggestions to the user/coordinator. The primary mechanism should be using the `ask_followup_question` tool with a summary and actionable suggestions (e.g., "Approve all suggestions?", "Approve suggestions [1, 3, 5]?", "Approve no changes?"). Alternatively, direct the user to review the `consolidated_findings_path` file and provide explicit approval input.
3.  **Receive Approval:** Capture the user's/coordinator's decision regarding which suggestions are approved for implementation.
4.  **Store Approved Changes:** Filter the suggestions based on the approval received and save only the approved changes (in a structured format specifying the file, location, and change needed) to the path specified by `approved_changes_path`.
5.  **Set Control Flag:** Set the `proceed_with_changes` output variable to `true` if any changes were approved, otherwise set it to `false`.
6.  **Log Decision:** Record the outcome of the approval step (which changes were approved, if any).

## Inputs

*   `consolidated_findings_path` (from step 03)
*   `iteration_count` (from step 00 or 05)
*   `max_iterations` (from step 00)
*   `approved_changes_path` (from step 00)

## Outputs

*   `approved_changes_path`: Path to the file containing only the *approved* changes/suggestions.
*   `proceed_with_changes`: Boolean flag indicating whether any changes were approved (`true`) or not (`false`).

## Next Step

*   **If `proceed_with_changes` is `true` AND `iteration_count < max_iterations`:** Proceed to `05_apply_changes.md`.
*   **Otherwise (no changes approved OR `iteration_count >= max_iterations`):** Proceed to `06_simulate_flow.md`.

## Error Handling

*   If reading findings fails, transition to `EE_audit_error.md`.
*   If interaction with the user/coordinator fails or times out, transition to `EE_audit_error.md`.
*   If saving approved changes fails, transition to `EE_audit_error.md`.