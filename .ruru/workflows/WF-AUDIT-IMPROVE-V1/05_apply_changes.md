+++
# --- Basic Metadata (Auto-Generated) ---
id = "WF-AUDIT-IMPROVE-V1_05_apply_changes" # Unique ID for this step
step_number = 6 # Step sequence number
# --- Workflow Step Configuration ---
title = "Apply Approved Workflow Changes"
description = "Delegates the application of approved changes (from the approval step) to the target workflow files using util-workflow-manager."
step_type = "standard" # start, standard, finish, error
# --- Execution Details ---
# mode = "self" # Optional: Mode responsible for executing this step (defaults to workflow executor)
delegate_to = "util-workflow-manager" # Delegate execution to another mode
# --- Input/Output ---
inputs = ["approved_changes_path", "target_workflow_dir", "iteration_count"] # List of input variables required by this step
outputs = ["iteration_count"] # List of output variables produced by this step (incremented count)
# --- Control Flow ---
next_step = "01_gather_context.md" # Loop back to gather context and re-analyze after changes
error_step = "EE_audit_error.md" # Optional: Step file to execute on error
# --- Logging & Auditing ---
# log_level = "info" # Default logging level for this step
# --- Tags & Categorization ---
tags = ["modification", "apply-changes", "delegation", "iteration", "refactor"]
# --- Notes ---
# - This step only runs if changes were approved in step 04 and iteration limit not reached.
# - Assumes util-workflow-manager can parse the approved changes file and apply them using precise tools like apply_diff.
# - Increments the iteration counter.
+++

# Step 05: Apply Approved Workflow Changes

## Description

This step orchestrates the application of the specific improvements that were approved in the previous step (`04_request_approval.md`). It reads the file containing the approved changes and delegates the modification task to the `util-workflow-manager`, instructing it to use precise file editing tools. After delegation, it increments the iteration counter and loops back to the context gathering step to re-analyze the modified workflow.

## Actions

1.  **Read Approved Changes:** Load the structured list of approved changes from the file specified by `approved_changes_path`.
2.  **Prepare Delegation Context:** Package the `approved_changes_path` content and the `target_workflow_dir` path for the `util-workflow-manager`.
3.  **Delegate Task:** Use `new_task` to delegate the application of changes to `util-workflow-manager`. The message should clearly instruct the mode to:
    *   Parse the approved changes file (`approved_changes_path`).
    *   For each approved change, apply it to the corresponding file within the `target_workflow_dir` using precise tools like `apply_diff` or `search_and_replace`. **Avoid** using `write_to_file` unless absolutely necessary for a specific change type.
    *   Report success or failure for each applied change.
4.  **Receive Confirmation:** Await confirmation from `util-workflow-manager` that the changes have been attempted/applied.
5.  **Increment Iteration Count:** Increase the `iteration_count` variable by 1.
6.  **Log Application & Iteration:** Record the delegation action, the confirmation received, and the incremented iteration count.

## Inputs

*   `approved_changes_path` (from step 04)
*   `target_workflow_dir` (from step 00)
*   `iteration_count` (from step 04)

## Outputs

*   `iteration_count`: The incremented iteration counter.

## Next Step

*   Proceed to `01_gather_context.md` to start the next iteration of analysis on the modified workflow.

## Error Handling

*   If reading the approved changes file fails, transition to `EE_audit_error.md`.
*   If delegation to `util-workflow-manager` fails or the mode reports failure in applying changes, transition to `EE_audit_error.md`.