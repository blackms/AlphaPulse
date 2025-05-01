+++
# --- Basic Metadata (Auto-Generated) ---
id = "WF-AUDIT-IMPROVE-V1_06_simulate_flow" # Unique ID for this step
step_number = 7 # Step sequence number
# --- Workflow Step Configuration ---
title = "Simulate Workflow Flow (Static Check)"
description = "Performs static checks on the workflow definition: validates step connections (next_step, error_step), delegation targets, and basic input/output consistency."
step_type = "standard" # start, standard, finish, error
# --- Execution Details ---
# mode = "self" # This analysis can likely be done by the workflow executor itself.
# delegate_to = "" # Optional: Delegate execution to another mode
# --- Input/Output ---
inputs = ["target_workflow_dir", "workflow_files_content", "available_modes"] # List of input variables required by this step (needs latest workflow content)
outputs = ["simulation_results"] # List of output variables produced by this step (report on findings)
# --- Control Flow ---
next_step = "99_finish.md" # The next step file to execute
error_step = "EE_audit_error.md" # Optional: Step file to execute on error
# --- Logging & Auditing ---
# log_level = "info" # Default logging level for this step
# --- Tags & Categorization ---
tags = ["validation", "simulation", "static-analysis", "flow-check", "consistency"]
# --- Notes ---
# - This is NOT a runtime simulation, just a static check of the definition files.
# - Input/output check is heuristic based on variable names.
# - Relies on `workflow_files_content` reflecting the latest state. The workflow loop (`05` -> `01`) ensures this by re-gathering context before this step is reached after modifications.
+++

# Step 06: Simulate Workflow Flow (Static Check)

## Description

This step performs a series of static checks on the (potentially modified) workflow definition files to identify potential inconsistencies or broken links in the flow logic *before* attempting execution. It verifies step connections, delegation targets, and performs a basic heuristic check on input/output consistency between steps.

## Actions

1.  **Use Latest Content:** Process the `workflow_files_content` input, which reflects the latest state due to the workflow looping back through `01_gather_context` after any changes applied in `05_apply_changes`.
2.  **Parse Workflow Files:** Process the verified `workflow_files_content` to extract metadata from each step file (`README.md`, `NN_*.md`, `EE_*.md`).
3.  **Check Step Connections:**
    *   For each step file, verify that the files specified in `next_step`, `error_step`, and `conditional_next_steps[*].next_step` exist within the `target_workflow_dir`. Record any missing target files.
4.  **Check Delegation Targets:**
    *   For each step file with a `delegate_to` field, verify that the specified mode slug exists in the `available_modes` list. Record any invalid delegation targets.
5.  **Check Input/Output Consistency (Heuristic):**
    *   For each step, identify its declared `outputs`.
    *   Identify the potential next step(s) based on `next_step` and `conditional_next_steps`.
    *   For each potential next step, identify its declared `inputs`.
    *   Perform a basic check: Do the outputs of the current step seem to cover the required inputs of the next step(s), based on matching variable names? Record potential mismatches (this is a heuristic and may have false positives/negatives).
6.  **Compile Results:** Aggregate all findings (missing step files, invalid delegation targets, potential input/output mismatches) into a structured report (Markdown or JSON). Store this report in the `simulation_results` output variable.
7.  **Log Simulation Results:** Record that the static check is complete and summarize the findings briefly.

## Inputs

*   `target_workflow_dir` (from step 00)
*   `workflow_files_content` (from step 01, reflecting the latest state after any iterations)
*   `available_modes` (List of available mode slugs)

## Outputs

*   `simulation_results`: Structured data or Markdown report detailing the findings of the static flow check.

## Next Step

*   Proceed to `99_finish.md`.

## Error Handling

*   If parsing workflow files fails, transition to `EE_audit_error.md`.