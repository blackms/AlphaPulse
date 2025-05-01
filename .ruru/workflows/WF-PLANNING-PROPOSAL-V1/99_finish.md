+++
# --- Step Metadata ---
step_id = "WF-PLANNING-PROPOSAL-V1-99-FINISH" # (String, Required) Unique ID for this step.
# Convention: The final step in a workflow should always use step_number = 99
step_number = 99
title = "Step 99: Finish Workflow & Verify Artifacts" # (String, Required) Title of this specific step.
description = """
Verify all expected artifacts (input directory, refinement notes, whitepaper, implementation documents) exist.
Log the completion status and report the final outcome, including the path to the proposal directory.
"""

# --- Flow Control ---
depends_on = ["WF-PLANNING-PROPOSAL-V1-03-GENERATE-IMPL-DOCS"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "" # (String, Required) Always empty for the finish step.
error_step = "" # (String, Optional) Errors in finish usually mean reporting failure directly.

# --- Execution ---
delegate_to = "" # (String, Optional) Handled by the workflow orchestrator.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed from previous steps.
    "Output from step WF-PLANNING-PROPOSAL-V1-00-START: proposal_base_path",
    "Output from step WF-PLANNING-PROPOSAL-V1-00-START: input_path",
    "Output from step WF-PLANNING-PROPOSAL-V1-01-REFINE: refinement_notes_path",
    "Output from step WF-PLANNING-PROPOSAL-V1-02-GENERATE-WHITEPAPER: whitepaper_path",
    "Output from step WF-PLANNING-PROPOSAL-V1-03-GENERATE-IMPL-DOCS: implementation_docs_paths", # List of paths
]
outputs = [ # (Array of Strings, Optional) Final workflow outputs.
    "workflow_result: Summary of success/failure and path to the proposal directory (`proposal_base_path`).",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/26_workflow_step_finish.md" # (String, Required) Link to this template definition.
+++

# Step 99: Finish Workflow & Verify Artifacts

## Actions

1.  **Verify Artifacts:**
    *   Check for the existence of the `{{inputs[1]}}` directory (input_path).
    *   Check for the existence of the file `{{inputs[1]}}/initial_request.md` (if text input was provided in step 00).
    *   Check for the existence of the `{{inputs[2]}}` file (refinement_notes_path).
    *   Check for the existence of the `{{inputs[3]}}` file (whitepaper_path).
    *   Check for the existence of all files listed in `{{inputs[4]}}` (implementation_docs_paths).
    *   *(Implementation Note: This might involve using `list_files` or similar tools based on the orchestrator's capabilities).*
2.  **Log Completion:**
    *   If all artifacts exist, log the successful completion of the workflow, referencing `{{inputs[0]}}` (proposal_base_path).
    *   If any artifact is missing, or if an error was explicitly passed from a previous step (e.g., via `error_details` input), log the failure. Specify missing artifacts or the received error details.
3.  **Prepare Final Report:**
    *   If successful, prepare a success message for the user, indicating completion and providing the `proposal_base_path`.
    *   If failed, prepare a failure message detailing the reason (e.g., missing artifacts, specific error from previous step like "Whitepaper generation failed").
4.  **Determine Final Status:** Set status to Success or Failure based on verification.
5.  **Report Outcome:** Provide the final status and message as `workflow_result`.

## Acceptance Criteria

*   Verification check for all expected artifacts has been performed.
*   Workflow completion status (Success/Failure) is determined and logged.
*   A final `workflow_result` message summarizing the outcome and providing the `proposal_base_path` (on success) is prepared.

## Error Handling

*   If verification fails (artifacts missing), the workflow status is set to Failure, and the `workflow_result` indicates the reason.