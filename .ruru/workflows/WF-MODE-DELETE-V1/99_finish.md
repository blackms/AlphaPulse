+++
# --- Step Metadata ---
step_id = "WF-MODE-DELETE-V1-99-FINISH" # (String, Required) Unique ID for this step.
# Convention: The final step in a workflow should always use step_number = 99
step_number = 99
title = "Step 99: Finish Mode Deletion" # (String, Required) Title of this specific step.
description = """
(String, Required) Logs the successful completion of the mode deletion workflow
and reports the final outcome to the user/initiator.
"""

# --- Flow Control ---
depends_on = ["WF-MODE-DELETE-V1-04-RUN-BUILD-SCRIPT"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "" # (String, Required) Always empty for the finish step.
error_step = "" # (String, Optional) Usually empty.

# --- Execution ---
delegate_to = "prime-coordinator" # (String, Optional) Coordinator handles final reporting.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed from previous steps.
    "Output from step WF-MODE-DELETE-V1-00-START: mode_slug",
    "Output from step WF-MODE-DELETE-V1-02-EXECUTE-DELETION: deletion_status",
    "Output from step WF-MODE-DELETE-V1-03-UPDATE-BUILD-COLLECTIONS: build_collections_update_status",
    "Output from step WF-MODE-DELETE-V1-04-RUN-BUILD-SCRIPT: build_script_status", # Should be 'Success'
]
outputs = [ # (Array of Strings, Optional) Final workflow outputs.
    "workflow_result: Summary of successful mode deletion.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/26_workflow_step_finish.md" # (String, Required) Link to this template definition.
+++

# Step 99: Finish Mode Deletion

## Actions

(Coordinator `{{delegate_to}}` performs these actions)

1.  **Verify Previous Steps:** Check that `build_script_status` is "Success". If not, this step should ideally not be reached, but handle as an error if it is (**-> Error Step** defined in previous step).
2.  **Log Completion:** Log the successful deletion of mode `{{mode_slug}}` and the status of associated updates (`{{deletion_status}}`, `{{build_collections_update_status}}`). (Follow Rule `08-logging-procedure-simplified.md`).
3.  **Prepare Final Report:** Construct the `workflow_result` output message confirming success.
    *   Example: "Successfully deleted mode `{{mode_slug}}`. Mode and rules directories removed. Build collections status: `{{build_collections_update_status}}`. Build script executed successfully to update configurations."
4.  **Report Outcome:** Use `<attempt_completion>` to report the `workflow_result` back to the User or initiating process.

## Acceptance Criteria

*   The build script executed successfully in the previous step.
*   Successful completion is logged.
*   A final success message (`workflow_result`) is generated and reported.

## Error Handling

*   If the build script status was not "Success" (should be caught earlier), log an error.
*   Errors during final logging should be noted but may not prevent reporting overall success.