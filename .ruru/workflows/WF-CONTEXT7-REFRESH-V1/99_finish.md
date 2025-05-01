+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-REFRESH-V1-99-FINISH" # (String, Required) Unique ID for this step.
# Convention: The final step in a workflow should always use step_number = 99
step_number = 99
title = "Step 99: Finish Workflow" # (String, Required) Title of this specific step.
description = """
(String, Required) Logs the successful completion of the Context7 KB refresh,
including the target mode, source URL, and detail status. Reports success back to the initiator.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-REFRESH-V1-07-USER-REVIEW"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "" # (String, Required) Always empty for the finish step.
error_step = "" # (String, Optional) Usually empty.

# --- Execution ---
delegate_to = "roo-commander" # (String, Optional) Coordinator handles final logging and reporting.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed from previous steps.
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: mode_slug",
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: context7_base_url",
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: detail_status",
    "Output from step WF-CONTEXT7-REFRESH-V1-07-USER-REVIEW: user_feedback", # Should indicate approval
]
outputs = [ # (Array of Strings, Optional) Final workflow outputs.
    "workflow_result: Summary of successful refresh for the specified mode.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/26_workflow_step_finish.md" # (String, Required) Link to this template definition.
+++

# Step 99: Finish Workflow

## Actions

(Coordinator `{{delegate_to}}` performs these actions)

1.  **Verify User Approval:** Check that `user_feedback` from the previous step indicates approval. If not, this step should ideally not be reached, but handle as an error if it is.
2.  **Log Completion:** Log the successful completion of the Context7 refresh for mode `{{mode_slug}}` using URL `{{context7_base_url}}`. Include the detail status (`{{detail_status}}`) in the log message. (Follow Rule `08-logging-procedure-simplified.md`).
3.  **Prepare Final Report:** Construct the `workflow_result` output message confirming success.
    *   Example: "Successfully refreshed Context7 KB for mode `{{mode_slug}}` from source `{{context7_base_url}}`. Source details status: `{{detail_status}}`."
4.  **Report Outcome:** Use `<attempt_completion>` to report the `workflow_result` back to the User or initiating process.

## Acceptance Criteria

*   User approval was confirmed in the previous step.
*   Successful completion is logged, including mode, URL, and detail status.
*   A final success message (`workflow_result`) is generated and reported.

## Error Handling

*   If user approval is missing or negative (should be caught earlier), treat as an unexpected state and log an error.
*   Errors during logging should be noted but may not prevent reporting success if the core task finished.