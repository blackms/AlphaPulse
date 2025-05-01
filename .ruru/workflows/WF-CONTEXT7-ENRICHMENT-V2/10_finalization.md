+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-ENRICHMENT-V2-10-FINALIZATION" # (String, Required) Unique ID for this step.
# Convention: The final step in a workflow should always use step_number = 99, but we use 10 here as it's the 11th step (0-indexed)
step_number = 10
title = "Step 10: Finalization" # (String, Required) Title of this specific step.
description = """
Finalizes the Context7 KB enrichment workflow after successful user review. Logs the successful completion, including the mode and source URL. Reports the final success status to the User or initiating process.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-ENRICHMENT-V2-09-USER_REVIEW"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "" # (String, Required) Always empty for the finish step.
error_step = "" # (String, Optional) Usually empty.
condition = """
# Only run if user approved in the previous step
inputs.user_approval_status == "Yes, looks good."
"""

# --- Execution ---
delegate_to = "roo-commander" # (String, Optional) Coordinator handles final logging and reporting.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed from previous steps.
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-00-START: mode_slug",
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-00-START: context7_base_url",
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-09-USER_REVIEW: user_approval_status",
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-08-QUALITY_ASSURANCE: qa_status",
]
outputs = [ # (Array of Strings, Optional) Final workflow outputs.
    "workflow_result: Summary of success, including mode and source URL.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/26_workflow_step_finish.md" # (String, Required) Link to the finish step template definition.
+++

# Step 10: Finalization

## Actions

1.  **Aggregate Results:** Collect key status outputs (`user_approval_status`, `qa_status`).
2.  **Determine Final Status:** Confirm that `user_approval_status` is "Yes, looks good." and `qa_status` is "Pass".
3.  **Log Completion:** Log the successful completion of the Context7 KB enrichment for `[mode_slug]` using URL `[context7_base_url]`. Mention successful metadata fetching/storage and summary rule generation.
4.  **Report Outcome:** Prepare the final `workflow_result` message confirming success and summarizing the key details (mode, source URL).

## Acceptance Criteria

*   User approval and QA checks passed in previous steps.
*   Successful completion is logged.
*   The final `workflow_result` summary is generated.

## Error Handling

*   If previous steps indicated failure (e.g., User rejection, QA failure), this step should be skipped (due to the condition). The workflow would have proceeded to `EE_handle_error.md` earlier.