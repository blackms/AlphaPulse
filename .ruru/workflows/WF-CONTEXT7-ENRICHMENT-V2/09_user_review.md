+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-ENRICHMENT-V2-09-USER_REVIEW" # (String, Required) Unique ID for this step
title = "Step 09: User Review" # (String, Required) Title of this specific step.
description = """
Presents a summary of the Context7 KB enrichment changes to the User for final review and feedback. The Coordinator uses `ask_followup_question` to confirm if the enrichment (including generated files, source info, summary rule, and metadata) meets the User's expectations. Allows for potential refinement loops if necessary.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-ENRICHMENT-V2-08-QUALITY_ASSURANCE"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "10_finalization.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails (e.g., user rejects).

# --- Execution ---
delegate_to = "roo-commander" # (String, Optional) Coordinator handles interaction with the User.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-00-START: mode_slug",
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-00-START: context7_base_url",
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-08-QUALITY_ASSURANCE: qa_status", # To confirm QA passed
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-03-STORE_SOURCE_INFO: source_info_path", # For summary
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-04-GENERATE_SUMMARY_RULE: summary_rule_path", # For summary
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-07-GENERATE_KB_README: kb_readme_path", # For summary
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "user_approval_status: Confirmation that the user approved the enrichment (Yes/No/Refinements Needed).",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 09: User Review

## Actions

1.  **Check QA Status:** Verify that `qa_status` from the previous step is "Pass". If not, handle appropriately (potentially proceed to `{{error_step}}` or log warning).
2.  **Summarize Changes:** Prepare a concise summary for the User, mentioning:
    *   The target mode (`[mode_slug]`) and source URL (`[context7_base_url]`).
    *   Confirmation that KB files were generated in `kb/context7/`.
    *   Confirmation that `kb/context7/_index.json` and `kb/context7/source_info.json` (with fetched/default metadata) were created.
    *   Confirmation that a summary rule (`[summary_rule_path]`) was generated.
    *   Confirmation that the KB README (`[kb_readme_path]`) and mode definition were updated.
    *   *(Optional: Provide direct links/paths to key files like `source_info.json` and the summary rule for easy review).*
3.  **Prompt User for Feedback:**
    *   Use `ask_followup_question` to present the summary and ask: "Does this Context7 KB enrichment for mode `[mode_slug]`, including the generated artifacts and metadata, meet your expectations?"
    *   Suggestions:
        *   `<suggest>Yes, looks good.</suggest>`
        *   `<suggest>No, refinements are needed.</suggest>`
        *   `<suggest>Let me review the files first.</suggest>`
4.  **Store Response:** Store the User's response in `[user_approval_status]`.
5.  **Handle Refinements:** If the User requests refinements:
    *   Gather details about the required changes.
    *   *(Workflow Logic Note: This might involve looping back to earlier steps like Script Execution, Metadata Fetching, Rule Generation, etc., followed by QA again. This complex branching isn't fully represented here but should be handled by the orchestrator).*
    *   For now, if refinements are needed, consider this path potentially leading to `{{error_step}}` or a manual intervention state.

## Acceptance Criteria

*   QA status from the previous step was "Pass".
*   A summary of changes was presented to the User.
*   The User provided feedback via `ask_followup_question`.
*   `user_approval_status` reflects the User's response.

## Error Handling

*   If QA did not pass, proceed to `{{error_step}}` or handle as defined by the QA process.
*   If the `ask_followup_question` fails, log the error and proceed to `{{error_step}}`.
*   If the User selects "No, refinements are needed", the orchestrator needs to manage the refinement loop. For this step definition, treat it as a non-success path (potentially leading to `{{error_step}}`).