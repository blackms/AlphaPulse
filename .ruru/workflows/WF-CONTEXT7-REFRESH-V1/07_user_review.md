+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-REFRESH-V1-07-USER-REVIEW" # (String, Required) Unique ID for this step.
title = "Step 07: User Review" # (String, Required) Title of this specific step.
description = """
(String, Required) Presents the summary of the Context7 KB refresh to the User for final review and feedback.
Handles potential refinements by looping back to relevant steps.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-REFRESH-V1-06-QUALITY-ASSURANCE"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "99_finish.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails (e.g., user rejects).

# --- Execution ---
delegate_to = "roo-commander" # (String, Optional) Coordinator handles interaction with the User.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: mode_slug",
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: context7_output_dir",
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: detail_status",
    "Output from step WF-CONTEXT7-REFRESH-V1-02-GENERATE-UPDATE-SUMMARY-RULE: summary_rule_path",
    "Output from step WF-CONTEXT7-REFRESH-V1-05-UPDATE-KB-README: kb_readme_path",
    "Output from step WF-CONTEXT7-REFRESH-V1-06-QUALITY-ASSURANCE: qa_status", # Should be 'Passed'
    "Output from step WF-CONTEXT7-REFRESH-V1-06-QUALITY-ASSURANCE: qa_report",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "user_feedback: User's confirmation or rejection/refinement requests.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 07: User Review

## Actions

(Coordinator `{{delegate_to}}` performs these actions)

1.  **Check QA Status:** Verify that `qa_status` from the previous step is "Passed". If not, **-> Error Step**.
2.  **Summarize Changes:** Present a summary of the completed refresh to the User, highlighting:
    *   The target mode: `{{mode_slug}}`.
    *   The refreshed directory: `{{context7_output_dir}}`.
    *   The generated `_index.json` within that directory.
    *   The updated summary rule: `{{summary_rule_path}}`.
    *   The updated KB README: `{{kb_readme_path}}`, mentioning the source detail status (`{{detail_status}}`).
    *   Confirmation that QA passed (`{{qa_report}}`).
3.  **Request Feedback:** Ask the User for feedback using `ask_followup_question`:
    *   Question: "The Context7 KB refresh for mode `{{mode_slug}}` has passed QA. Please review the summary. Does this meet your expectations?"
    *   Suggestions: "Yes, proceed to finish.", "No, refinements needed."
4.  **Handle Feedback:**
    *   If "Yes, proceed to finish.": Log user approval. Proceed to `{{next_step}}`.
    *   If "No, refinements needed.":
        *   Ask User for specific refinement details via `ask_followup_question`.
        *   Log the requested refinements.
        *   Determine which previous step(s) need to be revisited based on feedback.
        *   Delegate the refinement task(s) back to the appropriate agent(s)/step(s). This might involve looping back in the workflow execution logic (handled by the orchestrator). For this step definition, consider it an error/loop condition. **-> Error Step** (or trigger loop).
    *   Handle cancellation/errors. **-> Error Step**.

## Acceptance Criteria

*   QA status from the previous step is "Passed".
*   A summary of the refresh is presented to the User.
*   User feedback is obtained via `ask_followup_question`.
*   User feedback indicates approval ("Yes, proceed to finish.").

## Error Handling

*   If QA status is not "Passed", proceed to `{{error_step}}`.
*   If the User requests refinements or rejects the changes, log the feedback and proceed to `{{error_step}}` (or initiate a refinement loop).
*   If interaction with the User fails, proceed to `{{error_step}}`.