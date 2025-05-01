+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-REFRESH-V1-06-QUALITY-ASSURANCE" # (String, Required) Unique ID for this step.
title = "Step 06: Quality Assurance (ACQA)" # (String, Required) Title of this specific step.
description = """
(String, Required) Performs Quality Assurance checks on the refreshed Context7 KB artifacts
according to the ACQA process. Verifies file presence, structure, validity, and updates.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-REFRESH-V1-05-UPDATE-KB-README"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "07_user_review.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "roo-commander" # (String, Optional) Coordinator initiates and oversees QA. Can delegate checks to QA Agent.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: mode_slug",
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: context7_output_dir",
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: context7_base_url",
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: detail_status",
    "Output from step WF-CONTEXT7-REFRESH-V1-02-GENERATE-UPDATE-SUMMARY-RULE: summary_rule_path",
    "Output from step WF-CONTEXT7-REFRESH-V1-03-UPDATE-CREATE-USAGE-STRATEGY: usage_strategy_path",
    "Output from step WF-CONTEXT7-REFRESH-V1-04-UPDATE-MODE-DEFINITION: mode_definition_path",
    "Output from step WF-CONTEXT7-REFRESH-V1-05-UPDATE-KB-README: kb_readme_path",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "qa_status: 'Passed' or 'Failed'.",
    "qa_report: Summary of checks performed and any issues found.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 06: Quality Assurance (ACQA)

## Actions

(Coordinator `{{delegate_to}}` initiates ACQA process - `.ruru/processes/acqa-process.md`. Checks can be delegated to a QA Agent, e.g., `code-reviewer`)

1.  **Initiate ACQA:** Follow the standard ACQA process.
2.  **Perform Checks:**
    *   `- [ ]` **Context7 Files:** Verify presence and basic structure of files within `{{context7_output_dir}}`.
    *   `- [ ]` **Index File:** Check validity and consistency of `{{context7_output_dir}}/_index.json`. (e.g., valid JSON, expected keys).
    *   `- [ ]` **Summary Rule:** Check presence, validity (TOML+MD), and correctness of `{{summary_rule_path}}`. Ensure listed topics match `_index.json` structure.
    *   `- [ ]` **KB README:** Check presence and validity of `{{kb_readme_path}}`. Verify it correctly references `context7/_index.json`, the source URL `{{context7_base_url}}`, and the detail status `{{detail_status}}`.
    *   `- [ ]` **Usage Strategy:** Check presence of `{{usage_strategy_path}}`.
    *   `- [ ]` **Mode Definition:** Check `{{mode_definition_path}}` to ensure `kb/context7/_index.json` is present in `related_context`.
3.  **Handle Issues:** If any checks fail:
    *   Log the specific issues found.
    *   Initiate correction loop (delegate back to relevant previous step/agent) or follow AFR process (`.ruru/processes/afr-process.md`). Set `qa_status` = "Failed". **-> Error Step** (or loop).
4.  **Log & Report:**
    *   If all checks pass, set `qa_status` = "Passed".
    *   Log the QA outcome.
    *   Prepare `qa_report` summarizing findings.
    *   Report completion (`qa_status`, `qa_report`) back.

## Acceptance Criteria

*   All specified checks are performed according to the ACQA process.
*   All checked artifacts meet the defined standards and requirements.
*   `qa_status` is set to "Passed".

## Error Handling

*   If any QA check fails, `qa_status` is set to "Failed", issues are logged, and the workflow proceeds to `{{error_step}}` or initiates a correction loop/AFR process.