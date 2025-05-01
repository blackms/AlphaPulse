+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-REFRESH-V1-05-UPDATE-KB-README" # (String, Required) Unique ID for this step.
title = "Step 05: Update KB README" # (String, Required) Title of this specific step.
description = """
(String, Required) Updates the target mode's KB README file (`kb/README.md`)
to include references to the `context7` directory, its `_index.json`, the source URL,
and the status of source detail fetching (obtained/assumed).
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-REFRESH-V1-04-UPDATE-MODE-DEFINITION"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "06_quality_assurance.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "mode-maintainer" # (String, Optional) Mode responsible for executing the core logic of this step. Alt: technical-writer

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: mode_slug",
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: context7_base_url",
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: detail_status",
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: source_token_count", # (Optional, depends on detail_status)
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: source_update_date", # (Optional, depends on detail_status)
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "kb_readme_path: Path to the KB README file.",
    "kb_readme_update_status: 'Updated' or 'Error'.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 05: Update KB README

## Actions

(Instructions for the delegate: `{{delegate_to}}`)

1.  **Define Path:**
    *   `[mode_slug]` = Input `mode_slug`.
    *   `[kb_readme_path]` = `.ruru/modes/[mode_slug]/kb/README.md`.
2.  **Read KB README:**
    *   `- [ ]` Read the content of `[kb_readme_path]`. (Prefer MCP `read_file_content`, fallback `read_file`). Handle file not found error (**-> Error Step**).
3.  **Prepare Update:**
    *   `- [ ]` Identify the section in the README that lists KB subdirectories (e.g., under a "KB Files" or "Directory Structure" heading).
    *   `- [ ]` Ensure there is an entry for the `context7` directory.
    *   `- [ ]` Update or add the description for the `context7` entry to include:
        *   A link to `context7/_index.json`.
        *   The confirmed source URL: `{{context7_base_url}}`.
        *   The detail status: `(Details {{detail_status}})`.
        *   (Optional) If `detail_status` is "obtained", include relevant details like token count or update date if available in inputs.
        *   Example text: `*   **context7/**: Context derived from external documentation ([_index.json](context7/_index.json)). Source: {{context7_base_url}} (Details {{detail_status}} [Token Count: {{source_token_count}}]).`
4.  **Apply Update:**
    *   `- [ ]` Use file editing tools (Prefer MCP `edit_file_content`, fallback `apply_diff` or `search_and_replace`) to modify the README content with the updated `context7` description. Be precise to avoid corrupting the file. Handle errors (**-> Error Step**).
5.  **Log & Report:**
    *   `- [ ]` Log any errors encountered during reading or writing.
    *   `- [ ]` Report completion (providing `kb_readme_path` and `kb_readme_update_status` = "Updated") or failure (`kb_readme_update_status` = "Error") back to the Coordinator.

## Acceptance Criteria

*   The KB README file `[kb_readme_path]` has been read successfully.
*   The README content includes an entry for the `context7` directory.
*   The `context7` entry correctly references `context7/_index.json`, the source URL (`{{context7_base_url}}`), and the detail status (`{{detail_status}}`), potentially including fetched details.
*   The changes have been successfully saved to the file.

## Error Handling

*   Failure to read the KB README proceeds to `{{error_step}}`.
*   Failure to apply the changes to the KB README proceeds to `{{error_step}}`.