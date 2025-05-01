+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-REFRESH-V1-04-UPDATE-MODE-DEFINITION" # (String, Required) Unique ID for this step.
title = "Step 04: Update Mode Definition Context" # (String, Required) Title of this specific step.
description = """
(String, Required) Ensures the target mode's definition file (`.mode.md`)
references the newly generated Context7 index file (`kb/context7/_index.json`) in its `related_context` array.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-REFRESH-V1-03-UPDATE-CREATE-USAGE-STRATEGY"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "05_update_kb_readme.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "mode-maintainer" # (String, Optional) Mode responsible for executing the core logic of this step. Alt: technical-writer

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: mode_slug",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "mode_definition_path: Path to the mode definition file.",
    "mode_definition_update_status: 'Updated' or 'Already Present'.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 04: Update Mode Definition Context

## Actions

(Instructions for the delegate: `{{delegate_to}}`)

1.  **Define Paths:**
    *   `[mode_slug]` = Input `mode_slug`.
    *   `[mode_file_path]` = `.ruru/modes/[mode_slug]/[mode_slug].mode.md`.
    *   `[context7_index_context_path]` = `kb/context7/_index.json`.
2.  **Read Mode File:**
    *   `- [ ]` Read the content of `[mode_file_path]`. (Prefer MCP `read_file_content`, fallback `read_file`). Handle file not found error (**-> Error Step**).
3.  **Check and Update TOML:**
    *   `- [ ]` Parse the TOML frontmatter block (`+++ ... +++`).
    *   `- [ ]` Check if the string `"[context7_index_context_path]"` already exists within the `related_context` array in the parsed TOML.
    *   `- [ ]` If it does NOT exist:
        *   Add `"[context7_index_context_path]"` to the `related_context` array.
        *   Use `apply_diff` (or MCP `edit_file_content`) to modify the `related_context` line(s) in the original file content, adding the new path. Ensure valid TOML array syntax is maintained. Handle errors (**-> Error Step**).
        *   Set `[mode_definition_update_status]` = "Updated".
    *   `- [ ]` If it already exists:
        *   Log that the context path is already present.
        *   Set `[mode_definition_update_status]` = "Already Present".
4.  **Log & Report:**
    *   `- [ ]` Log any errors encountered during reading, parsing, or writing.
    *   `- [ ]` Report completion (providing `mode_definition_path` and `mode_definition_update_status`) or failure back to the Coordinator.

## Acceptance Criteria

*   The mode definition file `[mode_file_path]` has been read successfully.
*   The `related_context` array within the TOML block of `[mode_file_path]` includes the path `kb/context7/_index.json`.
*   The update status ('Updated' or 'Already Present') is reported correctly.

## Error Handling

*   Failure to read or parse the mode definition file proceeds to `{{error_step}}`.
*   Failure to apply the diff/edit to update the `related_context` array proceeds to `{{error_step}}`.