+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-ENRICHMENT-V2-06-UPDATE_MODE_DEFINITION" # (String, Required) Unique ID for this step
title = "Step 06: Update Mode Definition Context" # (String, Required) Title of this specific step.
description = """
Delegates to the Mode Structure Agent (e.g., mode-maintainer) to update the target mode's definition file (`[mode_slug].mode.md`). The agent reads the mode file, parses the TOML frontmatter, and adds the path to the Context7 index (`kb/context7/_index.json`) to the `related_context` array if it's not already present. Uses precise modification tools.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-ENRICHMENT-V2-05-UPDATE_KB_USAGE_STRATEGY"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "07_generate_kb_readme.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "mode-maintainer" # (String, Optional) Mode responsible for editing the mode definition file.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-00-START: mode_slug",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "mode_definition_update_status: Confirmation that the mode definition was updated (or context already present).",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 06: Update Mode Definition Context

## Actions

1.  **Delegate Mode Definition Update:**
    *   Instruct the `mode-maintainer` delegate to update the mode definition file.
    *   Provide the `[mode_slug]`.
    *   Instruct the agent to:
        *   Define the mode file path: `[mode_file_path] = .ruru/modes/[mode_slug]/[mode_slug].mode.md`.
        *   Define the context path to add: `[context7_index_context_path] = kb/context7/_index.json`.
        *   Read `[mode_file_path]` (Prefer MCP `read_file_content`, fallback `read_file`). Handle read errors -> **Stop**.
        *   Parse the TOML frontmatter. Handle parse errors -> **Stop**.
        *   Check if `[context7_index_context_path]` is already present in the `related_context` array within the TOML.
        *   If **not** present:
            *   Add `[context7_index_context_path]` to the `related_context` array.
            *   Re-serialize the TOML block (or prepare the diff).
            *   Use a precise modification tool (Prefer MCP `edit_file_content`, fallback `apply_diff`) to update only the `related_context` line(s) within the TOML block of `[mode_file_path]`. Handle update errors -> **Stop**.
            *   Log that the context was added.
        *   If **present**:
            *   Log that the context path already exists in `related_context`. No update needed.
2.  **Receive Confirmation:** Await confirmation of success, skip, or failure from the delegate.

## Acceptance Criteria

*   Delegation to `mode-maintainer` was successful.
*   The mode definition file `.ruru/modes/[mode_slug]/[mode_slug].mode.md` has the path `kb/context7/_index.json` included in its `related_context` array in the TOML frontmatter.
*   `mode_definition_update_status` indicates success or already present status.

## Error Handling

*   If delegation fails or the `mode-maintainer` reports critical errors (e.g., cannot read/parse/update mode file), log the error and proceed to `{{error_step}}`.