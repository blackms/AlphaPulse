+++
# --- Step Metadata ---
step_id = "WF-MODE-DELETE-V1-00-START" # (String, Required) Unique ID for this step.
title = "Step 00: Identify Mode for Deletion" # (String, Required) Title of this specific step.
description = """
(String, Required) Prompts the user for the exact slug of the mode to be deleted
and verifies the existence of the corresponding mode directory.
"""

# --- Flow Control ---
depends_on = [] # (Array of Strings, Required) Always empty for the start step.
next_step = "01_identify_files_confirm.md" # (String, Required) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "prime-coordinator" # (String, Optional) Coordinator handles this initial user interaction.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Usually references overall workflow inputs.
    "User request: Triggered from 'Delete Modes' option.",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "mode_slug: The exact, verified slug of the mode to be deleted.",
    "mode_dir_path: Path to the mode directory (e.g., '.ruru/modes/<mode-slug>/').",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/24_workflow_step_start.md" # (String, Required) Link to this template definition.
+++

# Step 00: Identify Mode for Deletion

## Actions

(Instructions for the delegate: `{{delegate_to}}`)

1.  **Prompt for Slug:**
    *   `- [ ]` Use `<ask_followup_question>` to ask the user for the exact `slug` of the mode they wish to delete. Provide context about where to find slugs if necessary.
    *   Store the user's input as `[mode_slug]`. Handle cancellation (**-> Error Step**).
2.  **Verify Directory:**
    *   `- [ ]` Define `[mode_dir_path]` = `.ruru/modes/[mode_slug]/`.
    *   `- [ ]` Check if the directory `[mode_dir_path]` exists. (Prefer MCP `get_filesystem_info`, fallback `list_files` or `execute_command ls`).
    *   `- [ ]` If the directory does NOT exist: Inform the user the slug is invalid or the mode doesn't exist. Ask if they want to try a different slug or cancel. Handle response (Loop back to Action 1 or **-> Error Step**).
3.  **Log & Prepare Output:**
    *   `- [ ]` If directory exists, log the verified `[mode_slug]` and `[mode_dir_path]`.
    *   `- [ ]` Prepare outputs `mode_slug` and `mode_dir_path`.

## Acceptance Criteria

*   User provides a `mode_slug`.
*   The corresponding mode directory `.ruru/modes/<mode-slug>/` exists.
*   The verified `mode_slug` and `mode_dir_path` are available as outputs.

## Error Handling

*   If the user cancels the initial prompt, proceed to `{{error_step}}`.
*   If the provided `mode_slug` does not correspond to an existing directory and the user chooses to cancel, proceed to `{{error_step}}`.