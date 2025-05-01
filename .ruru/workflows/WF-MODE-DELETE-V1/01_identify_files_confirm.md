+++
# --- Step Metadata ---
step_id = "WF-MODE-DELETE-V1-01-IDENTIFY-FILES-CONFIRM" # (String, Required) Unique ID for this step.
title = "Step 01: Identify Associated Files & Confirm Deletion" # (String, Required) Title of this specific step.
description = """
(String, Required) Identifies the mode directory, associated rules directory (if it exists),
and other files potentially needing updates. Presents this list to the user for final deletion confirmation.
"""

# --- Flow Control ---
depends_on = ["WF-MODE-DELETE-V1-00-START"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "02_execute_deletion.md" # (String, Optional) Filename of the next step if user confirms.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if user cancels or error occurs.

# --- Execution ---
delegate_to = "prime-coordinator" # (String, Optional) Coordinator handles identification and user interaction.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed.
    "Output from step WF-MODE-DELETE-V1-00-START: mode_slug",
    "Output from step WF-MODE-DELETE-V1-00-START: mode_dir_path",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "mode_dir_to_delete: Path to the mode directory.",
    "rules_dir_to_delete: Path to the rules directory (or empty string if none).",
    "files_to_check: List of files potentially needing manual updates (e.g., build_collections.json).",
    "user_confirmation: Boolean indicating user confirmed deletion.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 01: Identify Associated Files & Confirm Deletion

## Actions

(Instructions for the delegate: `{{delegate_to}}`)

1.  **Identify Files/Dirs:**
    *   `- [ ]` Set `[mode_dir_to_delete]` = Input `mode_dir_path`.
    *   `- [ ]` Define `[rules_dir_path]` = `.roo/rules-{{mode_slug}}/`.
    *   `- [ ]` Check if `[rules_dir_path]` exists. (Prefer MCP `get_filesystem_info`, fallback `list_files` or `execute_command ls`).
    *   `- [ ]` If `[rules_dir_path]` exists, set `[rules_dir_to_delete]` = `[rules_dir_path]`. Otherwise, set `[rules_dir_to_delete]` = "".
    *   `- [ ]` Define `[files_to_check]` = [".ruru/config/build_collections.json"]. (Note: `.roomodes` and mode summary are handled by build script later).
2.  **Present Information:**
    *   `- [ ]` Construct a message summarizing the findings:
        *   Mode directory to be deleted: `[mode_dir_to_delete]`
        *   Rules directory to be deleted: `[rules_dir_to_delete]` (or "None found")
        *   Files to check/potentially update manually: `[files_to_check]`
        *   Files updated automatically by build script: `.roomodes`, `.ruru/modes/roo-commander/kb/kb-available-modes-summary.md`
3.  **Request Confirmation:**
    *   `- [ ]` Use `<ask_followup_question>` to present the summary and ask for explicit confirmation.
    *   Question: "Please review the items listed above that will be deleted or potentially need checking. Proceed with deleting mode `{{mode_slug}}`?"
    *   Suggestions: "Yes, proceed with deletion.", "No, cancel deletion."
4.  **Handle Confirmation:**
    *   `- [ ]` If user confirms ("Yes, proceed..."): Set `[user_confirmation]` = true. Log confirmation. Proceed to `{{next_step}}`.
    *   `- [ ]` If user cancels ("No, cancel..."): Set `[user_confirmation]` = false. Log cancellation. Proceed to `{{error_step}}` (or a specific cancellation step if defined).
    *   Handle errors/timeouts (**-> Error Step**).
5.  **Prepare Outputs:**
    *   `- [ ]` Prepare outputs: `mode_dir_to_delete`, `rules_dir_to_delete`, `files_to_check`, `user_confirmation`.

## Acceptance Criteria

*   Associated directories (`.ruru/modes/<mode-slug>/`, `.roo/rules-<mode-slug>/` if exists) are identified.
*   Files potentially needing manual checks are identified.
*   The list of items is presented to the user.
*   Explicit user confirmation (true/false) is obtained and stored.

## Error Handling

*   If checking for the rules directory fails unexpectedly, log the error but potentially continue to confirmation, noting the uncertainty.
*   If the user cancels the deletion, set `user_confirmation` to false and proceed to `{{error_step}}`.