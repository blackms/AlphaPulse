+++
# --- Step Metadata ---
step_id = "WF-MODE-DELETE-V1-02-EXECUTE-DELETION" # (String, Required) Unique ID for this step.
title = "Step 02: Execute Deletion" # (String, Required) Title of this specific step.
description = """
(String, Required) Executes the deletion of the mode directory and the associated rules directory (if applicable),
based on user confirmation from the previous step.
"""

# --- Flow Control ---
depends_on = ["WF-MODE-DELETE-V1-01-IDENTIFY-FILES-CONFIRM"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "03_update_build_collections.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "prime-coordinator" # (String, Optional) Coordinator executes the deletion commands.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed.
    "Output from step WF-MODE-DELETE-V1-01-IDENTIFY-FILES-CONFIRM: mode_dir_to_delete",
    "Output from step WF-MODE-DELETE-V1-01-IDENTIFY-FILES-CONFIRM: rules_dir_to_delete",
    "Output from step WF-MODE-DELETE-V1-01-IDENTIFY-FILES-CONFIRM: user_confirmation",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "deletion_status: Summary of deletion operations performed.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 02: Execute Deletion

## Actions

(Instructions for the delegate: `{{delegate_to}}`)

1.  **Check Confirmation:**
    *   `- [ ]` Verify that `user_confirmation` input is true. If false, **-> Error Step** (User cancelled).
2.  **Delete Mode Directory:**
    *   `- [ ]` Construct the command: `rm -rf "{{mode_dir_to_delete}}"`
    *   `- [ ]` Use `<execute_command>` to run the command. Ensure OS-aware syntax (Rule `05`). Explain the command's purpose (deleting mode directory).
    *   `- [ ]` Check the exit code. If non-zero, log error and **-> Error Step**.
    *   `- [ ]` Log successful deletion of the mode directory.
3.  **Delete Rules Directory (If Applicable):**
    *   `- [ ]` Check if `rules_dir_to_delete` input is not empty.
    *   `- [ ]` If it's not empty:
        *   Construct the command: `rm -rf "{{rules_dir_to_delete}}"`
        *   Use `<execute_command>` to run the command. Ensure OS-aware syntax (Rule `05`). Explain the command's purpose (deleting rules directory).
        *   Check the exit code. If non-zero, log error and **-> Error Step**.
        *   Log successful deletion of the rules directory.
4.  **Prepare Output:**
    *   `- [ ]` Prepare `deletion_status` output summarizing which directories were deleted.

## Acceptance Criteria

*   User confirmation was true.
*   The command to delete `mode_dir_to_delete` executed successfully (exit code 0).
*   If `rules_dir_to_delete` was provided, the command to delete it executed successfully (exit code 0).
*   Deletion operations are logged.

## Error Handling

*   If `user_confirmation` is false, proceed to `{{error_step}}`.
*   If any `rm -rf` command fails (non-zero exit code), log the error and output, and proceed to `{{error_step}}`.