+++
# --- Step Metadata ---
step_id = "WF-MODE-DELETE-V1-04-RUN-BUILD-SCRIPT" # (String, Required) Unique ID for this step.
title = "Step 04: Run Build Script" # (String, Required) Title of this specific step.
description = """
(String, Required) Executes the `build_roomodes.js` script to update the `.roomodes` file
and the mode summary KB file, reflecting the deletion of the mode.
"""

# --- Flow Control ---
depends_on = ["WF-MODE-DELETE-V1-03-UPDATE-BUILD-COLLECTIONS"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "99_finish.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "prime-coordinator" # (String, Optional) Coordinator executes the build script.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed.
    # No specific inputs required from previous steps, just confirmation they succeeded.
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "build_script_status: 'Success' or 'Failure'.",
    "updated_roomodes_file: Path to the updated .roomodes file.",
    "updated_mode_summary_kb: Path to the updated mode summary KB file.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 04: Run Build Script

## Actions

(Instructions for the delegate: `{{delegate_to}}`)

1.  **Construct Command:**
    *   `- [ ]` Prepare the command: `node .ruru/scripts/build_roomodes.js`
2.  **Execute Command:**
    *   `- [ ]` Use `<execute_command>` to run the command. Ensure OS-aware syntax (Rule `05`). Explain the command's purpose (updating `.roomodes` and mode summary KB).
    *   `- [ ]` Check the exit code. If non-zero, log error, set `[build_script_status]` = "Failure", and **-> Error Step**.
    *   `- [ ]` If exit code is 0, log successful execution. Set `[build_script_status]` = "Success".
3.  **Prepare Outputs:**
    *   `- [ ]` Prepare outputs: `build_script_status`, `updated_roomodes_file`=".roomodes", `updated_mode_summary_kb`=".ruru/modes/roo-commander/kb/kb-available-modes-summary.md".

## Acceptance Criteria

*   The `build_roomodes.js` script executes successfully (exit code 0).
*   The `.roomodes` file is updated.
*   The `.ruru/modes/roo-commander/kb/kb-available-modes-summary.md` file is updated.
*   `build_script_status` is set to "Success".

## Error Handling

*   If the build script fails (non-zero exit code), log the error and output, set `build_script_status` to "Failure", and proceed to `{{error_step}}`.