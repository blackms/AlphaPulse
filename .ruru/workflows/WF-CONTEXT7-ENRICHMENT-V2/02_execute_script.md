+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-ENRICHMENT-V2-02-EXECUTE_SCRIPT" # (String, Required) Unique ID for this step
title = "Step 02: Execute Processing Script" # (String, Required) Title of this specific step.
description = """
Executes the `.ruru/scripts/process_llms_json.js` Node.js script. Passes the Context7 base URL, the target output directory (`kb/context7`), and the fetched/default token count as command-line arguments. Checks for successful execution.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-ENRICHMENT-V2-01-FETCH_METADATA"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "03_store_source_info.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "roo-commander" # (String, Optional) Coordinator handles command execution.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-00-START: mode_slug",
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-00-START: context7_base_url",
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-01-FETCH_METADATA: fetched_tokens", # Fetched or default token count
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "context7_output_dir: Path to the output directory used by the script.",
    "script_execution_status: Confirmation of script success.",
    "script_output_log: Captured output/errors from the script execution.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 02: Execute Processing Script

## Actions

1.  **Define Output Directory:** Set `[context7_output_dir] = .ruru/modes/[mode_slug]/kb/context7`.
2.  **Construct Command:**
    *   Build the base command: `cmd = "node .ruru/scripts/process_llms_json.js --baseUrl \"[context7_base_url]\" --outputDir \"[context7_output_dir]\""`
    *   Append the token argument: `cmd += " --tokens \"[fetched_tokens]\""` (using the value determined in the previous step, which is either fetched/manual or the default 10,000,000).
3.  **Explain Command:** Clearly explain the command being executed, including the source of the `--tokens` value (fetched or default).
4.  **Execute Command:**
    *   Use the `execute_command` tool to run the constructed `cmd`.
    *   Capture the exit code and output/errors into `[script_output_log]`.
5.  **Check Result:**
    *   If the exit code is non-zero, log the error, set `[script_execution_status]` to "Failed", and proceed to `{{error_step}}`.
    *   If the exit code is zero, log success, set `[script_execution_status]` to "Success".

## Acceptance Criteria

*   The `.ruru/scripts/process_llms_json.js` script was executed with the correct arguments (`--baseUrl`, `--outputDir`, `--tokens`).
*   The command execution completed successfully (exit code 0).
*   The script generated files within the `[context7_output_dir]`, including `_index.json`.
*   `script_execution_status` is "Success".

## Error Handling

*   If the `execute_command` tool fails or the script returns a non-zero exit code, log the error details from `[script_output_log]`, inform the user, and proceed to `{{error_step}}`.