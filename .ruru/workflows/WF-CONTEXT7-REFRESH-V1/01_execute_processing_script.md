+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-REFRESH-V1-01-EXECUTE-SCRIPT" # (String, Required) Unique ID for this step.
title = "Step 01: Execute Processing Script" # (String, Required) Title of this specific step.
description = """
(String, Required) Executes the Node.js script (`.ruru/scripts/process_llms_json.js`)
to process the Context7 source `llms.json` and generate the KB files in the target mode's directory.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-REFRESH-V1-00-START"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "02_generate_update_summary_rule.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "roo-commander" # (String, Optional) Coordinator executes the script.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: context7_base_url",
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: context7_output_dir",
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: detail_status", # To inform logging/explanation
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "script_execution_status: Success or Failure.",
    "Generated KB files within context7_output_dir.",
    "Generated index file: context7_output_dir/_index.json",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 01: Execute Processing Script

## Actions

1.  **Construct Command:** Prepare the command to execute the script using the inputs from the previous step:
    `node .ruru/scripts/process_llms_json.js --baseUrl "{{context7_base_url}}" --outputDir "{{context7_output_dir}}"`
2.  **Execute Command:** Use the `execute_command` tool to run the constructed command. Ensure OS-aware syntax (Rule `05`).
3.  **Explain Command:** Inform the user about the command being executed and its purpose (processing the `llms.json` from the source URL and generating KB files). Note if source details were assumed (`detail_status == "assumed"`), the script might rely on internal defaults.
4.  **Check Exit Code:** Verify the command's exit code.
5.  **Log Result:** Log the execution attempt and the outcome (success or failure, including exit code and any relevant output).

## Acceptance Criteria

*   The `execute_command` for the script completes with an exit code of 0.
*   The target output directory (`{{context7_output_dir}}`) contains generated KB files.
*   The index file (`{{context7_output_dir}}/_index.json`) is generated.

## Error Handling

*   If the script execution fails (non-zero exit code), log the error and output, inform the user, and proceed to `{{error_step}}`.