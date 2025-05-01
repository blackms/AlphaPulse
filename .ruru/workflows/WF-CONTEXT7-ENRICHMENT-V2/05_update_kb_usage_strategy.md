+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-ENRICHMENT-V2-05-UPDATE_KB_USAGE_STRATEGY" # (String, Required) Unique ID for this step
title = "Step 05: Update/Create KB Usage Strategy" # (String, Required) Title of this specific step.
description = """
Delegates to the Mode Structure Agent (e.g., mode-maintainer) to ensure the mode's KB usage strategy document (`kb/00-kb-usage-strategy.md`) exists. If it doesn't exist, the agent creates it using a standard template or content. If it already exists, no action is taken besides logging.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-ENRICHMENT-V2-04-GENERATE_SUMMARY_RULE"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "06_update_mode_definition.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "mode-maintainer" # (String, Optional) Mode responsible for checking/creating the file.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-00-START: mode_slug",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "usage_strategy_path: Path to the usage strategy document.",
    "usage_strategy_status: Confirmation that the strategy doc exists or was created.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 05: Update/Create KB Usage Strategy

## Actions

1.  **Delegate Strategy Check/Creation:**
    *   Instruct the `mode-maintainer` delegate to ensure the KB usage strategy document exists.
    *   Provide the `[mode_slug]`.
    *   Instruct the agent to:
        *   Define the path: `[usage_strategy_path] = .ruru/modes/[mode_slug]/kb/00-kb-usage-strategy.md`.
        *   Check if `[usage_strategy_path]` exists (Prefer MCP `get_filesystem_info`, fallback `list_files`).
        *   If it does **not** exist:
            *   Create the file at `[usage_strategy_path]` using standard content (e.g., copied from a template like `.ruru/templates/mode/kb/00-kb-usage-strategy.md` if available, or basic boilerplate). (Prefer MCP `write_file_content`, fallback `write_to_file`). Handle creation errors -> **Log Warning/Non-critical Failure**.
            *   Log that the file was created.
        *   If it **does** exist:
            *   Log that the file already exists. No creation needed.
2.  **Receive Confirmation:** Await confirmation of success or failure from the delegate. Store the `[usage_strategy_path]`.

## Acceptance Criteria

*   Delegation to `mode-maintainer` was successful.
*   The KB usage strategy document exists at `.ruru/modes/[mode_slug]/kb/00-kb-usage-strategy.md`.
*   `usage_strategy_status` indicates whether the file existed or was created.

## Error Handling

*   If delegation fails, log the error and proceed to `{{error_step}}`.
*   If the agent fails to *create* the strategy document when it's missing, log a warning and proceed to the next step (this is desirable but not strictly critical for the rest of the enrichment).