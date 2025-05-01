+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-REFRESH-V1-00-START" # (String, Required) Unique ID for this step.
title = "Step 00: Initiation, URL Retrieval, Detail Handling & Cleanup" # (String, Required) Title of this specific step.
description = """
(String, Required) Gathers target mode from user, retrieves/confirms Context7 source URL,
handles optional fetching of source details (token count, update date) with user interaction and fallback,
confirms details with user, and cleans the existing context7 output directory before proceeding.
"""

# --- Flow Control ---
depends_on = [] # (Array of Strings, Required) Always empty for the start step.
next_step = "01_execute_processing_script.md" # (String, Required) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "roo-commander" # (String, Optional) Coordinator handles this initial setup phase.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Usually references overall workflow inputs.
    "User request: Approximate name/slug of target mode.",
    "(Potentially) User input: Context7 base URL if not found in source_info.json.",
    "User decision: Attempt fetching source details (Yes/No).",
    "(Potentially) User input: Manual source details (e.g., token count) if fetching fails or is skipped.",
    "User confirmation: Target mode, base URL, and detail status.",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "mode_slug: Confirmed slug of the target mode.",
    "context7_base_url: Confirmed base URL for Context7 source.",
    "detail_status: Status of source detail fetching ('obtained' or 'assumed').",
    "source_token_count: Token count (fetched, manual, or assumed).",
    "source_update_date: Update date if obtained.",
    "context7_output_dir: Path to the cleaned output directory (e.g., '.ruru/modes/[mode_slug]/kb/context7').",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/24_workflow_step_start.md" # (String, Required) Link to this template definition.
+++

# Step 00: Initiation, URL Retrieval, Detail Handling & Cleanup

## Actions

1.  **Identify Target Mode:**
    *   Ask User for approximate name/slug (`[approx_mode_name]`).
    *   List modes in `.ruru/modes/`.
    *   Filter list by `[approx_mode_name]`.
    *   Confirm exact `[mode_slug]` with User (use `ask_followup_question` if multiple matches). Handle errors/cancellation (**-> Error Step**).
2.  **Retrieve/Confirm Source URL:**
    *   Define `[source_info_path] = .ruru/modes/[mode_slug]/kb/context7/source_info.json`.
    *   Read `[source_info_path]`. If error (**-> Error Step**).
    *   Parse JSON, extract `"baseUrl"` as `[context7_base_url]`.
    *   If file missing, invalid JSON, or no `"baseUrl"` key: Ask User for URL via `ask_followup_question`. Handle errors/cancellation (**-> Error Step**).
3.  **Handle Source Details (Optional Fetching):**
    *   Ask User: Attempt fetching details from `[context7_base_url]`? (Yes/No).
    *   If Yes:
        *   Delegate to Specialist (e.g., MCP tool user, crawler mode) to fetch/parse `llms.json` from `[context7_base_url]`.
        *   If success: Extract `total_tokens`, `last_updated`. Store details. Set `[detail_status] = "obtained"`.
        *   If fail: Log failure. Ask User: Provide details manually or use defaults?
            *   Manual: Ask for details (e.g., token count). Store. Set `[detail_status] = "obtained"`.
            *   Defaults: Set `[source_token_count] = 10000000` (or other default). Log fallback. Set `[detail_status] = "assumed"`.
    *   If No (User skips fetch): Set `[source_token_count] = 10000000` (or other default). Log fallback. Set `[detail_status] = "assumed"`.
4.  **Confirm Inputs:**
    *   Ask User via `ask_followup_question` to confirm `[mode_slug]`, `[context7_base_url]`, and acknowledge `[detail_status]` (and obtained/assumed details). Handle rejection (**-> Error Step**).
    *   Log confirmed inputs.
5.  **Clean Output Directory:**
    *   Define `[context7_output_dir] = .ruru/modes/[mode_slug]/kb/context7`.
    *   Execute `rm -rf "[context7_output_dir]"` (OS-aware). Explain command.
    *   Log execution attempt. Handle permission errors (**-> Error Step**).

## Acceptance Criteria

*   `[mode_slug]` is confirmed by the User.
*   `[context7_base_url]` is confirmed by the User.
*   `[detail_status]` is determined ("obtained" or "assumed") and acknowledged by the User.
*   Relevant details (`source_token_count`, `source_update_date`) are stored based on `[detail_status]`.
*   The directory `[context7_output_dir]` has been removed (or removal attempted without permission errors).
*   All necessary outputs (`mode_slug`, `context7_base_url`, `detail_status`, etc.) are prepared for the next step (`{{next_step}}`).

## Error Handling

*   Failure to identify/confirm `[mode_slug]` or `[context7_base_url]` proceeds to `{{error_step}}`.
*   User rejection during confirmation proceeds to `{{error_step}}`.
*   Permission errors during directory cleaning proceed to `{{error_step}}`.