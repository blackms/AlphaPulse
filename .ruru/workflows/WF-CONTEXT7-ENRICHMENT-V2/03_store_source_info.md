+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-ENRICHMENT-V2-03-STORE_SOURCE_INFO" # (String, Required) Unique ID for this step
title = "Step 03: Store Source Info & Metadata" # (String, Required) Title of this specific step.
description = """
Delegates to a File Editor Agent (e.g., prime-dev) to create or update the `source_info.json` file within the `kb/context7` directory. This file stores the original Context7 base URL, the derived `llms.json` URL, and the metadata (original source URL, tokens, snippets, last updated) fetched or determined in Step 01.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-ENRICHMENT-V2-02-EXECUTE_SCRIPT"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "04_generate_summary_rule.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "prime-dev" # (String, Optional) Agent responsible for creating/editing the JSON file.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-00-START: mode_slug",
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-00-START: context7_base_url",
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-01-FETCH_METADATA: fetched_original_url",
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-01-FETCH_METADATA: fetched_tokens",
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-01-FETCH_METADATA: fetched_snippets",
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-01-FETCH_METADATA: fetched_last_updated",
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-02-EXECUTE_SCRIPT: context7_output_dir", # To confirm directory exists
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "source_info_path: Path to the created/updated source_info.json file.",
    "source_info_status: Confirmation of successful creation/update.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 03: Store Source Info & Metadata

## Actions

1.  **Delegate JSON Creation/Update:**
    *   Instruct the `prime-dev` delegate (or similar File Editor Agent) to create/update the `source_info.json` file.
    *   Provide the `[mode_slug]` and `[context7_base_url]`.
    *   Provide the fetched metadata: `[fetched_original_url]`, `[fetched_tokens]`, `[fetched_snippets]`, `[fetched_last_updated]` (values might be null).
    *   Instruct the agent to:
        *   Define the target path: `[source_info_path] = .ruru/modes/[mode_slug]/kb/context7/source_info.json`.
        *   Construct the `llms.json` URL: `[llms_json_url] = [context7_base_url]/llms.json`.
        *   Prepare the JSON content precisely as follows (using null for missing optional metadata):
            ```json
            {
              "baseUrl": "[context7_base_url]",
              "llmsJsonUrl": "[llms_json_url]",
              "originalSourceUrl": [fetched_original_url or null],
              "metadata": {
                "tokens": [fetched_tokens or null],
                "snippets": [fetched_snippets or null],
                "lastUpdated": "[fetched_last_updated or null]"
              }
            }
            ```
        *   Ensure the output is valid JSON (e.g., numeric values for tokens/snippets if not null, string or null for dates/URLs).
        *   Use file writing tools (Prefer MCP `write_file_content`, fallback `write_to_file`) to create/overwrite `[source_info_path]` with the prepared JSON content.
        *   Report success or failure.
2.  **Receive Confirmation:** Await confirmation from the delegate. Store the `[source_info_path]`.

## Acceptance Criteria

*   Delegation to the File Editor Agent was successful.
*   The file `source_info.json` exists at `.ruru/modes/[mode_slug]/kb/context7/source_info.json`.
*   The file contains valid JSON with `baseUrl`, `llmsJsonUrl`, and the `metadata` object (including `tokens` which should have a value, and potentially null values for other metadata fields).
*   `source_info_status` indicates success.

## Error Handling

*   If delegation fails or the agent reports an error writing the JSON file, log the error and proceed to `{{error_step}}`.