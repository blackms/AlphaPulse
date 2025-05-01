+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-ENRICHMENT-V2-04-GENERATE_SUMMARY_RULE" # (String, Required) Unique ID for this step
title = "Step 04: Generate Context7 Summary Rule" # (String, Required) Title of this specific step.
description = """
Delegates to a Specialist Agent (e.g., agent-context-condenser or technical-writer) to generate a summary rule file for the Context7 KB. The agent scans the `kb/context7` directory for `_index.json` files, extracts topic names, and creates a rule file (e.g., `.roo/rules-[mode_slug]/05-context7-summary.md`) listing these topics and linking to the relevant index and source info files.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-ENRICHMENT-V2-03-STORE_SOURCE_INFO"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "05_update_kb_usage_strategy.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "agent-context-condenser" # (String, Optional) Agent responsible for scanning and generating the rule file.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-00-START: mode_slug",
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-02-EXECUTE_SCRIPT: context7_output_dir", # Directory to scan
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-03-STORE_SOURCE_INFO: source_info_path", # Path to source_info.json
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "summary_rule_path: Path to the generated summary rule file.",
    "summary_rule_status: Confirmation of successful rule generation.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 04: Generate Context7 Summary Rule

## Actions

1.  **Delegate Rule Generation:**
    *   Instruct the `agent-context-condenser` delegate (or `technical-writer`) to generate the summary rule.
    *   Provide the `[mode_slug]` and `[context7_output_dir]`.
    *   Provide the `[source_info_path]`.
    *   Instruct the agent to:
        *   Define the target rule directory: `[rule_dir] = .roo/rules-[mode_slug]/`. Ensure this directory exists (Prefer MCP `create_directory`, fallback `execute_command mkdir -p`).
        *   Determine the next available rule number prefix (e.g., `05`, `06`...) within `[rule_dir]` and define the path: `[summary_rule_path] = [rule_dir]/[NN]-context7-summary.md`.
        *   Scan `[context7_output_dir]` recursively for files named `_index.json` (Prefer MCP `search_filesystem`, fallback `search_files`). Store the list of found index file paths `[index_files]`.
        *   Prepare the rule content:
            *   **TOML:** Use standard rule metadata, including `id`, `title`, `context_type="rules"`, `scope="summary"`, `status="active"`, `tags=["rules", "summary", "context7", "kb", "[mode_slug]"]`, `related_context` pointing to the main `_index.json` and `source_info.json` in `kb/context7/`.
            *   **Markdown:** Include a header, description explaining the summary's purpose and linking to `source_info.json`, and a "Main Topics" section. List each found `[index_file]`, deriving the topic name from its directory path relative to `[context7_output_dir]` and providing the full workspace-relative path to the index file.
        *   Use file writing tools (Prefer MCP `write_file_content`, fallback `write_to_file`) to create/overwrite `[summary_rule_path]` with the prepared content.
        *   Report success or failure.
2.  **Receive Confirmation:** Await confirmation from the delegate. Store the `[summary_rule_path]`.

## Acceptance Criteria

*   Delegation to the specialist agent was successful.
*   The summary rule file exists at the determined `[summary_rule_path]`.
*   The rule file contains correct TOML frontmatter and a Markdown body listing the topics derived from `_index.json` files found in the `kb/context7` directory.
*   `summary_rule_status` indicates success.

## Error Handling

*   If delegation fails or the agent reports critical errors (e.g., cannot scan directory, cannot write rule file), log the error and proceed to `{{error_step}}`.
*   If no `_index.json` files are found, the agent should still create the rule file but with an empty "Main Topics" list or appropriate note.