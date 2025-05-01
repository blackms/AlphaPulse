+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-REFRESH-V1-02-GENERATE-UPDATE-SUMMARY-RULE" # (String, Required) Unique ID for this step.
title = "Step 02: Generate/Update Context7 Summary Rule" # (String, Required) Title of this specific step.
description = """
(String, Required) Scans the generated `kb/context7` directory for `_index.json` files
and creates/updates a rule file (`.roo/rules-[mode_slug]/05-context7-summary.md`) listing these topics.
This provides a quick overview of the KB's scope.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-REFRESH-V1-01-EXECUTE-SCRIPT"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "03_update_create_usage_strategy.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "agent-context-condenser" # (String, Optional) Mode responsible for executing the core logic of this step. Alt: technical-writer

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: mode_slug",
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: context7_output_dir", # e.g., .ruru/modes/[mode_slug]/kb/context7
    "Output from step WF-CONTEXT7-REFRESH-V1-01-EXECUTE-SCRIPT: Generated index file", # Confirmation that _index.json exists
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "summary_rule_path: Path to the generated/updated summary rule file (e.g., .roo/rules-[mode_slug]/05-context7-summary.md).",
    "summary_rule_content: Content of the summary rule file.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 02: Generate/Update Context7 Summary Rule

## Actions

(Instructions for the delegate: `{{delegate_to}}`)

1.  **Define Paths:**
    *   `[mode_slug]` = Input `mode_slug`.
    *   `[context7_kb_dir]` = Input `context7_output_dir`.
    *   `[rules_dir]` = `.roo/rules-[mode_slug]/`.
    *   `[summary_rule_path]` = `[rules_dir]05-context7-summary.md`. (Check if `05` prefix is available in `[rules_dir]`, adjust if necessary, e.g., `06`, `07`...).
2.  **Scan for Index Files:**
    *   `- [ ]` Recursively scan `[context7_kb_dir]` to find all files named `_index.json`. (Prefer MCP `search_filesystem`, fallback `search_files` or `list_files` recursive + filter). Store the list of found paths.
3.  **Prepare Rule Content:**
    *   `- [ ]` Generate the TOML frontmatter for the rule file:
        ```toml
        +++
        id = "RULE-[MODE_SLUG]-CONTEXT7-SUMMARY-V1" # Use uppercase mode_slug
        title = "Context7 KB Scope Summary ([mode_slug])"
        context_type = "rules"
        scope = "summary"
        status = "active"
        last_updated = "2025-04-30" # Use current date
        tags = ["rules", "summary", "context7", "kb", "[mode_slug]"]
        related_context = [".ruru/modes/[mode_slug]/kb/context7/_index.json"] # Link to main index
        template_schema_doc = ".ruru/templates/toml-md/16_ai_rule.README.md"
        relevance = "Provides a quick overview of the topics covered in the Context7 KB."
        +++
        ```
    *   `- [ ]` Generate the Markdown body:
        ```markdown
        # Context7 Knowledge Base Summary for `[mode_slug]`

        This rule provides a summary of the main topics found within the Context7-derived Knowledge Base located at `.ruru/modes/[mode_slug]/kb/context7/`.

        ## Main Topics

        [For each found `_index.json` path:]
        *   [Derive topic name from directory path relative to `kb/context7/`] : [Path relative to workspace root]
            *   Example: `- Core Concepts: .ruru/modes/[mode_slug]/kb/context7/core_concepts/_index.json`
            *   Example: `- API Reference: .ruru/modes/[mode_slug]/kb/context7/api/_index.json`

        *Note: This summary is auto-generated based on the presence of `_index.json` files.*
        ```
4.  **Ensure Directory Exists:**
    *   `- [ ]` Check if `[rules_dir]` exists. If not, create it. (Prefer MCP `create_directory`, fallback `execute_command mkdir -p`).
5.  **Write Rule File:**
    *   `- [ ]` Use file writing tools (Prefer MCP `write_file_content`, fallback `write_to_file`) to create/overwrite `[summary_rule_path]` with the prepared TOML and Markdown content.
6.  **Log & Report:**
    *   `- [ ]` Log any errors encountered during scanning or writing.
    *   `- [ ]` Report completion (providing `summary_rule_path` and `summary_rule_content`) or failure back to the Coordinator.

## Acceptance Criteria

*   The summary rule file exists at the determined `[summary_rule_path]`.
*   The rule file contains valid TOML frontmatter and Markdown content.
*   The Markdown body correctly lists the main topics derived from the found `_index.json` files, linking to their paths.

## Error Handling

*   Failure during directory scanning or file writing should be logged.
*   If critical errors occur (e.g., cannot write file), proceed to `{{error_step}}`.