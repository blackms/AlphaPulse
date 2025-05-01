+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-REFRESH-V1-03-UPDATE-CREATE-USAGE-STRATEGY" # (String, Required) Unique ID for this step.
title = "Step 03: Update/Create KB Usage Strategy" # (String, Required) Title of this specific step.
description = """
(String, Required) Ensures that the target mode has a KB usage strategy document
(`.ruru/modes/[mode_slug]/kb/00-kb-usage-strategy.md`). If it doesn't exist, it creates a basic one.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-REFRESH-V1-02-GENERATE-UPDATE-SUMMARY-RULE"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "04_update_mode_definition.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "mode-maintainer" # (String, Optional) Mode responsible for executing the core logic of this step. Alt: technical-writer

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CONTEXT7-REFRESH-V1-00-START: mode_slug",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "usage_strategy_path: Path to the usage strategy file.",
    "usage_strategy_status: 'Exists' or 'Created'.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 03: Update/Create KB Usage Strategy

## Actions

(Instructions for the delegate: `{{delegate_to}}`)

1.  **Define Path:**
    *   `[mode_slug]` = Input `mode_slug`.
    *   `[usage_strategy_path]` = `.ruru/modes/[mode_slug]/kb/00-kb-usage-strategy.md`.
2.  **Check Existence:**
    *   `- [ ]` Check if `[usage_strategy_path]` exists. (Prefer MCP `get_filesystem_info`, fallback `list_files`).
3.  **Create if Missing:**
    *   `- [ ]` If the file does NOT exist:
        *   Prepare basic content for the usage strategy file (e.g., using a standard template or boilerplate text like below):
          ```markdown
          +++
          id = "KB-USAGE-STRATEGY-[MODE_SLUG]" # Use uppercase mode_slug
          title = "KB Usage Strategy: [mode_slug]"
          context_type = "knowledge_base"
          scope = "Defines how the [mode_slug] mode should utilize its knowledge base"
          target_audience = ["[mode_slug]"]
          tags = ["kb", "strategy", "usage", "[mode_slug]"]
          +++

          # KB Usage Strategy: `[mode_slug]`

          ## Overview

          This document outlines how the `[mode_slug]` mode should prioritize and utilize the information contained within its knowledge base (KB) located in the `.ruru/modes/[mode_slug]/kb/` directory.

          ## Prioritization

          1.  **Rules (`.roo/rules-[mode_slug]/`):** Mandatory operational constraints MUST be followed.
          2.  **Local KB (`kb/*.md`, excluding README):** Specific procedures, guidelines, and examples relevant to this mode's tasks.
          3.  **Context7 KB (`kb/context7/`):** Broader context derived from external documentation (refer to `kb/context7/_index.json` for topics). Use this for understanding concepts, APIs, and background information related to the source documentation.
          4.  **KB README (`kb/README.md`):** Provides an overview and index of the local KB files.
          5.  **General Project Rules (`.roo/rules/`):** Workspace-wide standards and conventions.

          ## Process

          *   Before starting a task, consult relevant rules and local KB documents.
          *   Use the Context7 KB as needed for background information related to the source documentation.
          *   Refer to the KB README to locate specific local KB files.
          *   Always adhere to general project rules unless overridden by mode-specific rules.
          ```
        *   Use file writing tools (Prefer MCP `write_file_content`, fallback `write_to_file`) to create `[usage_strategy_path]` with the content.
        *   Set `[usage_strategy_status]` = "Created".
    *   `- [ ]` If the file exists:
        *   Log that it already exists.
        *   Set `[usage_strategy_status]` = "Exists".
4.  **Log & Report:**
    *   `- [ ]` Log any errors encountered.
    *   `- [ ]` Report completion (providing `usage_strategy_path` and `usage_strategy_status`) or failure back to the Coordinator.

## Acceptance Criteria

*   The file `[usage_strategy_path]` exists.
*   The status ('Exists' or 'Created') is reported correctly.

## Error Handling

*   Failure to check for or create the file should be logged.
*   If critical errors occur (e.g., cannot write file), proceed to `{{error_step}}`.