+++
# --- Workflow Metadata ---
id = "WF-CONTEXT7-REFRESH-V1" # (String, Required) Unique identifier
title = "Workflow: Context7 KB Refresh" # (String, Required)
description = """
(String, Required) Defines the step-by-step procedure for *refreshing* an existing specialist mode's Context7 KB.
This involves cleaning the existing directory, retrieving the source URL, optionally attempting to fetch source details
(like token count, update date) via MCP/Crawling or user input, falling back to assumptions if needed,
re-running the `.ruru/scripts/process_llms_json.js` script, and updating related artifacts.
"""
version = "1.0.0" # (String, Required) Semantic version for the workflow definition.
status = "Draft" # (String, Required) Current status: "Draft", "Active", "Deprecated", "Experimental".
tags = [
    "workflow", "kb-enrichment", "context7", "script-integration", "modes", "pipeline",
    "documentation", "sop", "context-handling", "json-context", "subfolders",
    "existing-mode", "url", "summary-rule", "refresh", "source-info", "mcp", "crawl", "fallback"
] # (Array of Strings, Optional) Keywords for search/categorization.

# --- Execution Control ---
entry_point = "00_start.md" # (String, Required) Filename of the first step to execute.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Describe overall inputs needed to start the workflow.
    "Target existing mode slug.",
    "Confirmation of Context7 base URL.",
    "Decision on fetching source details (Yes/No).",
    "(Optional) Manually provided source details (e.g., token count) if fetching fails or is skipped.",
]
outputs = [ # (Array of Strings, Optional) Describe the expected final artifacts or outcomes.
    "Refreshed KB files within the target mode's `kb/context7/` directory.",
    "A valid `_index.json` file within `kb/context7/`.",
    "An updated or created Context7 summary rule file at `.roo/rules-[mode_slug]/05-context7-summary.md` (or similar prefix).",
    "An updated KB README (`kb/README.md`) for the target mode, referencing the `context7` directory, source URL, and detail status.",
    "Confirmation that a KB usage strategy document exists at `kb/00-kb-usage-strategy.md`.",
    "An updated mode definition file (`.mode.md`) for the target mode, referencing `kb/context7/_index.json`.",
    "Summary report of the refresh process.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/23_workflow_readme.md" # (String, Required) Link to this template definition.
related_docs = [
    ".ruru/workflows/WF-CONTEXT7-ENRICHMENT-001.md", # Source workflow (old format)
    ".ruru/scripts/process_llms_json.js",
    ".roo/rules/01-standard-toml-md-format.md",
    ".roo/rules/04-mdtm-workflow-initiation.md",
    ".roo/rules/08-logging-procedure-simplified.md",
    ".roo/rules/10-vertex-mcp-usage-guideline.md",
    ".ruru/processes/acqa-process.md",
    ".ruru/processes/afr-process.md",
    ".ruru/processes/pal-process.md"
] # (Array of Strings, Optional) Links to related rules, KBs, ADRs.
+++

# Workflow: Context7 KB Refresh

## Overview

This workflow defines the step-by-step procedure for **refreshing** an existing specialist mode's Knowledge Base (KB) using AI-processed context derived **specifically** from a Context7 base URL.

The process involves:
1.  Identifying the target mode and retrieving its Context7 source URL.
2.  Optionally attempting to fetch source details (like token count, update date) using MCP/Crawling tools, or falling back to user input/assumptions.
3.  Cleaning the existing `kb/context7` directory.
4.  Executing the `.ruru/scripts/process_llms_json.js` script to generate new KB files from the source.
5.  Generating/updating a summary rule file (`.roo/rules-[mode_slug]/...`) reflecting the new KB content.
6.  Ensuring a KB usage strategy document exists.
7.  Updating the mode's KB README and definition file (`.mode.md`) to reference the new context.
8.  Performing Quality Assurance (QA) and facilitating User review.

## Usage

This workflow is typically initiated manually by a Coordinator (like Roo Commander) when an existing mode's Context7-derived knowledge base needs to be updated from its source URL. The Coordinator will guide the process, interacting with the User for necessary inputs and confirmations, and delegating specific tasks to specialist agents.

## Inputs

*   **Target Mode Slug:** The unique identifier of the existing specialist mode whose KB needs refreshing (e.g., `spec-huggingface`).
*   **Context7 Base URL Confirmation:** Verification of the URL pointing to the `llms.json` source file. This is usually retrieved from the mode's existing `kb/context7/source_info.json`.
*   **Detail Fetching Decision:** User choice on whether to attempt fetching source metadata (token count, update date) via automated tools.
*   **(Optional) Manual Source Details:** If automated fetching is skipped or fails, the user might need to provide estimated details like token count.

## Outputs

*   **Refreshed KB Files:** The `.ruru/modes/[mode_slug]/kb/context7/` directory will be populated with newly generated Markdown files based on the source `llms.json`.
*   **KB Index:** A valid `.ruru/modes/[mode_slug]/kb/context7/_index.json` file summarizing the generated KB structure.
*   **Summary Rule:** An updated `.roo/rules-[mode_slug]/05-context7-summary.md` (or similar) file listing the topics covered in the refreshed KB.
*   **Updated KB README:** The `.ruru/modes/[mode_slug]/kb/README.md` file will be updated to reflect the presence of the `context7` data, its source URL, and whether source details were obtained or assumed.
*   **KB Usage Strategy:** Confirmation that `.ruru/modes/[mode_slug]/kb/00-kb-usage-strategy.md` exists.
*   **Updated Mode Definition:** The `.ruru/modes/[mode_slug]/[mode_slug].mode.md` file will have `kb/context7/_index.json` added to its `related_context`.
*   **Final Report:** A summary confirming the successful completion of the refresh process.