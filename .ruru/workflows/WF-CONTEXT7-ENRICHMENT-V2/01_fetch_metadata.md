+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-ENRICHMENT-V2-01-FETCH_METADATA" # (String, Required) Unique ID for this step
title = "Step 01: Fetch Source Metadata" # (String, Required) Title of this specific step.
description = """
Attempts to fetch metadata (original source URL, token count, snippet count, last updated date) associated with the Context7 base URL provided in Step 00. Prioritizes using MCP tools (Vertex AI, Scraper) for automated extraction. If automated methods fail, prompts the User for manual input or allows skipping (using a default token count). Determines the final metadata values to be used later.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-ENRICHMENT-V2-00-START"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "02_execute_script.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails (e.g., user interaction fails).

# --- Execution ---
delegate_to = "roo-commander" # (String, Optional) Coordinator handles MCP calls and user interaction.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-00-START: context7_base_url",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "fetched_original_url: Original source URL if found/provided, else null.",
    "fetched_tokens: Estimated token count (numeric, fetched/provided or default), else null.",
    "fetched_snippets: Estimated snippet count if found/provided, else null.",
    "fetched_last_updated: Last updated date (YYYY-MM-DD) if found/provided, else null.",
    "metadata_fetch_method: Method used ('MCP-Vertex', 'MCP-Scraper', 'Manual', 'Fallback')."
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 01: Fetch Source Metadata

## Actions

1.  **Attempt MCP Fetch (Vertex AI):**
    *   Check if `vertex-ai-mcp-server` and `answer_query_websearch` tool are available.
    *   If yes:
        *   Formulate query: "Extract the original source URL, estimated token count, estimated snippet count, and last updated date for the documentation hosted at [context7_base_url]".
        *   Use `use_mcp_tool` (`vertex-ai-mcp-server`, `answer_query_websearch`) with the query.
        *   Attempt to parse the response for the required metadata fields.
        *   If successful, store values, set `[metadata_fetch_method]` = "MCP-Vertex", log success, and proceed to Action 4.
        *   If parsing fails or tool fails, log the failure and proceed to Action 2.
    *   If no, proceed to Action 2.
2.  **Attempt MCP Fetch (Scraper):**
    *   Check for other available scraping MCPs (e.g., `firecrawl-mcp-server`).
    *   If yes:
        *   Use `use_mcp_tool` (e.g., `firecrawl-mcp-server`, `scrape_url`) targeting `[context7_base_url]`, requesting metadata if possible.
        *   Attempt to parse the response for the required metadata fields.
        *   If successful, store values, set `[metadata_fetch_method]` = "MCP-Scraper", log success, and proceed to Action 4.
        *   If parsing fails or tool fails, log the failure and proceed to Action 3.
    *   If no, proceed to Action 3.
3.  **Attempt Manual Input:**
    *   Use `ask_followup_question`: "Could not automatically fetch metadata for [context7_base_url]. Please provide the following if known (leave blank if unknown), or type 'skip' to proceed with default token count (10,000,000) and no other metadata: Original Source URL, Estimated Token Count, Estimated Snippet Count, Last Updated Date (YYYY-MM-DD)".
    *   Store the user's input. Log the input.
    *   If user input is 'skip', proceed to Action 4 (Fallback).
    *   If user provided input, attempt to parse it, store values, set `[metadata_fetch_method]` = "Manual", log success, and proceed to Action 4.
4.  **Determine Final Values & Fallback:**
    *   If metadata values were successfully obtained via MCP or Manual input (not 'skip'):
        *   Set `[fetched_original_url]`, `[fetched_tokens]`, `[fetched_snippets]`, `[fetched_last_updated]` based on the obtained values (ensure numeric types for counts, null if not found).
    *   If all attempts failed or user chose 'skip':
        *   Set `[fetched_tokens] = 10000000`.
        *   Set `[fetched_original_url] = null`.
        *   Set `[fetched_snippets] = null`.
        *   Set `[fetched_last_updated] = null`.
        *   Set `[metadata_fetch_method]` = "Fallback".
        *   Log that fallback logic was used.

## Acceptance Criteria

*   An attempt was made to fetch metadata using available MCPs.
*   If MCPs failed, the user was prompted for manual input.
*   Final values for `[fetched_original_url]`, `[fetched_tokens]`, `[fetched_snippets]`, `[fetched_last_updated]` are determined (using fetched, manual, or fallback values).
*   `[metadata_fetch_method]` indicates how the values were obtained.

## Error Handling

*   If MCP calls fail, log the error and attempt the next method.
*   If `ask_followup_question` fails, log the error and proceed to `{{error_step}}`.
*   Parsing errors for fetched/manual data should be logged, and null values should be used for the respective fields, allowing the workflow to continue with fallback logic if necessary.