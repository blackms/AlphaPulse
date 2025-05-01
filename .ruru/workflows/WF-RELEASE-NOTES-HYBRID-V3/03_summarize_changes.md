+++
# --- Step Metadata ---
step_id = "WF-RELEASE-NOTES-HYBRID-V2-03-SUMMARIZE_CHANGES" # Unique ID
title = "Step 03: Summarize Changes" # Based on original step 4
description = """
Formats the parsed commit data into Markdown sections (Features, Bug Fixes, etc.).
Optionally enhances summaries by fetching corresponding MDTM task titles if Task IDs are present.
Combines sections into a single release notes body string.
"""

# --- Flow Control ---
depends_on = ["WF-RELEASE-NOTES-HYBRID-V2-02-PARSE_COMMITS"] # Depends on parsed commit data
next_step = "04_generate_local_notes.md" # Based on original step 5
error_step = "99_finish.md" # Default error handling

# --- Execution ---
# Delegate for optional task title fetching is handled within the actions below.
# Core summarization logic might be handled by orchestrator or a text processing mode.
delegate_to = ""

# --- Interface ---
inputs = [
    "Output from step WF-RELEASE-NOTES-HYBRID-V2-02-PARSE_COMMITS: parsed_commits_data"
]
outputs = [
    "release_notes_body: A single Markdown string containing the formatted release notes content."
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # Placeholder
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md"
+++

# Step 03: Summarize Changes

## Actions

1.  **Initialize Sections:** Create empty Markdown strings or lists for standard sections (e.g., `### ✨ Features`, `### 🐞 Bug Fixes`, `### 🚀 Performance Improvements`, `### ♻️ Refactoring`, `### ⚙️ Miscellaneous Chores`, etc.).
2.  **Iterate Through Parsed Commits:** Loop through the `parsed_commits_data` structure, processing each commit group (features, fixes, etc.).
3.  **Format Each Commit Entry:** For each commit within a group:
    *   Determine the summary text:
        *   Check if the commit data includes a `taskId`.
        *   If `taskId` exists:
            *   Delegate to `agent-context-resolver` or `prime-txt` to read the corresponding MDTM task file. (Assumes standard path like `.ruru/tasks/.../{{taskId}}.md`).
            *   Attempt to extract the `title` field from the task file's TOML frontmatter.
            *   Use the fetched task `title` as the summary text. If fetching fails or the title is empty, log a warning and fall back to the commit subject line.
        *   If no `taskId` exists:
            *   Use the commit subject line as the summary text.
    *   Format the Markdown list item for the commit. Example format:
        *   `- {{Scope (if present) + ': '}}{{Summary Text}} ({{Commit Hash}}`{{Refs: TaskID (if present)}}`)`
        *   *Example:* `- api: Implement user login endpoint (a1b2c3d, Refs: TASK-API-123)`
        *   *Example:* `- Fix user profile update bug (e4f5g6h)`
    *   **Append to Section:** Add the formatted line to the appropriate Markdown section string/list.
4.  **Combine Sections:** Concatenate the populated Markdown sections into a single string, `release_notes_body`. Ensure appropriate headings and spacing.

## Acceptance Criteria

*   All relevant commits from `parsed_commits_data` are processed.
*   Commit summaries are correctly formatted into Markdown list items.
*   (Optional) MDTM task titles are used for summaries where Task IDs were found and resolvable.
*   The final `release_notes_body` string contains the complete, formatted Markdown content for the release notes.

## Error Handling

*   If fetching an MDTM task title fails (optional step), log a warning and fall back gracefully to using the commit subject line. Do not fail the entire step.
*   If formatting or combining sections fails unexpectedly, proceed to `{{error_step}}`.