+++
# --- Step Metadata ---
step_id = "WF-RELEASE-NOTES-HYBRID-V2-02-PARSE_COMMITS" # Unique ID
title = "Step 02: Parse & Filter Commits" # Based on original step 3
description = """
Parses the raw Git log output, filters for relevant commits (e.g., conventional commits),
and extracts key information like type, scope, subject, hash, and related Task IDs.
"""

# --- Flow Control ---
depends_on = ["WF-RELEASE-NOTES-HYBRID-V2-01-QUERY_GIT_HISTORY"] # Depends on the git log query
next_step = "03_summarize_changes.md" # Based on original step 4
error_step = "99_finish.md" # Default error handling

# --- Execution ---
delegate_to = "" # No specific delegate in original; assume orchestrator/general mode handles parsing

# --- Interface ---
inputs = [
    "Output from step WF-RELEASE-NOTES-HYBRID-V2-01-QUERY_GIT_HISTORY: raw_git_log_output"
]
outputs = [
    "parsed_commits_data: Structured data (e.g., JSON object/map) containing lists of parsed commits, grouped by conventional commit type (features, fixes, etc.). Each commit includes hash, subject, scope (if any), and taskId (if any)."
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # Placeholder
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md"
+++

# Step 02: Parse & Filter Commits

## Actions

1.  **Process Raw Log:** Take the `raw_git_log_output` string as input.
2.  **Split into Commits:** Split the raw log into individual commit blocks using the `---COMMIT-END---` delimiter.
3.  **Parse Each Commit:** For each commit block:
    *   Parse the line using the ` ||| ` delimiter to separate the commit hash, subject line, and body.
    *   **Identify Conventional Commit Type:** Analyze the subject line to determine the conventional commit type (e.g., `feat`, `fix`, `perf`, `refactor`, `chore`, `docs`, `test`). Assumes adherence to a standard like Conventional Commits v1.0.0 (see rule `.roo/rules/07-git-commit-standard-simplified.md`).
    *   **Filter:** Skip commits that do not match a recognized conventional commit type or appear to be merge commits (unless explicitly configured to include them).
    *   **Extract Scope:** If a scope is present in the subject (e.g., `feat(api): ...`), extract the scope (`api`).
    *   **Extract Task ID(s):** Search the commit body (footer) for lines matching the pattern `Refs: TASK-...` (or similar configured standard). Extract *all* found Task IDs (e.g., into a list). Assumes a consistent reference format.
    *   **Store Parsed Data:** Store the extracted hash, type, scope (optional), subject, and Task ID(s) (optional list) for this commit.
4.  **Group Commits:** Group the stored parsed commit data by their conventional commit type (e.g., create lists under keys like `features`, `fixes`, `performance`, etc.).
5.  **Output Structured Data:** Output the final grouped data structure as `parsed_commits_data`.

## Acceptance Criteria

*   The `raw_git_log_output` is successfully parsed.
*   Commits are correctly filtered based on conventional commit types.
*   Relevant information (hash, type, scope, subject, Task ID) is extracted accurately.
*   The `parsed_commits_data` output contains the structured, grouped commit information.

## Error Handling

*   If parsing fails due to unexpected log format, proceed to `{{error_step}}`.