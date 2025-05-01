+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-08-GENERATE-RELEASE-NOTES" # (String, Required) Unique ID for this step.
title = "Step 08: Generate Release Notes" # (String, Required) Title of this specific step.
description = """
(String, Required) Parses the raw commit history retrieved in the previous step
and generates formatted Markdown release notes based on commit messages.
"""

# --- Flow Control ---
depends_on = ["WF-CREATE-ROO-CMD-BUILD-V1-07-QUERY-GIT-HISTORY"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "09_save_release_notes.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_notes_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "technical-writer" # (String, Optional) Mode responsible for executing the core logic of this step.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-07-QUERY-GIT-HISTORY: commit_history_raw",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-07-QUERY-GIT-HISTORY: query_history_status", # Prerequisite check
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-01-VALIDATE-PARAMS: validated_build_params", # For new version/tag
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "release_notes_markdown: Generated release notes content in Markdown format.",
    "generate_notes_status: Success or Failure.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 08: Generate Release Notes

## Actions

1.  **Receive Context:** Get `commit_history_raw`, `query_history_status`, and `validated_build_params` from previous steps.
2.  **Check Prerequisite:** If `query_history_status` is Failure, immediately skip to the error step (`EE_handle_notes_error.md`).
3.  **Parse Commits:** Process the `commit_history_raw` data. Extract relevant information (e.g., commit messages, types like 'feat', 'fix', 'chore'). Filter out irrelevant commits if necessary (e.g., merge commits, chore commits depending on policy).
4.  **Format Notes:** Structure the parsed information into Markdown format. Typically includes sections for Features, Bug Fixes, etc., listing the relevant commit messages. Include the new version/tag from `validated_build_params`.
5.  **Set Status:** Set `generate_notes_status` to Success if parsing and formatting are successful, otherwise Failure.
6.  **Prepare Output:** Provide the generated `release_notes_markdown` and the `generate_notes_status`.

## Acceptance Criteria

*   Step proceeds only if `query_history_status` was Success.
*   Raw commit history is successfully parsed.
*   Release notes are generated in Markdown format, including the new version number.
*   `release_notes_markdown` contains the formatted content.
*   `generate_notes_status` is set accurately.

## Error Handling

*   If prerequisite `query_history_status` is Failure, proceed directly to `EE_handle_notes_error.md`.
*   If parsing the commit history fails or formatting the notes fails, set `generate_notes_status` to Failure and proceed to `EE_handle_notes_error.md`.