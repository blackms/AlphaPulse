+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-07-QUERY-GIT-HISTORY" # (String, Required) Unique ID for this step.
title = "Step 07: Query Git History" # (String, Required) Title of this specific step.
description = """
(String, Required) Queries the Git history to retrieve the list of commits made
between the previously determined tag and the current HEAD (or the new tag being created).
"""

# --- Flow Control ---
depends_on = ["WF-CREATE-ROO-CMD-BUILD-V1-06-DETERMINE-PREV-TAG"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "08_generate_release_notes.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_git_query_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "dev-git" # (String, Optional) Mode responsible for executing the core logic of this step.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-06-DETERMINE-PREV-TAG: previous_git_tag",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-06-DETERMINE-PREV-TAG: determine_tag_status", # Prerequisite check
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-01-VALIDATE-PARAMS: validated_build_params", # May contain new tag name
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "commit_history_raw: Raw list/log of commits between previous_git_tag and HEAD/new_tag.",
    "query_history_status: Success or Failure.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 07: Query Git History

## Actions

1.  **Receive Context:** Get `previous_git_tag`, `determine_tag_status`, and potentially `validated_build_params` (for the new tag) from previous steps.
2.  **Check Prerequisite:** If `determine_tag_status` is Failure, immediately skip to the error step (`EE_handle_git_query_error.md`).
3.  **Define Range:** Determine the commit range for the log query. This is typically `previous_git_tag..HEAD` or `previous_git_tag..new_tag_name` if a new tag is specified in parameters.
4.  **Execute Git Log:** Run `git log` with the determined range and a suitable format (e.g., `--pretty=format:"%H %s"`) to get commit hashes and subjects.
5.  **Capture Output:** Store the raw output of the `git log` command.
6.  **Set Status:** Set `query_history_status` to Success if the `git log` command executes successfully, otherwise Failure.
7.  **Prepare Output:** Provide the `commit_history_raw` and the `query_history_status`.

## Acceptance Criteria

*   Step proceeds only if `determine_tag_status` was Success.
*   The correct commit range is identified.
*   `git log` command is executed successfully.
*   Raw commit history is captured in `commit_history_raw`.
*   `query_history_status` is set accurately.

## Error Handling

*   If prerequisite `determine_tag_status` is Failure, proceed directly to `EE_handle_git_query_error.md`.
*   If the `git log` command fails, set `query_history_status` to Failure and proceed to `EE_handle_git_query_error.md`.