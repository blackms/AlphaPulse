+++
# --- Step Metadata ---
step_id = "WF-RELEASE-NOTES-HYBRID-V2-01-QUERY_GIT_HISTORY" # Unique ID for this step
title = "Step 01: Query Git History" # Based on original step 2
description = """
Queries the Git history between the determined previous tag and the target tag
using 'git log' to retrieve commit information for release notes generation.
"""

# --- Flow Control ---
depends_on = ["WF-RELEASE-NOTES-HYBRID-V2-00B-DETERMINE_PREVIOUS_TAG"] # Depends on the previous tag determination step
next_step = "02_parse_commits.md" # Based on original step 3
error_step = "99_finish.md" # Default error handling

# --- Execution ---
delegate_to = "dev-git" # As specified in the original workflow

# --- Interface ---
inputs = [
    "Output from step WF-RELEASE-NOTES-HYBRID-V2-00B-DETERMINE_PREVIOUS_TAG: resolved_previous_tag", # Use output from 00b
    "Output from step WF-RELEASE-NOTES-HYBRID-V2-00-START: validated_target_tag"
]
outputs = [
    "raw_git_log_output: The raw string output from the 'git log' command."
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # Placeholder
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md"
+++

# Step 01: Query Git History

## Actions

1.  **Delegate to `dev-git`:** Instruct the `dev-git` mode to execute the following command, replacing the placeholders with the actual tag values received as input:
    ```bash
    git log --pretty=format:"%H ||| %s ||| %b%n---COMMIT-END---%n" {{resolved_previous_tag}}..{{validated_target_tag}}
    ```
    *   Note: Assumes `resolved_previous_tag` provided by the previous step is a valid Git reference (tag).
    *   Note: Assumes `dev-git` delegate can correctly execute this `git log` command.
    *   The specific format with `|||` and `---COMMIT-END---` is crucial for parsing in the next step.
2.  **Store Output:** Capture the complete raw output from the `git log` command executed by `dev-git` and store it as `raw_git_log_output`.

## Acceptance Criteria

*   Input `resolved_previous_tag` is confirmed to be a valid Git reference.
*   The `git log` command is executed successfully by `dev-git`.
*   The complete raw output is captured and stored in `raw_git_log_output`.

## Error Handling

*   If `dev-git` reports an error executing the command (e.g., invalid tags, Git repository issues), capture the error and proceed to `{{error_step}}`.