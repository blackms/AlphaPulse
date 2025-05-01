+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-12-COMMIT-DOCS" # (String, Required) Unique ID for this step.
title = "Step 12: Commit Documentation Changes" # (String, Required) Title of this specific step.
description = """
(String, Required) Commits the updated documentation files (README.md, CHANGELOG.md,
and the new release notes file) to the Git repository.
"""

# --- Flow Control ---
depends_on = ["WF-CREATE-ROO-CMD-BUILD-V1-11-UPDATE-CHANGELOG"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "13_push_tag.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_git_commit_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "dev-git" # (String, Optional) Mode responsible for executing the core logic of this step.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-09-SAVE-RELEASE-NOTES: release_notes_file_path",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-11-UPDATE-CHANGELOG: changelog_update_status", # Prerequisite check
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-01-VALIDATE-PARAMS: validated_build_params", # For commit message (version)
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "commit_status: Success or Failure.",
    "commit_hash: The SHA hash of the documentation commit.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 12: Commit Documentation Changes

## Actions

1.  **Receive Context:** Get `release_notes_file_path`, `changelog_update_status`, and `validated_build_params` from previous steps.
2.  **Check Prerequisite:** If `changelog_update_status` is Failure, immediately skip to the error step (`EE_handle_git_commit_error.md`).
3.  **Stage Files:** Add the modified `README.md`, `CHANGELOG.md`, and the newly created `release_notes_file_path` to the Git staging area (`git add`).
4.  **Create Commit Message:** Generate a standard commit message (e.g., "docs: Update documentation for release vX.Y.Z" using the version from `validated_build_params`).
5.  **Commit Changes:** Execute `git commit` with the generated message.
6.  **Get Commit Hash:** Retrieve the SHA hash of the newly created commit.
7.  **Set Status:** Set `commit_status` to Success if the commit is successful, otherwise Failure.
8.  **Prepare Output:** Provide the `commit_status` and `commit_hash`.

## Acceptance Criteria

*   Step proceeds only if `changelog_update_status` was Success.
*   The correct documentation files (`README.md`, `CHANGELOG.md`, release notes file) are staged.
*   A standard commit message is generated.
*   The changes are successfully committed to the repository.
*   The `commit_hash` is retrieved.
*   `commit_status` is set accurately.

## Error Handling

*   If prerequisite `changelog_update_status` is Failure, proceed directly to `EE_handle_git_commit_error.md`.
*   If `git add` or `git commit` commands fail, set `commit_status` to Failure and proceed to `EE_handle_git_commit_error.md`.