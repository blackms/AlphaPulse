+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-13-PUSH-TAG" # (String, Required) Unique ID for this step.
title = "Step 13: Push Git Tag" # (String, Required) Title of this specific step.
description = """
(String, Required) Pushes the Git tag associated with the new release version
(specified in build parameters) to the remote origin repository.
"""

# --- Flow Control ---
depends_on = ["WF-CREATE-ROO-CMD-BUILD-V1-12-COMMIT-DOCS"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "14_create_github_release.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_git_tag_push_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "dev-git" # (String, Optional) Mode responsible for executing the core logic of this step.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-12-COMMIT-DOCS: commit_status", # Prerequisite check
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-01-VALIDATE-PARAMS: validated_build_params", # Contains the new tag name (e.g., version)
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "tag_push_status: Success or Failure.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 13: Push Git Tag

## Actions

1.  **Receive Context:** Get `commit_status` and `validated_build_params` from previous steps.
2.  **Check Prerequisite:** If `commit_status` is Failure, immediately skip to the error step (`EE_handle_git_tag_push_error.md`).
3.  **Get Tag Name:** Extract the new version/tag name from `validated_build_params`.
4.  **Create Tag (Locally):** Execute `git tag <tag_name>` using the extracted tag name. This assumes the tag doesn't exist yet; robust implementation might check first or use `-f`.
5.  **Push Tag:** Execute `git push origin <tag_name>` to push the specific tag to the remote repository.
6.  **Set Status:** Set `tag_push_status` to Success if the tag is created and pushed successfully, otherwise Failure.
7.  **Prepare Output:** Provide the `tag_push_status`.

## Acceptance Criteria

*   Step proceeds only if `commit_status` was Success.
*   The new tag name is correctly identified.
*   The Git tag is created locally.
*   The Git tag is successfully pushed to the origin remote.
*   `tag_push_status` is set accurately.

## Error Handling

*   If prerequisite `commit_status` is Failure, proceed directly to `EE_handle_git_tag_push_error.md`.
*   If `git tag` or `git push origin <tag_name>` commands fail, set `tag_push_status` to Failure and proceed to `EE_handle_git_tag_push_error.md`.