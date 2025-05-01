+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-14-CREATE-GITHUB-RELEASE" # (String, Required) Unique ID for this step.
title = "Step 14: Create GitHub Release (Optional)" # (String, Required) Title of this specific step.
description = """
(String, Required) Creates a new release on GitHub using the pushed tag,
attaching the packaged build artifact and using the generated release notes.
This step is conditional based on the `create_github_release` input parameter.
"""

# --- Flow Control ---
depends_on = ["WF-CREATE-ROO-CMD-BUILD-V1-13-PUSH-TAG"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "99_finish.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_github_release_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "lead-devops" # (String, Optional) Mode responsible for executing the core logic of this step.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-13-PUSH-TAG: tag_push_status", # Prerequisite check
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-01-VALIDATE-PARAMS: validated_build_params", # Contains tag name, create_github_release flag
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-08-GENERATE-RELEASE-NOTES: release_notes_markdown", # Body for the release
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-05-PACKAGE-ARTIFACTS: packaged_artifact_path", # Asset to upload
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "github_release_status: Success, Failure, or Skipped.",
    "github_release_url: URL of the created GitHub release (if successful).",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 14: Create GitHub Release (Optional)

## Actions

1.  **Receive Context:** Get `tag_push_status`, `validated_build_params`, `release_notes_markdown`, and `packaged_artifact_path` from previous steps.
2.  **Check Prerequisite:** If `tag_push_status` is Failure, immediately skip to the error step (`EE_handle_github_release_error.md`).
3.  **Check Condition:** Examine the `create_github_release` flag within `validated_build_params`. If `false`, set `github_release_status` to Skipped and proceed to the `next_step`.
4.  **Prepare Release Data:** Extract the tag name (version) from `validated_build_params`. Use the `release_notes_markdown` as the body. Identify the `packaged_artifact_path` as the asset to upload.
5.  **Create Release:** Use appropriate tools (e.g., GitHub CLI `gh release create`, GitHub API via MCP) to create a new release associated with the tag.
    *   Set the release title (e.g., "Release vX.Y.Z").
    *   Use the markdown notes as the body.
    *   Upload the `packaged_artifact_path` as a release asset.
6.  **Set Status:** Set `github_release_status` to Success if the release is created successfully, otherwise Failure.
7.  **Prepare Output:** Provide the `github_release_status` and the `github_release_url` (if successful).

## Acceptance Criteria

*   Step proceeds only if `tag_push_status` was Success.
*   Step correctly skips if `create_github_release` parameter is false.
*   If proceeding, the correct tag name, release notes, and artifact path are used.
*   A GitHub release is successfully created with the specified details and asset.
*   `github_release_status` is set accurately (Success, Failure, or Skipped).
*   `github_release_url` is populated on success.

## Error Handling

*   If prerequisite `tag_push_status` is Failure, proceed directly to `EE_handle_github_release_error.md`.
*   If `create_github_release` is true and the release creation process fails (e.g., API error, upload failure), set `github_release_status` to Failure and proceed to `EE_handle_github_release_error.md`.