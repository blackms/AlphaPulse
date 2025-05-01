+++
# --- Step Metadata ---
step_id = "WF-RELEASE-NOTES-HYBRID-V2-05-PUSH_TO_GITHUB" # Unique ID
title = "Step 05: Optional - Push to GitHub Release" # Based on original step 6
description = """
Checks if pushing to GitHub is requested. If yes, delegates to the GitHub MCP server
to create or update a GitHub Release using the generated notes body and provided parameters.
"""

# --- Flow Control ---
depends_on = [
    "WF-RELEASE-NOTES-HYBRID-V2-00-START", # Need push flag, github params, options, target tag
    "WF-RELEASE-NOTES-HYBRID-V2-03-SUMMARIZE_CHANGES" # Need release_notes_body
]
next_step = "99_finish.md" # Proceeds to finish step regardless of push success/failure or skip
error_step = "99_finish.md" # Errors in this optional step also go to finish for reporting

# --- Execution ---
# Delegation happens conditionally within actions.
# The MCP server itself is the target.
delegate_to = "github" # Target the GitHub MCP server

# --- Interface ---
inputs = [
    "Output from step WF-RELEASE-NOTES-HYBRID-V2-00-START: validated_push_flag",
    "Output from step WF-RELEASE-NOTES-HYBRID-V2-00-START: validated_github_params", # { owner, repo }
    "Output from step WF-RELEASE-NOTES-HYBRID-V2-00-START: validated_github_options", # { draft, prerelease }
    "Output from step WF-RELEASE-NOTES-HYBRID-V2-00-START: validated_target_tag",
    "Output from step WF-RELEASE-NOTES-HYBRID-V2-03-SUMMARIZE_CHANGES: release_notes_body"
]
outputs = [
    "github_push_status: A string indicating the outcome ('skipped', 'success', 'failed').",
    "github_release_link: URL to the created/updated GitHub release (if successful)."
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # Placeholder
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md"
+++

# Step 05: Optional - Push to GitHub Release

## Actions

1.  **Check Push Flag:** Evaluate the `validated_push_flag` input.
2.  **Execute Conditional Push:**
    *   **IF `validated_push_flag` is true:**
        *   Log the attempt to push to GitHub.
        *   **Delegate to `github` MCP:** Use the `create_release` tool. (Note: Future versions might need logic to check for existing releases and use an `update_release` tool if available/required). Provide the following parameters:
            *   `owner`: `{{validated_github_params.owner}}`
            *   `repo`: `{{validated_github_params.repo}}`
            *   `tag_name`: `{{validated_target_tag}}`
            *   `name`: "Release {{validated_target_tag}}" (or similar)
            *   `body`: `{{release_notes_body}}`
            *   `draft`: `{{validated_github_options.draft}}`
            *   `prerelease`: `{{validated_github_options.prerelease}}`
        *   **Handle MCP Result:**
            *   If the MCP call is successful, store `github_push_status = "success"` and extract the release URL (if provided by the tool) into `github_release_link`. Log success.
            *   If the MCP call fails, store `github_push_status = "failed"` and `github_release_link = null`. Log the error details received from the MCP. Log failure.
    *   **ELSE (`validated_push_flag` is false):**
        *   Store `github_push_status = "skipped"` and `github_release_link = null`.
        *   Log that the GitHub push was skipped.
3.  **Proceed:** Continue to the `{{next_step}}`.

## Acceptance Criteria

*   The `validated_push_flag` is correctly evaluated.
*   If true, the `github` MCP `create_release` tool is called with the correct parameters.
*   The `github_push_status` output reflects whether the push was skipped, successful, or failed.
*   If successful, `github_release_link` contains the URL.

## Error Handling

*   Errors during the MCP call should set `github_push_status` to "failed" and be logged, but the workflow should still proceed to the `{{error_step}}` (which is `99_finish.md`) for final reporting.