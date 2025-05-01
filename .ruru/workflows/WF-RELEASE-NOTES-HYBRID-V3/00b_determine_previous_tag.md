+++
# --- Step Metadata ---
step_id = "WF-RELEASE-NOTES-HYBRID-V2-00B-DETERMINE_PREVIOUS_TAG" # Unique ID
title = "Step 00b: Determine Previous Tag (If Needed)"
description = """
Determines the Git tag to use for comparison if 'previous_tag' was not provided
in the workflow inputs. Delegates to 'dev-git' to find the latest tag before the target tag.
"""

# --- Flow Control ---
depends_on = ["WF-RELEASE-NOTES-HYBRID-V2-00-START"] # Depends on initial validation
next_step = "01_query_git_history.md" # Leads into the git log query
error_step = "99_finish.md" # Default error handling

# --- Execution ---
delegate_to = "dev-git" # Delegate to Git specialist

# --- Interface ---
inputs = [
    "Output from step WF-RELEASE-NOTES-HYBRID-V2-00-START: determined_previous_tag", # The potentially null/indicator value
    "Output from step WF-RELEASE-NOTES-HYBRID-V2-00-START: validated_target_tag"
]
outputs = [
    "resolved_previous_tag: The actual previous tag string to use for comparison."
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # Placeholder
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md"
+++

# Step 00b: Determine Previous Tag (If Needed)

## Actions

1.  **Check Input `determined_previous_tag`:**
    *   If the input `determined_previous_tag` from Step 00 contains a valid tag string, pass it through directly as `resolved_previous_tag`.
    *   If the input `determined_previous_tag` is null or indicates determination is needed:
        *   **Delegate to `dev-git`:** Instruct `dev-git` to find the latest tag before `{{validated_target_tag}}`. A common command for this is `git describe --tags --abbrev=0 {{validated_target_tag}}^`. (Note: Verify this command with `dev-git` capabilities).
        *   Store the tag returned by `dev-git` as `resolved_previous_tag`.
2.  **Output Resolved Tag:** Ensure `resolved_previous_tag` contains a valid tag string.

## Acceptance Criteria

*   The `resolved_previous_tag` output contains a valid Git tag string suitable for use in `git log`.

## Error Handling

*   If `determined_previous_tag` requires determination and `dev-git` fails to find a previous tag (e.g., it's the first tag), capture the error and proceed to `{{error_step}}`.