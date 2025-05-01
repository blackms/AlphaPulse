+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-06-DETERMINE-PREV-TAG" # (String, Required) Unique ID for this step.
title = "Step 06: Determine Previous Git Tag" # (String, Required) Title of this specific step.
description = """
(String, Required) Determines the most recent Git tag matching the release pattern
(e.g., v*.*.*) to establish the starting point for generating release notes.
"""

# --- Flow Control ---
depends_on = ["WF-CREATE-ROO-CMD-BUILD-V1-05-PACKAGE-ARTIFACTS"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "07_query_git_history.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_git_tag_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "dev-git" # (String, Optional) Mode responsible for executing the core logic of this step.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-05-PACKAGE-ARTIFACTS: packaging_status", # Prerequisite check
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-01-VALIDATE-PARAMS: validated_build_params", # May contain tag pattern info
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "previous_git_tag: The most recent relevant Git tag found (e.g., v1.2.3).",
    "determine_tag_status: Success or Failure.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 06: Determine Previous Git Tag

## Actions

1.  **Receive Context:** Get `packaging_status` and potentially `validated_build_params` from previous steps.
2.  **Check Prerequisite:** If `packaging_status` is Failure, immediately skip to the error step (`EE_handle_git_tag_error.md`).
3.  **Fetch Tags:** Ensure local repository has up-to-date tags (`git fetch --tags`).
4.  **Identify Previous Tag:** Use Git commands (e.g., `git describe --tags --abbrev=0 --match "v*.*.*"`) to find the most recent tag matching the release version pattern. Handle cases where no previous tag exists (might use first commit or a default).
5.  **Set Status:** Set `determine_tag_status` to Success if a tag (or a suitable starting point) is found, otherwise Failure.
6.  **Prepare Output:** Provide the identified `previous_git_tag` and the `determine_tag_status`.

## Acceptance Criteria

*   Step proceeds only if `packaging_status` was Success.
*   Git tags are fetched.
*   The most recent relevant Git tag is correctly identified or a suitable alternative starting point is determined.
*   `previous_git_tag` is populated.
*   `determine_tag_status` is set accurately.

## Error Handling

*   If prerequisite `packaging_status` is Failure, proceed directly to `EE_handle_git_tag_error.md`.
*   If Git commands fail or no suitable previous tag/starting point can be determined, set `determine_tag_status` to Failure and proceed to `EE_handle_git_tag_error.md`.