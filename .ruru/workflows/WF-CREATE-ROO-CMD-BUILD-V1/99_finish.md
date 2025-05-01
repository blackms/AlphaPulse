+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-99-FINISH" # (String, Required) Unique ID for this step.
# Convention: The final step in a workflow should always use step_number = 99
step_number = 99
title = "Step 99: Finish Build Workflow" # (String, Required) Title of this specific step.
description = """
(String, Required) Finalizes the build and release workflow. Aggregates results
from all previous steps, performs cleanup, and reports the final status,
artifact location, release notes path, commit hash, and GitHub release URL.
"""

# --- Flow Control ---
depends_on = ["WF-CREATE-ROO-CMD-BUILD-V1-14-CREATE-GITHUB-RELEASE"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "" # (String, Required) Always empty for the finish step.
error_step = "EE_handle_finish_error.md" # (String, Optional) Filename to jump to if finalization fails.

# --- Execution ---
delegate_to = "lead-devops" # (String, Optional) Mode responsible for executing the core logic of this step.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed from previous steps.
    # Build & Package
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-03-RUN-BUILDS: build_status",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-04-VERIFY-ARTIFACTS: artifacts_verified_status",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-05-PACKAGE-ARTIFACTS: packaging_status",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-05-PACKAGE-ARTIFACTS: packaged_artifact_path",
    # Release Notes & Docs
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-06-DETERMINE-PREV-TAG: determine_tag_status",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-07-QUERY-GIT-HISTORY: query_history_status",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-08-GENERATE-RELEASE-NOTES: generate_notes_status",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-09-SAVE-RELEASE-NOTES: save_notes_status",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-09-SAVE-RELEASE-NOTES: release_notes_file_path",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-10-UPDATE-README: readme_update_status",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-11-UPDATE-CHANGELOG: changelog_update_status",
    # Git & GitHub
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-12-COMMIT-DOCS: commit_status",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-12-COMMIT-DOCS: commit_hash",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-13-PUSH-TAG: tag_push_status",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-14-CREATE-GITHUB-RELEASE: github_release_status",
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-14-CREATE-GITHUB-RELEASE: github_release_url",
    # Potentially needed for cleanup
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-03-RUN-BUILDS: raw_artifacts_path",
]
outputs = [ # (Array of Strings, Optional) Final workflow outputs.
    "workflow_result: Summary object, e.g., { overall_status: 'Success'|'Failure', build_status: string, verify_status: string, package_status: string, docs_commit_status: string, tag_push_status: string, github_release_status: string, artifact_path: string, release_notes_path: string, commit_hash: string, github_release_url: string }.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/26_workflow_step_finish.md" # (String, Required) Link to this template definition.
+++

# Step 99: Finish Build Workflow

## Actions

1.  **Aggregate Results:** Collect all status flags (`build_status`, `artifacts_verified_status`, `packaging_status`, `determine_tag_status`, `query_history_status`, `generate_notes_status`, `save_notes_status`, `readme_update_status`, `changelog_update_status`, `commit_status`, `tag_push_status`, `github_release_status`) and relevant data (`packaged_artifact_path`, `release_notes_file_path`, `commit_hash`, `github_release_url`, `raw_artifacts_path`) from all preceding steps.
2.  **Perform Cleanup:** Remove temporary build directories or files if necessary (e.g., `raw_artifacts_path`).
3.  **Determine Overall Status:** Calculate the final `overall_status`. Set to 'Success' only if all critical preceding status flags indicate success (e.g., build, verify, package, commit, tag push). 'Failure' otherwise. Note that `github_release_status` might be 'Skipped', which typically doesn't constitute overall failure.
4.  **Generate Final Report:** Create the comprehensive `workflow_result` object containing the `overall_status` and key details like artifact path, release notes path, commit hash, GitHub release URL (if applicable), and individual step statuses.
5.  **Report Outcome:** Provide the final `workflow_result` output.

## Acceptance Criteria

*   All required outputs and status flags from previous steps are aggregated.
*   Cleanup actions (if any) are completed.
*   The `overall_status` is correctly determined based on critical step outcomes.
*   A final summary object (`workflow_result`) is generated containing the overall status and relevant details/paths.

## Error Handling

*   If aggregation or cleanup fails, proceed to `EE_handle_finish_error.md` (if defined), setting `overall_status` to 'Failure'.