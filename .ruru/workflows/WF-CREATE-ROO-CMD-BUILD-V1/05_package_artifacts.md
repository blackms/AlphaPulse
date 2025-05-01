+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-05-PACKAGE-ARTIFACTS" # (String, Required) Unique ID for this step.
title = "Step 05: Package Build Artifacts" # (String, Required) Title of this specific step.
description = """
(String, Required) Packages the raw build artifacts generated in the previous step
into a distributable format (e.g., zip, tar.gz, installer).
"""

# --- Flow Control ---
depends_on = ["WF-CREATE-ROO-CMD-BUILD-V1-04-VERIFY-ARTIFACTS"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "06_determine_prev_tag.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_packaging_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "lead-devops" # (String, Optional) Mode responsible for executing the core logic of this step.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-03-RUN-BUILDS: raw_artifacts_path", # From the build step
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-01-VALIDATE-PARAMS: validated_build_params", # Needed for naming conventions etc.
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-04-VERIFY-ARTIFACTS: artifacts_verified_status", # Prerequisite check
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "packaged_artifact_path: Path to the final packaged artifact.",
    "packaging_log: Path to the packaging log file.",
    "packaging_status: Success or Failure.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 05: Package Build Artifacts

## Actions

1.  **Receive Context:** Get `raw_artifacts_path`, `validated_build_params`, and `artifacts_verified_status` from previous steps.
2.  **Check Prerequisite:** If `artifacts_verified_status` is Failure, immediately skip to the error step (`EE_handle_packaging_error.md`).
3.  **Create Package:** Use appropriate tools (e.g., `zip`, `tar`, installer creator) to package the contents of `raw_artifacts_path`. Use version/platform info from `validated_build_params` for naming.
4.  **Log Packaging:** Capture the output of the packaging process to a log file.
5.  **Verify Package:** Perform a basic check to ensure the package file exists and is not empty. Set `packaging_status` accordingly.
6.  **Prepare Output:** Provide the path to the final `packaged_artifact_path`, the `packaging_log`, and the `packaging_status`.

## Acceptance Criteria

*   Step proceeds only if `artifacts_verified_status` was Success.
*   Raw artifacts are correctly packaged based on parameters.
*   Packaging process is logged.
*   Final package file exists and is accessible.
*   `packaged_artifact_path`, `packaging_log`, and `packaging_status` are provided.

## Error Handling

*   If prerequisite `artifacts_verified_status` is Failure, proceed directly to `EE_handle_packaging_error.md`.
*   If packaging commands fail or the final package is invalid, set `packaging_status` to Failure and proceed to `EE_handle_packaging_error.md`.