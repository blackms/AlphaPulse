+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-03C-RUN-COLLECTION-BUILDS"
title = "Step 03c: Run Collection Build Script"
description = """
(String, Required) Executes the collection build script (`run_collection_builds.js`) for Roo Commander.
Captures logs and checks for success.
"""

# --- Flow Control ---
depends_on = ["WF-CREATE-ROO-CMD-BUILD-V1-03B-RUN-KILOCODE-BUILD"]
next_step = "04_verify_artifacts.md" # Next step after all builds
error_step = "EE_handle_build_error.md"

# --- Execution ---
delegate_to = "lead-devops"

# --- Interface ---
inputs = [
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-03B-RUN-KILOCODE-BUILD: kilocode_build_status",
    # Pass through validated_build_params for subsequent steps
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-03B-RUN-KILOCODE-BUILD: validated_build_params",
]
outputs = [
    "collection_build_log: Path to the build output log for run_collection_builds.js (build_collections.log).",
    "collection_build_status: Success or Failure.",
    # Pass through validated_build_params
    "validated_build_params: The validated build parameters object.",
    # Define the final raw artifacts path here, assuming collections build populates it
    "raw_artifacts_path: Path to the directory containing raw build output (e.g., '.builds/')."
]

# --- Housekeeping ---
last_updated = "2025-05-01"
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md"
+++

# Step 03c: Run Collection Build Script

## Actions

1.  **Receive Context:** Get `kilocode_build_status` and `validated_build_params` from the previous step.
2.  **Check Previous Step:** If `kilocode_build_status` is Failure, skip this step and proceed directly to the error step (`EE_handle_build_error.md`).
3.  **Execute Build Script (`run_collection_builds.js`):** Run `node .ruru/scripts/run_collection_builds.js > build_collections.log`. Check exit code.
4.  **Prepare Output:** Provide the path to `build_collections.log` (`build_collections.log`), the `collection_build_status`, the `raw_artifacts_path` (likely `.builds/`), and pass through `validated_build_params`.

## Acceptance Criteria

*   `kilocode_build_status` is Success.
*   `run_collection_builds.js` script is executed.
*   Script completes and exit code is checked.
*   `collection_build_status` is set correctly (Success/Failure).
*   `collection_build_log` output is `"build_collections.log"`.
*   `raw_artifacts_path` is identified (e.g., `.builds/`).
*   `validated_build_params` is passed through.

## Error Handling

*   If the `node` command fails (non-zero exit code), set `collection_build_status` to Failure and proceed to `EE_handle_build_error.md`.