+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-03B-RUN-KILOCODE-BUILD"
title = "Step 03b: Run Kilocode Build Script"
description = """
(String, Required) Executes the kilocode build script (`create_kilocode_build.js`) for Roo Commander.
Captures logs and checks for success.
"""

# --- Flow Control ---
depends_on = ["WF-CREATE-ROO-CMD-BUILD-V1-03A-RUN-MAIN-BUILD"]
next_step = "03c_run_collection_builds.md"
error_step = "EE_handle_build_error.md"

# --- Execution ---
delegate_to = "lead-devops"

# --- Interface ---
inputs = [
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-03A-RUN-MAIN-BUILD: main_build_status",
    # Pass through validated_build_params for subsequent steps
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-03A-RUN-MAIN-BUILD: validated_build_params",
]
outputs = [
    "kilocode_build_log: Path to the build output log for create_kilocode_build.js (build_kilocode.log).",
    "kilocode_build_status: Success or Failure.",
    # Pass through validated_build_params
    "validated_build_params: The validated build parameters object.",
]

# --- Housekeeping ---
last_updated = "2025-05-01"
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md"
+++

# Step 03b: Run Kilocode Build Script

## Actions

1.  **Receive Context:** Get `main_build_status` and `validated_build_params` from the previous step.
2.  **Check Previous Step:** If `main_build_status` is Failure, skip this step and proceed directly to the error step (`EE_handle_build_error.md`).
3.  **Execute Build Script (`create_kilocode_build.js`):** Run `node .ruru/scripts/create_kilocode_build.js > build_kilocode.log`. Check exit code.
4.  **Prepare Output:** Provide the path to `build_kilocode.log` (`build_kilocode.log`) and the `kilocode_build_status`. Pass through `validated_build_params`.

## Acceptance Criteria

*   `main_build_status` is Success.
*   `create_kilocode_build.js` script is executed.
*   Script completes and exit code is checked.
*   `kilocode_build_status` is set correctly (Success/Failure).
*   `kilocode_build_log` output is `"build_kilocode.log"`.
*   `validated_build_params` is passed through.

## Error Handling

*   If the `node` command fails (non-zero exit code), set `kilocode_build_status` to Failure and proceed to `EE_handle_build_error.md`.