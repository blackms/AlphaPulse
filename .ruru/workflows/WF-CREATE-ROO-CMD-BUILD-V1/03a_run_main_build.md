+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-03A-RUN-MAIN-BUILD"
title = "Step 03a: Run Main Build Script"
description = """
(String, Required) Executes the main build script (`create_build.js`) for Roo Commander
within the prepared environment. Captures logs and checks for success.
"""

# --- Flow Control ---
depends_on = ["WF-CREATE-ROO-CMD-BUILD-V1-02-SETUP-ENVIRONMENT"]
next_step = "03b_run_kilocode_build.md"
error_step = "EE_handle_build_error.md"

# --- Execution ---
delegate_to = "lead-devops"

# --- Interface ---
inputs = [
    "Output from step WF-CREATE-ROO-CMD-BUILD-V1-01-VALIDATE-PARAMS: validated_build_params",
    # build_directory_path from 02 is implicitly the current working directory
]
outputs = [
    "main_build_log: Path to the build output log for create_build.js (build_main.log).",
    "main_build_status: Success or Failure.",
    # Pass through validated_build_params for subsequent steps
    "validated_build_params: The validated build parameters object.",
]

# --- Housekeeping ---
last_updated = "2025-05-01" # Updated date
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md"
+++

# Step 03a: Run Main Build Script

## Actions

1.  **Receive Context:** Get `validated_build_params` from step 01.
2.  **Create Temp Dir:** Ensure `.tmp` directory exists (`mkdir -p .tmp`).
3.  **Execute Build Script (`create_build.js`):** Run `node .ruru/scripts/create_build.js [version] [codename] [readmePath] [changelogPath] > build_main.log`. Check exit code.
    *   Arguments derived from `validated_build_params`:
        *   `version`: e.g., `v7.1.5`
        *   `codename`: e.g., `Wallaby`
        *   `readmePath`: `.ruru/templates/build/README.dist.md`
        *   `changelogPath`: `.tmp/CHANGELOG.[version].md` (e.g., `.tmp/CHANGELOG.v7.1.5.md`)
4.  **Prepare Output:** Provide the path to `build_main.log` (`build_main.log`) and the `main_build_status`. Pass through `validated_build_params`.

## Acceptance Criteria

*   `.tmp` directory exists.
*   `create_build.js` script is executed with correct arguments derived from `validated_build_params`.
*   Script completes and exit code is checked.
*   `main_build_status` is set correctly (Success/Failure).
*   `main_build_log` output is `"build_main.log"`.
*   `validated_build_params` is passed through.

## Error Handling

*   If the `mkdir` or `node` command fails (non-zero exit code), set `main_build_status` to Failure and proceed to `EE_handle_build_error.md`.