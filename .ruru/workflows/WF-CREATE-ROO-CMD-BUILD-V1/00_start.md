+++
# --- Step Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-00-START" # (String, Required) Unique ID for this step.
title = "Step 00: Start Build Workflow" # (String, Required) Title of this specific step.
description = """
(String, Required) Gathers initial build parameters (e.g., target platform) from the workflow input
and prepares the context for the next step. Version is determined in the next step.
"""

# --- Flow Control ---
depends_on = [] # (Array of Strings, Required) Always empty for the start step.
next_step = "00b_suggest_next_version.md" # (String, Required) Filename of the next step on successful completion.
error_step = "EE_handle_start_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "" # (String, Optional) Mode responsible for executing the core logic of this step. Often empty for start.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Usually references overall workflow inputs.
    "Workflow Input: Request detailing the specific build parameters (e.g., target platform). Version is handled later.",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "build_params: Object containing the gathered build parameters (excluding version initially).",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/24_workflow_step_start.md" # (String, Required) Link to this template definition.
+++

# Step 00: Start Build Workflow

## Actions

1.  **Gather Initial Context:** Read the build parameters (excluding version) provided as input to the workflow.
2.  **Validate Preconditions:** Perform a basic check to ensure required parameters (e.g., platform) are present.
3.  **Prepare for Next Step:** Structure the gathered parameters into the `build_params` output.
4.  **(Optional) Delegate First Action:** (Not applicable for this step).

## Acceptance Criteria

*   Build parameters (excluding version) are successfully read from the workflow input.
*   Basic validation confirms the presence of essential parameters (excluding version).
*   Inputs for the next step (`00b_suggest_next_version.md`) are ready (`build_params`).

## Error Handling

*   If essential parameters (excluding version) are missing or invalid, proceed to `EE_handle_start_error.md` (if defined).