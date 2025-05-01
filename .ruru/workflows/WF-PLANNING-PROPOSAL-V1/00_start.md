+++
# --- Step Metadata ---
step_id = "WF-PLANNING-PROPOSAL-V1-00-START" # (String, Required) Unique ID for this step.
title = "Step 00: Initiation & Input Capture" # (String, Required) Title of this specific step.
description = """
Receive the user's request and initial input. Determine a suitable, filesystem-safe name for the proposal (`[ProposalName]`).
Create the necessary directory structure (`.ruru/planning/[ProposalName]/input/`) and save the initial input (text and/or files).
"""

# --- Flow Control ---
depends_on = [] # (Array of Strings, Required) Always empty for the start step.
next_step = "01_refine_proposal.md" # (String, Required) Filename of the next step on successful completion.
error_step = "99_finish.md" # (String, Optional) Filename to jump to if this step fails (go to finish to report error).

# --- Execution ---
delegate_to = "" # (String, Optional) Handled by the workflow orchestrator (e.g., prime-coordinator).

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Usually references overall workflow inputs.
    "Workflow Input: User request text",
    "Workflow Input: User provided file paths (optional)",
    "Workflow Input: workflow_version (e.g., '1.0.0' from README)",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "proposal_name: Filesystem-safe name derived from user input.",
    "proposal_base_path: Path to the main proposal directory (e.g., .ruru/planning/[ProposalName]).",
    "input_path: Path to the input subdirectory (e.g., .ruru/planning/[ProposalName]/input).",
    "Saved initial input files within input_path.",
    "initial_input_artifacts_list: List of paths to the saved initial input files (including initial_request.md if created).",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/24_workflow_step_start.md" # (String, Required) Link to this template definition.
+++

# Step 00: Initiation & Input Capture

## Actions

1.  **Analyze Input & Derive Name:**
    *   Read the user's request text.
    *   Determine a concise, filesystem-safe `proposal_name` by sanitizing the first ~30 characters of the user request title (replace spaces/special characters with underscores, remove non-alphanumeric except underscore). Example: "Feature_UserAuth", "Refactor_DatabaseSchema".
2.  **Define Paths:**
    *   Set `proposal_base_path = ".ruru/planning/" + proposal_name + "_V" + {{inputs[2]}}` # Incorporate workflow_version
    *   Set `input_path = proposal_base_path + "/input"`
3.  **Check Target Directory:**
    *   Verify if `proposal_base_path` already exists. If it does, **fail the step** and proceed to `{{error_step}}` (to prevent overwriting).
4.  **Create Directory:**
    *   If the check passes, use `execute_command` (`mkdir -p`) or equivalent tool to create the `input_path`.
5.  **Save Text Input:**
    *   If text input was provided, save it to `input_path/initial_request.md` using `write_to_file`.
6.  **Copy File Inputs (If Provided):**
    *   Verify the existence of all user-provided file paths. If any file is missing, **fail the step** and proceed to `{{error_step}}`.
    *   Check for duplicate filenames within the provided list. If duplicates exist, **fail the step** and proceed to `{{error_step}}` (to avoid ambiguity/overwriting during copy).
    *   Copy the verified files into the `input_path` directory (may require `execute_command` with `cp` or similar, depending on tool capabilities).

## Acceptance Criteria

*   `proposal_name`, `proposal_base_path`, and `input_path` variables are defined.
*   The directory structure `.ruru/planning/[ProposalName]/input/` exists.
*   Initial user input (text and/or files) is saved within the `input_path`.
*   All required outputs for the next step (`{{next_step}}`) are ready (proposal_name, proposal_base_path, input_path).

## Error Handling

*   If `proposal_base_path` already exists (Action 3), proceed to `{{error_step}}` and report the error (Potential Overwrite).
*   If directory creation fails (Action 4), proceed to `{{error_step}}` and report the error (Directory Creation Failed).
*   If any user-provided input file is missing (Action 6), proceed to `{{error_step}}` and report the error (Missing Input File).
*   If duplicate filenames are provided in the input list (Action 6), proceed to `{{error_step}}` and report the error (Duplicate Input Filenames).
*   If input files cannot be copied for other reasons (e.g., permissions) (Action 6), proceed to `{{error_step}}` and report the error (File Copy Failed).