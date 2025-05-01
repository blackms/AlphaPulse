+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-ENRICHMENT-V2-07-GENERATE_KB_README" # (String, Required) Unique ID for this step
title = "Step 07: Generate/Update KB README" # (String, Required) Title of this specific step.
description = """
Delegates to the Mode Structure Agent (e.g., mode-maintainer or technical-writer) to create or update the mode's main KB README file (`kb/README.md`). Ensures the README includes an overview, lists indexed KB directories (including `context7`), and specifically mentions the Context7 enrichment, linking to its `_index.json`.
"""

# --- Flow Control ---
depends_on = ["WF-CONTEXT7-ENRICHMENT-V2-06-UPDATE_MODE_DEFINITION"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "08_quality_assurance.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "mode-maintainer" # (String, Optional) Mode responsible for updating the README (could also be technical-writer).

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-00-START: mode_slug",
    "Output from step WF-CONTEXT7-ENRICHMENT-V2-02-EXECUTE_SCRIPT: context7_output_dir", # Path to context7 dir
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "kb_readme_path: Path to the updated KB README file.",
    "kb_readme_update_status: Confirmation that the README was updated.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 07: Generate/Update KB README

## Actions

1.  **Delegate README Update:**
    *   Instruct the `mode-maintainer` (or `technical-writer`) delegate to update the KB README.
    *   Provide the `[mode_slug]`.
    *   Instruct the agent to:
        *   Define the path: `[kb_readme_path] = .ruru/modes/[mode_slug]/kb/README.md`.
        *   Read the existing `[kb_readme_path]` if it exists (Prefer MCP `read_file_content`, fallback `read_file`). Handle not found by preparing to create a new one.
        *   *(Optional but recommended: Read `.ruru/modes/[mode_slug]/kb/index.toml` to get a list of all indexed KB directories).*
        *   Construct the updated README content:
            *   Include a general overview section for the mode's KB.
            *   Include a section listing the main KB subdirectories (like `context7`, and any others found from `index.toml` or directory listing), with brief descriptions and links to their respective index files (`_index.json` for `context7`, `index.toml` for others).
            *   Ensure the `context7` entry specifically mentions it contains content processed from the Context7 source and links to `kb/context7/_index.json`.
        *   Write the complete, updated content to `[kb_readme_path]` (Prefer MCP `write_file_content`, fallback `write_to_file`). Handle write errors -> **Log Warning/Non-critical Failure**.
2.  **Receive Confirmation:** Await confirmation of success or failure from the delegate. Store the `[kb_readme_path]`.

## Acceptance Criteria

*   Delegation to the agent was successful.
*   The KB README file `.ruru/modes/[mode_slug]/kb/README.md` exists and includes updated information referencing the `context7` KB directory and its `_index.json`.
*   `kb_readme_update_status` indicates success.

## Error Handling

*   If delegation fails, log the error and proceed to `{{error_step}}`.
*   If reading existing files (README, index.toml) fails, log the error, generate a basic README if possible, and log a warning.
*   If writing the README fails, log the error and proceed to the next step (non-critical failure).