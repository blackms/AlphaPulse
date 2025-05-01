+++
# --- Step Metadata ---
step_id = "WF-PLANNING-PROPOSAL-V1-02-GENERATE-WHITEPAPER" # (String, Required) Unique ID for this step.
title = "Step 02: Generate Whitepaper" # (String, Required) Title of this specific step.
description = """
Generate a formal whitepaper summarizing the refined proposal based on the refinement notes and initial input.
The delegate (`technical-writer`) reads the context files and saves the generated whitepaper.
"""

# --- Flow Control ---
depends_on = ["WF-PLANNING-PROPOSAL-V1-01-REFINE"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "03_generate_impl_docs.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "99_finish.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "technical-writer" # (String, Optional) Mode responsible for executing the core logic of this step.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-PLANNING-PROPOSAL-V1-00-START: proposal_base_path",
    "Output from step WF-PLANNING-PROPOSAL-V1-00-START: proposal_name",
    "Output from step WF-PLANNING-PROPOSAL-V1-00-START: input_path",
    "Output from step WF-PLANNING-PROPOSAL-V1-01-REFINE: refinement_notes_path",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "whitepaper_path: Path to the generated [ProposalName]_Whitepaper.md file.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 02: Generate Whitepaper

## Actions

(Instructions for the delegate: `{{delegate_to}}`)

1.  **Review Context:**
    *   Read the refinement notes file located at `{{inputs[3]}}` (refinement_notes_path).
    *   Review the initial input files located within `{{inputs[2]}}` (input_path) for additional context.
2.  **Generate Whitepaper:**
    *   Compose a comprehensive whitepaper document summarizing the refined proposal.
    *   Structure it logically (e.g., Introduction, Problem Statement, Proposed Solution, Scope, Key Features, Potential Benefits, Conclusion).
    *   Ensure the content adheres to standard documentation practices.
3.  **Save Whitepaper:**
    *   Define `whitepaper_path = "{{inputs[0]}}" + "/" + "{{inputs[1]}}" + "_Whitepaper.md"` (proposal_base_path + "/" + proposal_name + "_Whitepaper.md").
    *   Use `write_to_file` to save the generated whitepaper to `whitepaper_path`. *(Confirmation mechanism, e.g., orchestrator prompt or delegate capability like `confirm_write`, is assumed before execution).*
4.  **Report Completion:**
    *   Use `attempt_completion` to report success back to the orchestrator, providing the `whitepaper_path`.
    *   *(Review Point: Orchestrator may optionally pause here to allow user review of the generated `whitepaper_path` before proceeding to Step 03).*

## Acceptance Criteria

*   The refinement notes and initial input have been reviewed.
*   A comprehensive whitepaper document has been generated.
*   The whitepaper is saved to `{{inputs[0]}}/{{inputs[1]}}_Whitepaper.md`.
*   The `whitepaper_path` is provided as output.

## Error Handling

*   If reading context files fails, proceed to `{{error_step}}`.
*   If whitepaper generation fails (e.g., delegate error), proceed to `{{error_step}}`.
*   If saving the whitepaper file fails, proceed to `{{error_step}}`.