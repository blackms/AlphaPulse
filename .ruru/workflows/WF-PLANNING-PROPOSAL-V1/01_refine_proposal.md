+++
# --- Step Metadata ---
step_id = "WF-PLANNING-PROPOSAL-V1-01-REFINE" # (String, Required) Unique ID for this step.
title = "Step 01: Refine Proposal" # (String, Required) Title of this specific step.
description = """
Interactively refine the initial proposal idea with the user to clarify goals, scope, requirements, and potential challenges.
The delegate (ask or core-architect) reviews initial input, engages the user via `ask_followup_question`, summarizes the discussion, and saves it to `Refinement_Notes.md`.
"""

# --- Flow Control ---
depends_on = ["WF-PLANNING-PROPOSAL-V1-00-START"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "02_generate_whitepaper.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "99_finish.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "ask" # (String, Optional) Mode responsible for executing the core logic. Can be overridden (e.g., to core-architect).

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step {{depends_on[0]}}: proposal_base_path",
    "Output from step {{depends_on[0]}}: input_path",
    "Output from step {{depends_on[0]}}: Initial input files within input_path",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "refinement_notes_path: Path to the saved Refinement_Notes.md file.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 01: Refine Proposal

## Actions

(Instructions for the delegate: `{{delegate_to}}`)

1.  **Review Input:**
    *   Read the initial request and any input files located at `{{inputs[1]}}` (input_path).
2.  **Engage User:**
    *   Use the `ask_followup_question` tool to interactively discuss the proposal with the user.
    *   Focus on clarifying:
        *   Goals
        *   Scope (Inclusions/Exclusions)
        *   Key Features/Requirements
        *   Potential Challenges/Risks
        *   Success Metrics
        *   Non-Goals
    *   *(Interaction Note: Continue dialogue until clarifications are sufficient or user indicates completion. Consider a timeout (e.g., 2 attempts or 5 minutes) if user becomes unresponsive).*
3.  **Summarize:**
    *   Consolidate the key discussion points, decisions, and clarifications into a concise summary.
    *   *(Optional: Consider a brief confirmation loop with the user on the summary itself before saving, if feasible within the delegate's capabilities).*
4.  **Save Notes:**
    *   Define `refinement_notes_path = "{{inputs[0]}}" + "/Refinement_Notes.md"` (proposal_base_path + "/Refinement_Notes.md").
    *   Use `write_to_file` to save the summary to `refinement_notes_path`. *(Confirmation mechanism, e.g., orchestrator prompt or delegate capability like `confirm_write`, is assumed before execution).*
5.  **Report Completion:**
    *   Use `attempt_completion` to report success back to the orchestrator, providing the `refinement_notes_path`.
    *   *(Review Point: Orchestrator may optionally pause here to allow user review of `Refinement_Notes.md` before proceeding to Step 02).*

## Acceptance Criteria

*   User interaction via `ask_followup_question` has occurred.
*   A summary of the refinement discussion exists.
*   The summary is saved to `{{inputs[0]}}/Refinement_Notes.md`.
*   The `refinement_notes_path` is provided as output.

## Error Handling

*   If the user is unresponsive during the refinement dialogue (e.g., after 2 attempts or 5 minutes without response), log the issue and proceed to `{{error_step}}`.
*   If saving the `Refinement_Notes.md` file fails, proceed to `{{error_step}}`.