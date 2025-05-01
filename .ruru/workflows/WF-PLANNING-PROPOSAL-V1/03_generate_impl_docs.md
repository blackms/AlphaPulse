+++
# --- Step Metadata ---
step_id = "WF-PLANNING-PROPOSAL-V1-03-GENERATE-IMPL-DOCS" # (String, Required) Unique ID for this step.
title = "Step 03: Generate Implementation Documents" # (String, Required) Title of this specific step.
description = """
Generate documents related to the practical implementation of the proposal (e.g., Implementation Plan, Concerns Analysis).
The delegate (`project-manager` or `core-architect`) reviews the context and saves the generated documents.
"""

# --- Flow Control ---
depends_on = ["WF-PLANNING-PROPOSAL-V1-02-GENERATE-WHITEPAPER"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "99_finish.md" # (String, Optional) Filename of the next step on successful completion. This is the last main step before finish.
error_step = "99_finish.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "project-manager" # (String, Optional) Mode responsible for executing the core logic. Can be overridden (e.g., to core-architect).

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Can reference outputs from 'depends_on' steps.
    "Output from step WF-PLANNING-PROPOSAL-V1-00-START: proposal_base_path",
    "Output from step WF-PLANNING-PROPOSAL-V1-01-REFINE: refinement_notes_path",
    "Output from step WF-PLANNING-PROPOSAL-V1-02-GENERATE-WHITEPAPER: whitepaper_path",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "implementation_docs_paths: List of paths to the generated implementation documents (e.g., Implementation_Plan.md, Concerns_Analysis.md).",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 03: Generate Implementation Documents

## Actions

(Instructions for the delegate: `{{delegate_to}}`)

1.  **Review Context:**
    *   Review the refinement notes at `{{inputs[1]}}` (refinement_notes_path).
    *   Review the whitepaper at `{{inputs[2]}}` (whitepaper_path).
    *   Both files are located within `{{inputs[0]}}` (proposal_base_path).
2.  **Generate Documents:**
    *   Based on the reviewed context, generate the **mandatory** implementation documents:
        *   `Implementation_Plan.md`: High-level steps, phases, resources.
        *   `Concerns_Analysis.md`: Risks, challenges, open questions, mitigations.
    *   *(Optional)* Generate additional relevant documents based on the proposal complexity, such as:
        *   `Improvements_Suggestions.md`: Future enhancement ideas.
        *   `Data_Model.md`: Initial data schema ideas.
        *   `API_Design.md`: Preliminary API endpoints.
    *   Focus on actionable insights for planning and execution.
3.  **Save Documents:**
    *   Save each generated document as a separate Markdown file within `{{inputs[0]}}` (proposal_base_path) using `write_to_file`. *(Confirmation mechanism, e.g., orchestrator prompt or delegate capability like `confirm_write`, is assumed before execution).*
    *   Keep track of the paths of the created files.
4.  **Report Completion:**
    *   Use `attempt_completion` to report success back to the orchestrator, providing the list of `implementation_docs_paths`.
    *   *(Review Point: Orchestrator may optionally pause here to allow user review of the generated implementation documents before proceeding to Step 99).*

## Acceptance Criteria

*   Refinement notes and whitepaper have been reviewed.
*   Mandatory implementation documents (`Implementation_Plan.md`, `Concerns_Analysis.md`) have been generated.
*   Generated documents are saved within `{{inputs[0]}}`.
*   A list of paths (`implementation_docs_paths`) to the created documents is provided as output.

## Error Handling

*   If reading context files fails, proceed to `{{error_step}}`.
*   If document generation fails, proceed to `{{error_step}}`.
*   If saving any document fails, proceed to `{{error_step}}`.