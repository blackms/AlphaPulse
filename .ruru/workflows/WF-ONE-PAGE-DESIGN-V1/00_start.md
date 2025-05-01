+++
# --- Step Metadata ---
step_id = "WF-ONE-PAGE-DESIGN-V1-00-START"
title = "Step 00: Start One Page Design Workflow"
description = """
Confirms the user request for a one-page design, gathers initial details,
and prepares the context (design brief) for delegation to the design specialist.
"""

# --- Flow Control ---
depends_on = []
next_step = "01_generate_design.md"
error_step = "EE_workflow_error.md" # Standard error handler step

# --- Execution ---
delegate_to = "" # Orchestrator handles this step: context gathering and validation.

# --- Interface ---
inputs = [
    "Workflow Input: User request detailing the desired one-page website (e.g., purpose, style, content sections, target audience)."
]
outputs = [
    "design_brief: Formatted user request ready for the design specialist.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}"
template_schema_doc = ".ruru/templates/toml-md/24_workflow_step_start.md"
+++

# Step 00: Start One Page Design Workflow

## Actions

1.  **Receive User Request:** Obtain the detailed request for the one-page website from the workflow input.
2.  **Validate Request:** Check if the request contains minimal required details for the `design_brief`. Essential fields typically include: purpose, target audience, desired style/aesthetic, and key content sections. If essential details are missing, proceed to `error_step`. (Note: A more robust workflow might include a clarification step).
3.  **Format Design Brief:** Structure the validated user request into a clear `design_brief` suitable for the `design-one-shot` mode.
4.  **Prepare for Next Step:** Ensure the `design_brief` is available as output for the next step.

## Acceptance Criteria

*   User request is received and validated for basic completeness.
*   A `design_brief` containing the core requirements is formatted.
*   The workflow is ready to proceed to step `01_generate_design.md`.

## Error Handling

*   If the initial user request is missing critical details (validation fails), proceed to `{{error_step}}`.