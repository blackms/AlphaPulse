+++
# --- Step Metadata ---
step_id = "WF-ONE-PAGE-DESIGN-V1-99-FINISH"
step_number = 99
title = "Step 99: Finish One Page Design Workflow"
description = """
Finalizes the workflow by aggregating the results (generated file paths, review status),
performing any necessary cleanup, and reporting the final outcome.
"""

# --- Flow Control ---
depends_on = ["WF-ONE-PAGE-DESIGN-V1-02-REVIEW-DESIGN"] # Depends on the review step
next_step = "" # Final step
error_step = "EE_workflow_error.md" # Standard error handler step (though unlikely to be triggered directly from finish)

# --- Execution ---
delegate_to = "" # Orchestrator handles final reporting

# --- Interface ---
inputs = [
    "Output from step WF-ONE-PAGE-DESIGN-V1-01-GENERATE-DESIGN: generated_files",
    "Output from step WF-ONE-PAGE-DESIGN-V1-01-GENERATE-DESIGN: design_summary",
    "Output from step WF-ONE-PAGE-DESIGN-V1-02-REVIEW-DESIGN: review_status",
    "Output from step WF-ONE-PAGE-DESIGN-V1-02-REVIEW-DESIGN: review_comments", # Optional
]
outputs = [
    "workflow_result: Summary including paths to generated files and review status.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}"
template_schema_doc = ".ruru/templates/toml-md/26_workflow_step_finish.md"
+++

# Step 99: Finish One Page Design Workflow

## Actions

1.  **Aggregate Results:** Collect `generated_files`, `design_summary`, `review_status`, and optional `review_comments` from previous steps.
2.  **Perform Cleanup:** (No specific cleanup defined for this simple workflow, but could include deleting temporary artifacts if any were created).
3.  **Generate Final Report:** Prepare a summary message for the user.
    *   Include the paths to the `generated_files`.
    *   Mention the `review_status`.
    *   Include `review_comments` if the status was "Needs Revision".
4.  **Determine Final Status:** Workflow is successful if `generated_files` were produced. The `review_status` indicates the quality outcome.
5.  **Report Outcome:** Format the summary message as the `workflow_result` output.

## Acceptance Criteria

*   All required inputs from previous steps are available.
*   A final `workflow_result` summary is generated, including file paths and review status.

## Error Handling

*   If essential inputs (like `generated_files`) are missing, the workflow should report a failure state in the `workflow_result`.