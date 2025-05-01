+++
# --- Step Metadata ---
step_id = "WF-ONE-PAGE-DESIGN-V1-02-REVIEW-DESIGN"
title = "Step 02: Review Generated Design (Optional)"
description = """
Reviews the generated HTML/CSS/JS files for correctness, adherence to the brief,
and overall quality. This step can involve manual review (pausing workflow) or
delegation to a reviewer mode (e.g., `util-reviewer`). Determines if revisions are needed.
"""

# --- Flow Control ---
depends_on = ["WF-ONE-PAGE-DESIGN-V1-01-GENERATE-DESIGN"]
# next_step = "99_finish.md" # Replaced by conditional logic below
conditional_next_steps = [
    { condition = "outputs.review_status == 'Approved'", next_step = "99_finish.md" },
    { condition = "outputs.review_status == 'Needs Revision'", next_step = "01_generate_design.md" } # Loop back for revision
]
error_step = "EE_workflow_error.md" # Standard error handler step

# --- Execution ---
delegate_to = "" # Optional: Could delegate to 'util-reviewer' or similar

# --- Interface ---
inputs = [
    "Output from step WF-ONE-PAGE-DESIGN-V1-01-GENERATE-DESIGN: generated_files",
    "Output from step WF-ONE-PAGE-DESIGN-V1-01-GENERATE-DESIGN: design_summary",
    "Output from step WF-ONE-PAGE-DESIGN-V1-00-START: design_brief", # For comparison
    "Input: review_comments (Optional, from previous revision loop or manual input)", # Clarify potential input source
]
outputs = [
    "review_status: Approved | Needs Revision",
    "review_comments: (Optional) Feedback if revisions are needed.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}"
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md"
+++

# Step 02: Review Generated Design (Optional)

## Actions

1.  **Receive Generated Files:** Obtain the `generated_files` paths from the previous step.
2.  **Review Design:**
    *   Check if the generated files (HTML/CSS/JS) exist and are accessible.
    *   Compare the output against the original `design_brief`.
    *   Assess basic code quality and functionality (e.g., rendering without major errors).
    *   *(Review Method: Specify if manual pause or delegation to e.g., `util-reviewer` is intended).*
3.  **Determine Status:** Set `review_status` to "Approved" or "Needs Revision".
4.  **Provide Feedback (If Needed):** If `review_status` is "Needs Revision", populate the `review_comments` output with specific feedback for the next generation attempt (used if looping back to step 01).

## Acceptance Criteria

*   The generated files have been reviewed against the brief.
*   `review_status` is set.

## Error Handling

*   If generated files are missing or inaccessible, proceed to `{{error_step}}`.