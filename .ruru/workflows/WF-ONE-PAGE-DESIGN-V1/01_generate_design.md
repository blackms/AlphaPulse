+++
# --- Step Metadata ---
step_id = "WF-ONE-PAGE-DESIGN-V1-01-GENERATE-DESIGN"
title = "Step 01: Generate One Page Design"
description = """
Delegates the task of generating the one-page website design (HTML, CSS, minimal JS)
to the `design-one-shot` specialist mode, providing the design brief from the previous step.
"""

# --- Flow Control ---
depends_on = ["WF-ONE-PAGE-DESIGN-V1-00-START"]
next_step = "02_review_design.md" # Proceed to review step
error_step = "EE_workflow_error.md" # Standard error handler step

# --- Execution ---
delegate_to = "design-one-shot" # Specialist mode for this task

# --- Interface ---
inputs = [
    "Output from step WF-ONE-PAGE-DESIGN-V1-00-START: design_brief",
    "Workflow Input: output_directory (Optional, defaults to .ruru/artifacts/one-page-design/[timestamp]/)",
    "Output from step WF-ONE-PAGE-DESIGN-V1-02-REVIEW-DESIGN: review_comments (Optional, for revision loop)", # Added for potential future revision loop
]
outputs = [
    "generated_files: Paths to the generated HTML/CSS/JS files.",
    "design_summary: Brief summary from the design specialist.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}"
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md"
+++

# Step 01: Generate One Page Design

## Actions

1.  **Receive Inputs:** Obtain the `design_brief` artifact from step `00_start.md` and the optional `output_directory` from the workflow inputs. Also check for optional `review_comments` if this is part of a revision cycle.
2.  **Determine Output Path:** Define the target directory for generated files based on the `output_directory` input or a default timestamped path (e.g., `.ruru/artifacts/one-page-design/[timestamp]/`).
3.  **Delegate to Specialist:** Initiate a `new_task` for the `design-one-shot` mode.
    *   **Message:** "Generate a one-page website based on the following design brief: [Include content of `design_brief` artifact]. [Include `review_comments` if present]. Save the resulting HTML, CSS, and JS files to the specified directory: [Include determined output path]."
4.  **Receive Output:** Wait for the `design-one-shot` mode to complete and provide the paths to the `generated_files` (within the specified directory) and a `design_summary`.

## Acceptance Criteria

*   The `design-one-shot` mode successfully generates the HTML, CSS, and JS files.
*   The paths to the `generated_files` are received as output.
*   A `design_summary` is received.

## Error Handling

*   If the `design-one-shot` mode fails, cannot generate the design based on the brief, or fails to save files to the specified location, proceed to `{{error_step}}`.