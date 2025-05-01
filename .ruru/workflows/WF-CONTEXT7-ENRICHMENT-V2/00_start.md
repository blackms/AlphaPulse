+++
# --- Step Metadata ---
step_id = "WF-CONTEXT7-ENRICHMENT-V2-00-START" # (String, Required) Unique ID for this step.
title = "Step 00: Initiation & Requirements Gathering" # (String, Required) Title of this specific step.
description = """
Gathers initial requirements from the User: the target existing mode slug and the base URL for the Context7 documentation source. Confirms details before proceeding.
"""

# --- Flow Control ---
depends_on = [] # (Array of Strings, Required) Always empty for the start step.
next_step = "01_fetch_metadata.md" # (String, Required) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "roo-commander" # (String, Optional) Coordinator handles user interaction.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed. Usually references overall workflow inputs.
    "Workflow Input: User request specifying goal (enrich KB for existing mode X using Context7 URL Y).",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "mode_slug: Confirmed slug of the target existing mode.",
    "context7_base_url: Confirmed Context7 base URL.",
    "user_confirmation: Record of user confirming the gathered requirements."
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/24_workflow_step_start.md" # (String, Required) Link to this template definition.
+++

# Step 00: Initiation & Requirements Gathering

## Actions

1.  **Identify Target Mode:**
    *   Ask User for the approximate name/slug of the target *existing* mode.
    *   List mode slugs from `.ruru/modes/`.
    *   Filter and present options to User using `ask_followup_question` if needed.
    *   Confirm the exact `[mode_slug]` with the User. Handle errors/no match -> **Stop**.
2.  **Get Context7 URL:**
    *   Ask User for the **Context7 base URL** (e.g., `https://context7.com/library/docs`).
    *   Store as `[context7_base_url]`.
    *   Perform basic URL format validation. Handle errors -> **Stop**.
3.  **Final Confirmation:**
    *   Summarize the confirmed `[mode_slug]` and `[context7_base_url]`.
    *   Ask User for final confirmation to proceed using `ask_followup_question`. Handle cancellation -> **Stop**.
    *   Store confirmation status in `[user_confirmation]`.

## Acceptance Criteria

*   The target `[mode_slug]` is confirmed.
*   A valid `[context7_base_url]` is provided and confirmed.
*   User has confirmed readiness to proceed (`[user_confirmation]` is affirmative).

## Error Handling

*   If User cannot provide a valid mode slug or URL, or cancels confirmation, stop the workflow or proceed to `{{error_step}}` (if defined).
*   Log errors encountered during validation.