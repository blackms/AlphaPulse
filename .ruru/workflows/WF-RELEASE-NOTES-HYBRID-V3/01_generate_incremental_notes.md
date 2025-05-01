+++
# --- Step Metadata ---
step_id = "WF-RELEASE-NOTES-HYBRID-V3-01-GEN-INCREMENTAL" # Unique ID for this step
title = "Step 01: Generate Incremental Release Notes" # Title of this specific step
description = """
Analyzes Git history between the previously determined tag and HEAD to find new conventional commits.
Generates release notes content for these commits and saves it to a new timestamped file
in the incremental notes directory (`next/`). This step only runs if the action is 'add'.
"""

# --- Flow Control ---
depends_on = ["WF-RELEASE-NOTES-HYBRID-V3-00-START"] # Depends on the initialization step
next_step = "99_finish.md" # Proceeds to finish after adding notes
error_step = "99_finish.md" # Default error handling to finish step

# --- Execution ---
# Needs a mode capable of running git log, parsing commits, and generating notes.
# 'dev-git' might be suitable, or a future 'release-notes-generator' mode.
delegate_to = "dev-git" # Placeholder - adjust if a better mode exists

# --- Interface ---
inputs = [ # Data/artifacts needed from previous steps
    "Output from step {{depends_on[0]}}: determined_previous_tag",
    "Output from step {{depends_on[0]}}: incremental_notes_dir",
    "Output from step {{depends_on[0]}}: determined_action" # Needed to confirm action is 'add' (though orchestrator handles branching)
]
outputs = [ # Data/artifacts produced by this step
    "incremental_notes_file_path: Path to the newly created Markdown file containing notes for recent commits.",
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # Placeholder
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # Link to the template definition
+++

# Step 01: Generate Incremental Release Notes

## Precondition
*   This step assumes the `determined_action` from the start step is `"add"`.

## Actions

1.  **Define Commit Range:** Identify the Git commit range to analyze: `{{determined_previous_tag}}..HEAD`.
2.  **Execute Git Log:** Run `git log` for the defined range, formatting the output to clearly show commit hashes and full messages (suitable for conventional commit parsing).
    *   *Example Command (Conceptual):* `git log {{determined_previous_tag}}..HEAD --pretty=format:"%H %s%n%b"`
3.  **Parse Commits:** Process the `git log` output to identify commits adhering to the Conventional Commits specification (e.g., `feat:`, `fix:`, `perf:`, `BREAKING CHANGE:`).
4.  **Generate Notes Content:** Format the parsed commits into Markdown sections (e.g., Features, Bug Fixes, Breaking Changes).
5.  **Determine Output Filename:** Create a unique filename for the incremental notes, including a timestamp (e.g., `update_YYYYMMDDHHMMSS.md`).
6.  **Construct Full Path:** Combine `{{incremental_notes_dir}}` and the generated filename.
7.  **Save Incremental Notes:** Write the generated Markdown content to the constructed file path. Store this path as `incremental_notes_file_path`.

## Acceptance Criteria

*   A new Markdown file exists at `{{incremental_notes_file_path}}`.
*   The file contains formatted release notes based on conventional commits found in the specified range. If no relevant commits were found, the file might be empty or contain a "No changes" message.

## Error Handling

*   If `git log` fails, proceed to `{{error_step}}`.
*   If file writing fails, proceed to `{{error_step}}`.