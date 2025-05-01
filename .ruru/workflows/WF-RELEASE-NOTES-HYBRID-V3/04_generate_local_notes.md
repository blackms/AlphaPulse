+++
# --- Step Metadata ---
step_id = "WF-RELEASE-NOTES-HYBRID-V2-04-GENERATE_LOCAL_NOTES" # Unique ID
title = "Step 04: Generate Local Release Notes File" # Based on original step 5
description = """
Combines the generated release notes body with appropriate TOML frontmatter
(using a template) and writes the complete content to a local Markdown file.
"""

# --- Flow Control ---
depends_on = [
    "WF-RELEASE-NOTES-HYBRID-V2-00-START", # Need validated_output_dir, validated_target_tag
    "WF-RELEASE-NOTES-HYBRID-V2-03-SUMMARIZE_CHANGES" # Need release_notes_body
]
next_step = "05_push_to_github.md" # Based on original step 6
error_step = "99_finish.md" # Default error handling

# --- Execution ---
delegate_to = "prime-txt" # Suitable for file writing with templates

# --- Interface ---
inputs = [
    "Output from step WF-RELEASE-NOTES-HYBRID-V2-00-START: validated_output_dir",
    "Output from step WF-RELEASE-NOTES-HYBRID-V2-00-START: validated_target_tag",
    "Output from step WF-RELEASE-NOTES-HYBRID-V2-03-SUMMARIZE_CHANGES: release_notes_body"
]
outputs = [
    "local_release_notes_path: The full path to the generated local Markdown file."
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # Placeholder
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md"
+++

# Step 04: Generate Local Release Notes File

## Actions

1.  **Determine File Path:** Construct the output file path using the inputs: `{{validated_output_dir}}/{{validated_target_tag}}.md`. Store this as `local_release_notes_path`.
2.  **Prepare TOML Frontmatter:**
    *   Load the standard release notes template: `.ruru/templates/toml-md/13_release_notes.md`.
    *   Populate the template's TOML fields:
        *   `id`: Use `{{validated_target_tag}}` or a generated ID.
        *   `title`: e.g., "Release Notes - {{validated_target_tag}}"
        *   `version`: `{{validated_target_tag}}`
        *   `date`: Current date (e.g., `{{DATE}}`)
        *   `status`: "Published" (or similar)
        *   `tags`: ["release-notes", `{{validated_target_tag}}`]
        *   (Add any other relevant fields from the template)
3.  **Combine Content:** Concatenate the prepared TOML frontmatter (within `+++` delimiters) and the `release_notes_body` Markdown content.
4.  **Delegate to `prime-txt`:** Instruct `prime-txt` to use the `write_to_file` tool:
    *   `path`: `local_release_notes_path`
    *   `content`: The combined TOML and Markdown content.
    *   Note: Assumes `prime-txt`'s `write_to_file` tool handles creation of the output directory (`{{validated_output_dir}}`) if it doesn't exist.
    *   `line_count`: Calculate the total line count.
5.  **Report Path:** Output the `local_release_notes_path`.

## Acceptance Criteria

*   The TOML frontmatter is correctly prepared based on a template and inputs.
*   The combined content (TOML + Markdown body) is generated.
*   The `write_to_file` command is successfully executed by `prime-txt`.
*   The `local_release_notes_path` output variable contains the correct path to the created file.

## Error Handling

*   If the release notes template cannot be found or read, proceed to `{{error_step}}`.
*   If `prime-txt` reports an error during file writing, capture the error and proceed to `{{error_step}}`.