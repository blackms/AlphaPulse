+++
# --- Step Metadata ---
step_id = "WF-MODE-DELETE-V1-03-UPDATE-BUILD-COLLECTIONS" # (String, Required) Unique ID for this step.
title = "Step 03: Update Build Collections (If Necessary)" # (String, Required) Title of this specific step.
description = """
(String, Required) Checks if the deleted mode slug exists in the
`.ruru/config/build_collections.json` file. If found, delegates the removal
of the slug from the relevant collection(s) to `prime-dev`.
"""

# --- Flow Control ---
depends_on = ["WF-MODE-DELETE-V1-02-EXECUTE-DELETION"] # (Array of Strings, Required) step_ids this step needs completed first.
next_step = "04_run_build_script.md" # (String, Optional) Filename of the next step on successful completion.
error_step = "EE_handle_error.md" # (String, Optional) Filename to jump to if this step fails.

# --- Execution ---
delegate_to = "prime-coordinator" # (String, Optional) Coordinator checks the file and delegates if needed.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Data/artifacts needed.
    "Output from step WF-MODE-DELETE-V1-00-START: mode_slug",
]
outputs = [ # (Array of Strings, Optional) Data/artifacts produced by this step.
    "build_collections_update_status: 'Not Found', 'Update Delegated', or 'Update Failed'.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/25_workflow_step_standard.md" # (String, Required) Link to this template definition.
+++

# Step 03: Update Build Collections (If Necessary)

## Actions

(Instructions for the delegate: `{{delegate_to}}`)

1.  **Define Path & Slug:**
    *   `[mode_slug]` = Input `mode_slug`.
    *   `[build_collections_path]` = `.ruru/config/build_collections.json`.
2.  **Check File Content:**
    *   `- [ ]` Read the content of `[build_collections_path]`. (Prefer MCP `read_file_content`, fallback `read_file`). Handle file read error (**-> Error Step**).
    *   `- [ ]` Check if the string `"[mode_slug]"` exists within the file content. (Simple string search is likely sufficient, full JSON parsing might be overkill unless complex validation is needed).
3.  **Delegate Update (If Found):**
    *   `- [ ]` If `[mode_slug]` is found in the content:
        *   Delegate the task to `prime-dev` via `new_task`.
        *   Message: "Please remove all occurrences of the mode slug `[mode_slug]` from the arrays within the JSON file at `[build_collections_path]`. Ensure the resulting JSON remains valid."
        *   Await completion/confirmation from `prime-dev`.
        *   If `prime-dev` reports failure, log error, set `[build_collections_update_status]` = "Update Failed", and **-> Error Step**.
        *   If `prime-dev` reports success, log successful delegation and update, set `[build_collections_update_status]` = "Update Delegated".
    *   `- [ ]` If `[mode_slug]` is NOT found:
        *   Log that the slug was not found in the build collections file.
        *   Set `[build_collections_update_status]` = "Not Found".
4.  **Prepare Output:**
    *   `- [ ]` Prepare the `build_collections_update_status` output.

## Acceptance Criteria

*   The `.ruru/config/build_collections.json` file is checked for the presence of the `mode_slug`.
*   If the slug is found, the update task is successfully delegated to `prime-dev` and confirmed complete.
*   The `build_collections_update_status` reflects the outcome ('Not Found' or 'Update Delegated').

## Error Handling

*   Failure to read `build_collections.json` proceeds to `{{error_step}}`.
*   Failure reported by `prime-dev` during the update proceeds to `{{error_step}}`.