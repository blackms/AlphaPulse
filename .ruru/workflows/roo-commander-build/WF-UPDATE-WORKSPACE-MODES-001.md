+++
# --- Basic Metadata ---
id = "WF-UPDATE-WORKSPACE-MODES-001"
title = "Workflow: Update Local Workspace Modes"
description = "Updates the local workspace `.roomodes` file and mode summary during Roo Commander development."
status = "active"
tags = ["workflow", "build", "workspace", "development", "modes"]
version = "1.0"
# last_updated = "YYYY-MM-DD" # Optional: Add current date if needed, but not requested
# author = "Prime Documenter" # Optional
# related_context = [] # Optional
# template_schema_doc = ".ruru/templates/toml-md/14_workflow.README.md" # Assuming this is the correct template schema doc based on the request
+++

# Workflow: Update Local Workspace Modes

**Purpose:** This workflow outlines the steps to update the local workspace `.roomodes` file and the mode summary documentation. This is intended for use during the development of Roo Commander itself to reflect changes in available modes.

**Scope:** Local development environment only.

**Steps:**

1.  **Generate `.roomodes`:**
    *   **Action:** Execute the command `node .ruru/scripts/build_roomodes.js`.
    *   **Purpose:** This script reads mode definitions and generates/updates the `.roomodes` file in the workspace root, which is used by the Roo Commander extension.

2.  **Generate Mode Summary:**
    *   **Action:** Execute the command `node .ruru/scripts/build_mode_summary.js`.
    *   **Purpose:** This script generates a Markdown summary of available modes (typically located at `.ruru/modes/roo-commander/kb/kb-available-modes-summary.md`). This keeps the documentation in sync with the actual modes.
    *   **Note:** Verify the exact output path if it differs from the typical location mentioned above.

3.  **Confirmation:**
    *   **Action:** Check the output of both scripts in the terminal for success messages.
    *   **Verification:** Manually inspect the timestamps or content of the `.roomodes` file and the mode summary Markdown file to confirm they have been updated.