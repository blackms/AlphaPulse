+++
# --- Basic Metadata ---
step_id = "WF-CREATE-ROO-CMD-BUILD-V1-00B-SUGGEST-VERSION"
title = "Step 00b: Suggest & Confirm Next Version"
description = "Determines the latest Git tag, suggests the next patch version based on SemVer, and asks the user for confirmation."
workflow_id = "WF-CREATE-ROO-CMD-BUILD-V1" # Assuming this is needed for context
status = "Draft"
last_updated = "2025-05-01" # Using placeholder date

# --- Relationships ---
depends_on = ["WF-CREATE-ROO-CMD-BUILD-V1-00-START"]
next_step = "01_validate_params.md"
error_step = "EE_handle_git_tag_error.md" # Using the suggested error step

# --- Execution ---
delegate_to = "dev-git" # Using dev-git as primary suggestion
# timeout_seconds = 300 # Optional

# --- Data Flow ---
inputs = [
    # No explicit inputs listed, assumes context from previous step if needed
]
outputs = [
    "suggested_version", # The calculated next version string (e.g., "v1.2.4")
    "confirmed_version"  # The version confirmed or entered by the user
]

# --- Tags ---
tags = ["workflow-step", "git", "versioning", "semver", "user-interaction"]
+++

# Step 00b: Suggest & Confirm Next Version

## Description

This step automates the suggestion of the next release version based on existing Git tags following Semantic Versioning (SemVer) principles. It fetches the latest tags, identifies the most recent version tag (matching `v*.*.*`), calculates the next patch version, and prompts the user for confirmation or manual input.

## Actions

1.  **Fetch Tags:**
    *   Execute `git fetch --tags` to ensure local repository has all remote tags.
    *   *Tool:* `<execute_command>`
    *   *Command:* `git fetch --tags`

2.  **Find Latest Version Tag:**
    *   Execute `git describe --tags --abbrev=0 --match "v*.*.*"` to find the latest tag matching the `vX.Y.Z` pattern.
    *   *Tool:* `<execute_command>`
    *   *Command:* `git describe --tags --abbrev=0 --match "v*.*.*"`
    *   **Error Handling:** If the command fails (e.g., no matching tags found), proceed to suggest a default initial version (e.g., `v0.1.0`). Log this occurrence. If other errors occur, transition to `error_step`.

3.  **Calculate Suggested Version:**
    *   **If a latest tag was found:**
        *   Parse the tag (e.g., `v1.2.3`).
        *   Increment the patch number (e.g., `1.2.3` -> `1.2.4`).
        *   Construct the `suggested_version` string (e.g., `"v1.2.4"`).
    *   **If no latest tag was found:**
        *   Set `suggested_version` to a default initial version (e.g., `"v0.1.0"`).
    *   *Logic:* Implemented by the delegate (`dev-git`).

4.  **Ask for User Confirmation:**
    *   Use the `<ask_followup_question>` tool.
    *   *Question:* "The suggested next version is `[suggested_version]`. Please confirm or provide a different version."
    *   *Suggestions:*
        *   `Confirm [suggested_version]`
        *   `Enter different version` (This will require another interaction step or logic within the delegate to handle the input)
        *   `Cancel workflow` (This should trigger an appropriate termination or error path)
    *   *Tool:* `<ask_followup_question>`

5.  **Store Confirmed Version:**
    *   Based on the user's response to the follow-up question:
        *   If confirmed, set `confirmed_version` = `suggested_version`.
        *   If a different version is entered, set `confirmed_version` to the user-provided value (requires validation).
        *   If cancelled, handle workflow termination.
    *   *Logic:* Implemented by the delegate.

## Outputs

*   `suggested_version`: The automatically calculated next patch version (e.g., "v1.2.4").
*   `confirmed_version`: The version string that the user agreed upon or provided (e.g., "v1.2.4", "v1.3.0").