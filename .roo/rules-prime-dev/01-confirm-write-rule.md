+++
id = "PRIME-DEV-RULE-CONFIRM-WRITE-V1"
title = "Prime Config Editor: Rule - Confirm ALL Writes"
context_type = "rules"
scope = "Mandatory confirmation before any file modification"
target_audience = ["prime-dev"]
granularity = "procedure"
status = "active"
last_updated = "2025-04-21" # Assuming today's date
tags = ["rules", "safety", "confirmation", "write", "edit", "prime-dev"]
related_context = [".modes/prime-dev/prime-dev.mode.md"]
+++

# Rule: Confirm ALL Writes Before Execution

**Applies To:** All tasks involving file modification (`write_to_file`, `apply_diff`).

**CRITICAL PROCEDURE:**

1.  **Prepare Change:** Based on the instructions received from the Prime Coordinator, read the target file (`read_file`) and prepare the exact content or diff for the modification. Ensure you are only targeting the file path specified by the Coordinator and that your proposed changes maintain valid syntax for the target file type (e.g., TOML, JS, JSON, Markdown).
2.  **Generate Confirmation Prompt:** Before executing `write_to_file` or `apply_diff`, you **MUST** use the `<ask_followup_question>` tool to present the proposed change to the user for explicit approval.
    *   **Question Content:** The question must clearly state:
        *   The **full path** of the file being modified.
        *   The **intended action** (e.g., "Apply the following changes:", "Overwrite file with the following content:").
        *   The **exact changes** (using a diff format for `apply_diff` or showing significant portions/the entirety of the new content for `write_to_file`). Use Markdown code fences (` ``` `) with appropriate language identifiers (e.g., ` ```toml`, ` ```javascript`, ` ```diff`) for clarity.
    *   **Suggestions:** Include clear "Yes, apply change" and "No, cancel" suggestions.
        ```xml
        <ask_followup_question>
        <question>
        Proposed change to file: `[FULL_TARGET_FILE_PATH]`

        ```[language|diff]
        [PROPOSED_DIFF_OR_CONTENT_SNIPPET]
        ```

        Apply this change?
        </question>
        <follow_up>
        <suggest>Yes, apply the change to `[FULL_TARGET_FILE_PATH]`</suggest>
        <suggest>No, cancel the change</suggest>
        </follow_up>
        </ask_followup_question>
        ```
3.  **Await User Response:** **DO NOT** proceed with the file modification tool until the user responds to the `ask_followup_question`.
4.  **Execute if Confirmed:** If the user explicitly confirms (e.g., selects "Yes..."), then execute the planned `write_to_file` or `apply_diff` command.
5.  **Report Outcome:** Report the final outcome (success, failure, or user cancellation) back to the Prime Coordinator using `attempt_completion`. If the user cancelled, state that clearly.

**Rationale:** This rule is the primary safety mechanism preventing accidental or unapproved modifications to files when operating in Prime mode. Strict adherence is mandatory.