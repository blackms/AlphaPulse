+++
id = "PRIME-WORKER-RULE-SUMMARIZED-CONFIRM-V3" # Incremented version
title = "Prime Worker: Rule - Summarized & Confidence-Based Write Confirmation"
context_type = "rules"
scope = "Summarization, confidence assessment, and explicit flag check before file modification"
target_audience = ["prime-txt", "prime-dev"] # Apply to both workers
granularity = "procedure"
status = "active"
last_updated = "2025-04-21"
tags = ["rules", "safety", "confirmation", "confidence", "risk", "summary", "write", "edit", "prime-txt", "prime-dev", "explicit-flag"] # Added summary
related_context = [".modes/prime-txt/prime-txt.mode.md", ".modes/prime-dev/prime-dev.mode.md"]
+++

# Rule: Summarized & Confidence-Based Write Confirmation

**Applies To:** All tasks involving file modification (`write_to_file`, `apply_diff`) on paths provided by Prime Coordinator.

**Procedure:**

1.  **Prepare Change:** Based on instructions, read the target file (`read_file`) and prepare the exact content or diff for modification. Ensure valid syntax.
2.  **Check for Forced Confirmation:** Examine the instruction message received from Prime Coordinator. Does it contain the explicit flag `USER_CONFIRMATION_REQUIRED=TRUE`?
3.  **Self-Assess Confidence & Risk:** Critically evaluate your proposed change:
    *   **Confidence:** Are you highly confident (>95%) that the change exactly matches the instruction and is free from errors or unintended consequences?
    *   **Risk - Ambiguity:** Were the instructions clear?
    *   **Risk - Impact:** Does the change involve large deletions, core structural modifications, or potential syntax breaks?
4.  **Determine Confirmation Need:**
    *   **IF** `USER_CONFIRMATION_REQUIRED=TRUE` flag was present (Step 2)...
    *   **OR IF** you have significant doubt about confidence/ambiguity/impact (Step 3)...
    *   **THEN:** Proceed to Step 5 (Generate Summarized Confirmation Prompt).
    *   **ELSE (High Confidence / Low Risk / Not Forced):** Proceed directly to Step 7 (Propose Edit via Tool).
5.  **Generate Change Summary (If Confirmation Needed):**
    *   Based on the prepared change (diff or full content), create a **concise bullet-point summary** (1-3 points ideally) describing the *essence* of the modification (e.g., "- Updated TOML version field", "- Added section on error handling", "- Refactored function X for clarity").
    *   **If** the change is very small (e.g., fixing a typo, changing a single value) or a summary is difficult/risky to generate accurately, you MAY present the short diff/snippet directly instead of a summary.
6.  **Generate Confirmation Prompt (If Needed):**
    *   You **MUST** use `<ask_followup_question>` to present the summary (or short snippet/diff from Step 5) and the target file path.
    *   Include a brief reason if confirmation was triggered by *your* assessment (Step 3). If triggered by the flag (Step 2), state "Confirmation requested by Coordinator."
    *   Include clear "Yes, apply summarized change" and "No, cancel" suggestions. Add an option to "Show full diff first".
        ```xml
        <ask_followup_question>
        <question>
        Proposed change to file: `[TARGET_FILE_PATH]`
        Reason for Confirmation: [State reason]

        **Summary of Change:**
        *   [Summary Point 1]
        *   [Summary Point 2]
        (OR: ```diff/[lang]\n[Short Snippet/Diff]\n``` if summary not feasible)

        Apply this summarized change?
        </question>
        <follow_up>
        <suggest>Yes, apply summarized change to `[TARGET_FILE_PATH]`</suggest>
        <suggest>Show full diff first</suggest>
        <suggest>No, cancel the change</suggest>
        </follow_up>
        </ask_followup_question>
        ```
    *   **Await User Response:** Do not proceed until the user responds.
    *   **If User selects "Show full diff first":** Generate another `<ask_followup_question>` prompt, this time including the *full diff* or content snippet as in the previous rule version, asking again for confirmation (Yes/No). Await response. If *this* is confirmed, proceed to Step 7. If cancelled, abort and report cancellation.
    *   **If User selects "Yes, apply summarized change":** Proceed to Step 7.
    *   **If User selects "No, cancel":** Abort the operation and report the cancellation to Prime Coordinator via `attempt_completion`.
7.  **Propose Edit via Tool:**
    *   Propose the *original, full change* (not the summary) using the appropriate tool (`apply_diff` or `write_to_file`).
    *   **Note:** The standard Roo Code approval/auto-approval flow will still apply to the tool's proposal itself.
8.  **Report Outcome:** Report the final outcome (success after tool proposal, failure during tool proposal, or prior cancellation by user) back to the Prime Coordinator using `attempt_completion`.

**Rationale:** Reduces user friction by presenting a summary for confirmation in most cases where doubt exists. Provides an escape hatch ("Show full diff") for users who want to inspect the exact changes before approving. High-confidence/low-risk changes still bypass this confirmation prompt entirely.