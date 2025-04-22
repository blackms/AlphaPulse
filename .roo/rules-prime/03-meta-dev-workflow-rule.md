+++
id = "PRIME-RULE-METADEV-V1"
title = "Prime Coordinator: Rule - Meta-Development Workflow (Config Changes)"
context_type = "rules"
scope = "Handling requests to modify Roo Commander configuration files"
target_audience = ["prime"]
granularity = "procedure"
status = "active"
last_updated = "2025-04-21"
tags = ["rules", "workflow", "meta-development", "configuration", "staging", "safety", "prime"]
related_context = ["07-logging-confirmation-rule.md", "prime-txt", "prime-dev"]
+++

# Rule: Meta-Development Workflow (Config Changes)

This rule defines the procedure for handling requests to modify Roo Commander configuration files.

**Procedure:**

1.  **Receive Request:** Obtain the target file path (`TARGET_PATH`) and the desired changes from the user or previous steps.
2.  **Define Protected Paths:** Identify the set of critical core files:
    *   `PROTECTED_PATHS` = patterns matching: `.roo/rules/**`, `.modes/roo-commander/**`, `.roo/rules-roo-commander/**`, `.modes/prime*/**`, `.roo/rules-prime*/**`, `.roomodes*`, `build_*.js`, `create_build.js`.
3.  **Check Protection Status:** Determine if `TARGET_PATH` matches any pattern in `PROTECTED_PATHS`.
4.  **Execute Workflow:**
    *   **IF `TARGET_PATH` IS PROTECTED:**
        1.  **Initiate Staging:**
            *   Define `STAGING_PATH` (e.g., `.staging/` + relative path of `TARGET_PATH`).
            *   Use `execute_command` to copy `TARGET_PATH` to `STAGING_PATH` (`cp <TARGET_PATH> <STAGING_PATH>`). Log this action (Rule `07`). Handle copy errors.
        2.  **Delegate Staging Edit:** Use `new_task` to delegate the edit to `prime-txt` (for `.md`) or `prime-dev` (for others), providing the `STAGING_PATH` and clear editing instructions. Log delegation (Rule `07`).
        3.  **Await Completion:** Wait for `attempt_completion` from the worker. Handle worker errors (Rule `05`).
        4.  **Generate Diff:** Use `execute_command` to generate a diff (`diff -u <TARGET_PATH> <STAGING_PATH>`). Handle diff errors.
        5.  **Present Diff & Instruct:** Use `ask_followup_question` to:
            *   Present the generated diff clearly.
            *   State explicitly: "Changes applied to staging file (`<STAGING_PATH>`). Review the diff above. If approved, **you must manually apply these changes** to the original file (`<TARGET_PATH>`)."
            *   Ask: "Shall I remove the staging file (`<STAGING_PATH>`) now?" Suggest "Yes, remove staging file" and "No, keep staging file".
        6.  **Cleanup (If Confirmed):** If user confirms removal, use `execute_command` (`rm <STAGING_PATH>`). Log cleanup (Rule `07`).
        7.  **Report Outcome:** Inform user the staging workflow is complete using `attempt_completion`.
    *   **ELSE (`TARGET_PATH` IS NOT PROTECTED):**
        1.  **Delegate Direct Edit:** Use `new_task` to delegate the edit to `prime-txt` (for `.md`) or `prime-dev` (for others), providing the **direct operational `TARGET_PATH`** and clear editing instructions. Remind the worker they MUST seek confirmation before writing (as per their own rules). Log delegation (Rule `07`).
        2.  **Await Completion:** Wait for `attempt_completion` from the worker.
        3.  **Report Outcome:** Report the success or failure (including any confirmation rejections or permission errors reported by the worker) back to the user using `attempt_completion`.

**Safety Note:** The safety of the "Direct Edit Workflow" relies entirely on the mandatory confirmation rule within `prime-txt` and `prime-dev`.