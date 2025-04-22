+++
id = "ROO-CMD-RULE-MONITOR-COMPLETE-V1"
title = "Roo Commander: Rule - Task Monitoring & Completion Handling"
context_type = "rules"
scope = "Monitoring delegated tasks and processing completion signals"
target_audience = ["roo-commander"]
granularity = "procedure"
status = "active"
last_updated = "2025-04-21" # Assuming today's date
tags = ["rules", "monitoring", "completion", "delegation", "workflow", "mdtm", "roo-commander"]
related_context = ["01-operational-principles.md", "03-delegation-procedure-rule.md", "05-error-handling-rule.md", "12-logging-procedures.md"]
+++

# Rule: Task Monitoring & Completion Handling

This rule defines the procedure for monitoring delegated tasks (both simple and MDTM) and processing the `attempt_completion` signal from specialist modes.

**Procedure:**

1.  **Await Completion Signal:** After delegating a task (via simple `new_task` or MDTM workflow initiation), wait for the specialist mode to respond with `<attempt_completion>`.

2.  **Process Completion Signal:** Upon receiving `<attempt_completion>`:
    *   **Extract Result:** Parse the `<result>` tag content provided by the specialist.
    *   **Identify Task:** Correlate the completion signal with the originally delegated task (using context, Task IDs referenced in the result, etc.).
    *   **Assess Outcome:** Determine if the specialist reported success, failure, or a blocked state based on the result message and structure (e.g., presence of success markers ✅ or error indicators ❌/🧱).

3.  **Update Task Status (MDTM):**
    *   **If** the completed task was part of an MDTM workflow (i.e., the specialist was processing a `.tasks/TASK-[MODE]-....md` file):
        1.  Use `read_file` to check the *current* `status` in the TOML block of the specialist's task file (`.tasks/TASK-[MODE]-....md`).
        2.  Based on the specialist's reported outcome (Step 2c), update the `status` field in the **TOML block** of that same task file using `apply_diff`. Set it to:
            *   `"🟣 Review"` (if successful and requires Commander/User review).
            *   `"🟢 Done"` (if successful and no further review needed).
            *   `"⚪ Blocked"` (if the specialist reported a blocker).
            *   **(Keep as is if specialist failed and is expected to retry/report failure).**
        3.  Update the `updated_date` field in the TOML block.
        4.  Log this status update according to Rule `12`.

4.  **Review Specialist Output/Result:**
    *   Carefully review the `<result>` content. Does it meet the acceptance criteria of the original task?
    *   If the task involved file modifications, consider using `read_file` to verify the changes reported by the specialist.
    *   If the task involved complex logic or critical changes, consider initiating a formal review process (e.g., delegating to `util-reviewer` or involving the user).

5.  **Handle Failures/Blockers:**
    *   If the specialist reported failure (❌) or a blocked state (🧱), initiate the consolidated error handling procedure defined in Rule `05`.

6.  **Log Completion:** Record the completion of the delegated task (success, requires review, or failure handled) in the Commander's coordination log (`.tasks/TASK-CMD-....md`) according to Rule `12`.

7.  **Proceed:** Determine the next step in the overall workflow based on the completed task's outcome (e.g., delegate the next sequential task, report progress to the user, initiate review).

**Handling No Response / Stalled Tasks:**

*   If an `attempt_completion` signal is not received within a reasonable timeframe for an MDTM task:
    1.  Use `read_file` on the `.tasks/TASK-[MODE]-....md` file to check the checklist status.
    2.  If progress seems stalled, consider re-delegating using `new_task`, pointing to the *existing* task file and asking the specialist to resume. Log this action (Rule `12`).
    3.  If the task file indicates completion but no signal was received, investigate potential communication issues or proceed based on the file content (use caution).

**Key Objective:** To ensure delegated tasks are tracked, completion signals are processed correctly, MDTM statuses are updated accurately, and failures are routed to the appropriate error handling procedure.