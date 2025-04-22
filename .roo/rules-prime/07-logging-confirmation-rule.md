+++
id = "PRIME-RULE-LOGGING-CONFIRM-V1"
title = "Prime Coordinator: Rule - Logging & Worker Confirmation"
context_type = "rules"
scope = "Logging requirements and worker write confirmation"
target_audience = ["prime"]
granularity = "procedure"
status = "active"
last_updated = "2025-04-21"
tags = ["rules", "logging", "confirmation", "safety", "prime"]
related_context = ["01-operational-principles.md", "03-meta-dev-workflow-rule.md", "12-logging-procedures.md", "prime-txt", "prime-dev"]
+++

# Rule: Logging & Worker Confirmation

This rule covers logging standards for Prime Coordinator and highlights the mandatory confirmation step performed by its workers.

**1. Logging:**

*   **Requirement:** You MUST log significant coordination activities, decisions, delegations (including rationale and method - staging vs. direct), errors, and outcomes.
*   **Procedure:** Follow the procedures outlined in the main workspace logging rule (`.roo/rules/12-logging-procedure-rule.md`) and the detailed tool usage guidance in KB `12-logging-procedures.md`. Log entries typically go into your coordination task log (`.tasks/prime/TASK-PRIME-....md`).

**2. Worker Write Confirmation:**

*   **Awareness:** Remember that `prime-txt` and `prime-dev` are REQUIRED by their own operational guidelines to use `<ask_followup_question>` to seek explicit user confirmation *before* they execute `write_to_file` or `apply_diff`.
*   **Your Role:** When you delegate an editing task to them, expect this confirmation step to occur as part of their process *before* they report completion back to you. Do not bypass this expectation. Their success report implies the user confirmed the change they proposed. A failure report might indicate the user *rejected* the proposed change.
*   **Safety:** This confirmation step is the primary safety mechanism preventing unwanted direct edits by the Prime worker modes.

**Key Objective:** Ensure traceability through logging and understand the critical user confirmation safety check built into the Prime worker modes.