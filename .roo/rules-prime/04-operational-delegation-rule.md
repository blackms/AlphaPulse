+++
id = "PRIME-RULE-OPS-DELEGATE-V1"
title = "Prime Coordinator: Rule - Operational Task Delegation"
context_type = "rules"
scope = "Delegating standard development tasks (features, bugs, etc.)"
target_audience = ["prime"]
granularity = "procedure"
status = "active"
last_updated = "2025-04-21"
tags = ["rules", "delegation", "operational", "mdtm", "prime"]
related_context = ["01-operational-principles.md", "03-delegation-procedure-rule.md", "04-monitoring-completion-rule.md", "12-logging-procedures.md", "kb-available-modes-summary.md", ".context/stack_profile.json"]
+++

# Rule: Operational Task Delegation

This rule defines the procedure for delegating standard software development tasks (implementing features, fixing bugs, writing tests, etc.) to operational specialist modes.

**Procedure:**

1.  **Receive Goal:** Understand the operational goal from the user.
2.  **Select Specialist:** Choose the appropriate *operational* specialist mode (e.g., `framework-react`, `dev-api`, `test-e2e`) using the guidance in Rule `03` (Analyze Task, Consult Stack Profile, Consult Mode Summary, Match Tags, Prioritize Specificity). **Do NOT select Prime modes (`prime-txt`, `prime-dev`) for operational tasks.**
3.  **Log Rationale:** Justify specialist choice according to Rule `07` / KB `12`.
4.  **Determine Delegation Method:** Decide between simple `new_task` or MDTM workflow based on complexity/risk criteria (similar to Rule `03` criteria, but applied to operational tasks).
5.  **Prepare Context:** Gather necessary context (requirements, related file paths, Stack Profile, parent Task IDs, etc.).
6.  **Execute Delegation:**
    *   **If MDTM:** Follow KB `04` procedure (Create MDTM task file in `.tasks/`, then `new_task` pointing to it).
    *   **If Simple:** Delegate directly using `new_task` with clear objective, acceptance criteria, and context.
7.  **Log Delegation:** Record delegation details according to Rule `07` / KB `12`.
8.  **Monitor Task:** Track progress according to Rule `04`.