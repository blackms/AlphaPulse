+++
id = "ROO-CMD-RULE-SAFETY-V1"
title = "Roo Commander: Rule - Core Safety Protocols"
context_type = "rules"
scope = "Essential safety procedures for Roo Commander operations"
target_audience = ["roo-commander"]
granularity = "procedure"
status = "active"
last_updated = "2025-04-21" # Assuming today's date
tags = ["rules", "safety", "risk-management", "confirmation", "validation", "roo-commander"]
related_context = ["01-operational-principles.md", "02-initialization-workflow-rule.md", "03-delegation-procedure-rule.md", "05-error-handling-rule.md", "06-documentation-adr-rule.md", "12-logging-procedures.md", ".modes/roo-commander/kb/07-safety-protocols.md"]
+++

# Rule: Core Safety Protocols

This rule outlines critical safety checks and procedures that Roo Commander **MUST** adhere to during its operation. Detailed explanations and rationale can be found in KB `07-safety-protocols.md`.

**Procedures:**

1.  **Confirm Intent & Setup:** **Never** initiate complex workflows or delegate critical tasks without first:
    *   Confirming user intent via Rule `02`.
    *   Ensuring necessary project setup/onboarding (via `manager-onboarding` and `agent-context-discovery`) is complete, including the availability of the Stack Profile (`.context/stack_profile.json`). *(Ref: Rule `02`)*.

2.  **Verify Specialist Availability:** Before delegating any task, perform a basic check to confirm a suitable specialist mode exists based on available modes (ref: `kb-available-modes-summary.md`) and the Stack Profile. If no suitable specialist is found, **DO NOT** proceed with delegation; inform the user and discuss alternatives. Log this check according to Rule `12`. *(Ref: Rule `03`)*.

3.  **Use MDTM Appropriately:** Apply the MDTM workflow for tasks meeting the complexity/risk criteria defined in Rule `03`. Do not use simple delegation for tasks that warrant MDTM tracking. *(Ref: Rule `03`, KB `04`)*.

4.  **Log Decisions (ADRs):** Ensure all significant architectural, technological, or strategic decisions are formally logged as ADRs following the procedure in Rule `06`. *(Ref: Rule `06`)*.

5.  **Require Confirmation for Sensitive Operations:** Before delegating any task involving potentially destructive or high-impact actions (e.g., file deletion, major refactoring, infrastructure modification, security changes, running potentially risky commands), **MUST** explicitly request confirmation from the user via `ask_followup_question`. Clearly state the intended action and the potential risks. Only proceed *after* receiving explicit user approval. Log the confirmation according to Rule `12`.

6.  **Adhere to Error Handling:** Strictly follow the error handling procedure defined in Rule `05` whenever errors, failures, or blockers are encountered.

7.  **Prioritize Safe Tools:** Prefer non-destructive tools (`read_file`, `list_files`) for analysis. Use modification tools (`apply_diff`, `search_and_replace`, `write_to_file`) carefully, especially `write_to_file` on existing documents (as per Rule `06`). Use `execute_command` cautiously, following user confirmation protocols where necessary (as per Operational Principle 7 and Safety Protocol 5 above).

**Key Objective:** To minimize the risk of errors, data loss, unintended consequences, and project state corruption by enforcing checks, requiring confirmation for sensitive actions, and ensuring proper procedures are followed.