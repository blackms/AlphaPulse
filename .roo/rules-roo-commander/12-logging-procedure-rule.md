+++
id = "ROO-CMD-RULE-LOGGING-V1"
title = "Roo Commander: Rule - Logging Requirements & Locations"
context_type = "rules"
scope = "Standard requirements for logging events and decisions"
target_audience = ["roo-commander"]
granularity = "procedure"
status = "active"
last_updated = "2025-04-21" # Assuming today's date
tags = ["rules", "logging", "auditing", "traceability", "mdtm", "adr", "roo-commander"]
related_context = ["01-operational-principles.md", ".modes/roo-commander/kb/12-logging-procedures.md", ".tasks/", ".decisions/", ".logs/", ".planning/"]
+++

# Rule: Logging Requirements & Locations

This rule defines **what** events and decisions must be logged and **where** the logs should reside. For detailed instructions on **how** to use specific tools (`write_to_file`, `append_to_file`, `insert_content`, etc.) for different logging scenarios, consult the Knowledge Base document: **`.modes/roo-commander/kb/12-logging-procedures.md`**.

**1. Events Requiring Logging:**

You **MUST** create log entries for the following significant events:

*   Initiation of a new Commander coordination task.
*   Delegation of any task (simple or MDTM) to a specialist.
*   Rationale for selecting a specific specialist for delegation.
*   Creation or finalization of significant planning documents (`.planning/`).
*   Creation of Architecture Decision Records (ADRs) (`.decisions/`).
*   Detection of errors, failures, blockers, or safety violations.
*   Decisions made regarding error resolution or alternative approaches.
*   Escalation of issues to other modes.
*   Receipt of explicit user confirmation for sensitive operations.
*   Creation or significant updates to Workflow (`.workflows/`) or Process (`.processes/`) documents.
*   Updates to index files (`.modes/roo-commander/kb/10-*`, `.modes/roo-commander/kb/11-*`).
*   Significant status changes or milestones reached in coordinated tasks.
*   Completion of a Commander coordination task.

**2. Standard Logging Locations:**

*   **Commander's Task Log:**
    *   **Location:** `.tasks/`
    *   **Naming:** `TASK-CMD-YYYYMMDD-HHMMSS.md`
    *   **Use:** Primary log for Commander's own actions, decisions, delegation records, monitoring notes, rationale, user interactions, and coordination efforts related to a specific high-level user request or workflow.
*   **Specialist MDTM Task Files:**
    *   **Location:** `.tasks/`
    *   **Naming:** `TASK-[MODE]-[YYYYMMDD-HHMMSS].md`
    *   **Use:** Primarily managed by the specialist, but Commander may append coordination notes if necessary (use KB `12` for tool guidance).
*   **Architecture Decision Records (ADRs):**
    *   **Location:** `.decisions/`
    *   **Naming:** `YYYYMMDD-brief-topic-summary.md`
    *   **Use:** Formal record of significant decisions.
*   **General Logs:**
    *   **Location:** `.logs/`
    *   **Naming:** Descriptive (e.g., `YYYYMMDD-HHMMSS-command-output.log`).
    *   **Use:** Storing raw output from tools like `execute_command`, build processes, etc.
*   **Planning Documents:**
    *   **Location:** `.planning/`
    *   **Naming:** Descriptive (e.g., `project_plan_v1.md`).
    *   **Use:** Record high-level plans and strategic decisions.

**3. Tool Usage Guidance:**

*   Refer to **`.modes/roo-commander/kb/12-logging-procedures.md`** for detailed instructions on *which tools* (`write_to_file`, `append_to_file`, `insert_content`, `apply_diff`) are appropriate for creating new logs versus updating existing ones in different locations.

**Key Objective:** To ensure a consistent, traceable, and comprehensive record of Roo Commander's coordination activities, decisions, and interactions with other modes and the user.