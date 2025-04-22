+++
id = "ROO-CMD-RULE-DOC-ADR-V1"
title = "Roo Commander: Rule - Documentation Oversight & ADR Creation"
context_type = "rules"
scope = "Managing documentation and logging significant decisions"
target_audience = ["roo-commander"]
granularity = "procedure"
status = "active"
last_updated = "2025-04-21" # Assuming today's date
tags = ["rules", "documentation", "adr", "logging", "decision-record", "architecture", "roo-commander"]
related_context = ["01-operational-principles.md", "12-logging-procedures.md", ".modes/roo-commander/kb/06-documentation-logging.md", ".templates/toml-md/07_adr.md", ".decisions/"]
+++

# Rule: Documentation Oversight & ADR Creation

This rule outlines Roo Commander's role in managing documentation and the procedure for creating Architecture Decision Records (ADRs).

**1. Documentation Management:**

*   **Oversight:** While not typically the primary author, Roo Commander oversees the existence and location of high-level project documentation (e.g., in `.docs/`, `.planning/`).
*   **Delegation Preference:** For creating or significantly updating stable documentation (especially within `.docs/`), **prefer delegating** the task to relevant specialists (e.g., `util-writer`, `core-architect`, the specialist whose work is being documented). Log the delegation according to Rule `12`.
*   **Direct Edits (Caution):** Commander may directly edit planning documents (`.planning/`) or create initial drafts using `write_to_file`. For *modifications* to existing documents, especially larger ones, strongly prefer `apply_diff` or `search_and_replace` over `write_to_file`. (See KB `06` for details).

**2. ADR Creation Trigger:**

*   An ADR **MUST** be created to log any **significant** architectural, technological, or strategic decision made during the project lifecycle. Examples include:
    *   Choice of primary programming language or framework.
    *   Selection of database technology.
    *   Adoption of a major architectural pattern (e.g., microservices, event-driven).
    *   Choice of cloud provider or core infrastructure components.
    *   Significant changes to API design philosophy or versioning strategy.
    *   Decisions with major performance, security, or cost implications.
    *   Resolution of complex technical disagreements with wide impact.

**3. ADR Creation Procedure:**

1.  **Identify Decision:** Recognize that a significant decision (as defined above) has been made or is being finalized.
2.  **Select Template:** Use the standard ADR template (`.templates/toml-md/07_adr.md`).
3.  **Determine Filename:** Construct filename: `.decisions/YYYYMMDD-brief-topic-summary.md`.
4.  **Gather Content:** Collect the necessary information for the ADR fields (Context, Decision, Rationale, Consequences).
5.  **Create ADR File:** Use `write_to_file` to create the new ADR file with the gathered content, ensuring correct TOML and Markdown formatting. Refer to KB `06` for detailed structure guidance.
6.  **Log ADR Creation:** Log the creation of the ADR (including its path and a brief summary of the decision) according to Rule `12`.

**Key Objective:** To ensure significant technical decisions are formally documented for traceability and future reference, and that documentation updates are handled appropriately, preferably through delegation for stable artifacts.