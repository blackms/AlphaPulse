+++
id = "ROO-CMD-RULE-DELEGATION-V1"
title = "Roo Commander: Rule - Standard Delegation Procedure"
context_type = "rules"
scope = "Delegating tasks to specialist modes"
target_audience = ["roo-commander"]
granularity = "procedure"
status = "active"
last_updated = "2025-04-21" # Assuming today's date
tags = ["rules", "delegation", "workflow", "specialists", "mdtm", "roo-commander"]
related_context = ["01-operational-principles.md", "kb-available-modes-summary.md", ".context/stack_profile.json", "04-delegation-mdtm.md", "12-logging-procedures.md", ".templates/toml-md/"]
+++

# Rule: Standard Delegation Procedure

This rule outlines the process for selecting specialists and delegating tasks, including determining whether to use the MDTM workflow.

**Procedure:**

1.  **Define Task & Goal:** Clearly define the specific task objective and desired outcome based on the overall plan or user request.
2.  **Select Specialist:** Choose the *most appropriate* specialist mode for the task by:
    *   a. **Analyzing Task:** Identify the primary domain (e.g., frontend, backend, database, infrastructure) and required skills/technologies (e.g., React, Python, Terraform, CSS, SQL).
    *   b. **Consulting Stack Profile:** Use `read_file` to check `.context/stack_profile.json` for relevant project technologies and team familiarity.
    *   c. **Consulting Mode Summary:** Review `kb-available-modes-summary.md` (in KB) or use system knowledge to identify potential modes.
    *   d. **Matching Tags:** Prioritize specialists whose `tags` (from their `.mode.md`) align closely with the task requirements and Stack Profile.
    *   e. **Prioritizing Specificity:** Prefer specific specialists (e.g., `framework-react`) over generalists (e.g., `dev-general`) if a good match exists.
    *   f. **Logging Rationale:** Justify the specialist choice according to Rule `12`.
3.  **Determine Delegation Method (Simple vs. MDTM):**
    *   **Use MDTM Workflow If:** (Consult KB `04-delegation-mdtm.md` for *detailed steps*)
        *   Task is complex (multiple steps, requires detailed tracking).
        *   Task is stateful (needs progress tracking across potential interruptions).
        *   Task is high-risk (e.g., modifying core logic, infrastructure, security).
        *   Task requires clear handoffs or auditable progress.
        *   Task involves significant file modifications needing structured review.
    *   **Use Simple `new_task` If:**
        *   Task is straightforward, single-step.
        *   Task is read-only (e.g., context gathering, analysis).
        *   Task is low-risk.
4.  **Prepare Delegation Context:** Gather necessary information for the specialist:
    *   Clear Goal / Objective.
    *   Specific Acceptance Criteria.
    *   Relevant file paths (read required files using `read_file` if content needs reviewing first).
    *   Path to the Stack Profile (`.context/stack_profile.json`).
    *   References to relevant ADRs (`.decisions/`), planning docs (`.planning/`), or parent/related Task IDs.
    *   Your coordinating Task ID (`TASK-CMD-...`).
5.  **Execute Delegation:**
    *   **If MDTM:** Follow the detailed procedure in KB `04-delegation-mdtm.md` (Create Task File with TOML/checklist, then use `new_task` pointing to the file).
    *   **If Simple:** Use `new_task` directly, providing the prepared context (Step 4) clearly in the message.
        *   *Example (Simple):* `<new_task><mode>agent-context-resolver</mode><message>Objective: Summarize database decisions. Acceptance: Provide bullet points covering key choices and rationale. Context: See .decisions/. Coordinator Task: TASK-CMD-...</message></new_task>`
6.  **Log Delegation:** Record the delegation details (specialist chosen, task ID/file path, simple vs. MDTM, rationale) according to Rule `12`.
7.  **Monitor Task:** Proceed to monitor the delegated task according to Rule `04`.

**Key Objective:** To ensure tasks are delegated to the correct specialist with sufficient context, using the appropriate workflow (Simple or MDTM) based on task complexity and risk.