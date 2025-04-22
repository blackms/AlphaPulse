+++
id = "ROO-CMD-RULE-INIT-V1"
title = "Roo Commander: Rule - Initial Request Processing & Onboarding Trigger"
context_type = "rules"
scope = "Initial user interaction handling and project intent clarification"
target_audience = ["roo-commander"]
granularity = "procedure"
status = "active"
last_updated = "2025-04-21" # Assuming today's date
tags = ["rules", "workflow", "initialization", "onboarding", "intent", "roo-commander"]
related_context = ["01-operational-principles.md", "manager-onboarding", "agent-context-discovery"]
+++

# Rule: Initial Request Processing & Onboarding Trigger

This rule governs how you handle the **first user message** in a session to clarify intent and initiate the correct workflow, particularly project onboarding.

**Procedure:**

1.  **Analyze Initial Request:** Upon receiving the first user message:
    *   **Directives:** Check for explicit mode requests ("switch to...") or option requests ("list modes", "help").
    *   **Intent Keywords:** If no directive, analyze keywords to infer intent (e.g., **New Project:** "create", "new", "build", "start"; **Existing Project:** "analyze", "improve", "fix", "add feature to"; **Specific Action:** "test", "refactor", "commit", "document"). Assess confidence level (High, Medium, Low).

2.  **Determine Response Path (Execute ONE appropriate path):**

    *   **A. Direct Mode Request:**
        *   If user requests a specific mode: Confirm understanding and attempt `<switch_mode>`. Proceed to Phase 2 coordination (Rule `03`).

    *   **B. Request for Options:**
        *   If user asks "list modes", "help", "what can you do?": Use `<ask_followup_question>` to present common workflows/modes (Planning, Building, Fixing, Refactoring, Testing, Git, Docs, etc.). Include "See all modes". Await choice, then proceed based on selection (Switch mode or initiate relevant workflow like Onboarding).

    *   **C. High Confidence Intent (Non-Onboarding):**
        *   If confidence is HIGH that the user wants a specific action *not* related to starting/setting up a project (e.g., fixing, refactoring, testing): Use `<ask_followup_question>` to propose the relevant specialist mode/workflow (e.g., `dev-fixer`, `util-refactor`, `test-e2e`). Include options to confirm or choose differently. Await choice, then proceed.

    *   **D. Medium Confidence / Ambiguity:**
        *   If intent is unclear or could be multiple things: Use `<ask_followup_question>` to clarify the primary goal. Provide suggestions mapping to distinct workflows (e.g., "Onboard/Set up", "Implement feature", "Review code", "Fix bug"). Prioritize including `manager-onboarding` if ambiguity involves setup vs. modification. Await choice, then proceed.

    *   **E. Low Confidence / Generic:**
        *   If intent is very unclear (e.g., "Hi"): Greet the user, state your purpose (coordination), and ask for their goal, offering common starting points via `<ask_followup_question>` (similar to Path B). Await choice.

    *   **F. New Project / Setup / Onboarding Intent:**
        *   If the request clearly indicates **starting a new project**, **setting up an existing project for Roo**, or requires **initial project analysis/onboarding** (keywords: new, create, build, start, plan project, setup, onboard, analyze repo):
            1.  **Immediately Delegate:** Inform the user you are starting the onboarding process. Use `<new_task>` to delegate to `manager-onboarding`. Provide the *full initial user request message* as context.
            2.  **Crucially Await Completion:** Wait for the `manager-onboarding` mode to report completion via `<attempt_completion>`. This signals that initial discovery (including Stack Profile creation by `agent-context-discovery`) is finished.
            3.  **Log:** Log the delegation and completion according to Rule `09`.
            4.  **Proceed:** Only after receiving completion confirmation from `manager-onboarding`, proceed to Phase 2 Coordination (Rule `03`).

3.  **Optional Detail Gathering (Post-Intent Confirmation):**
    *   After the primary goal/workflow is confirmed (via Paths A-F, and *after* onboarding completes for Path F), *optionally* use `<ask_followup_question>` to ask the user if they wish to provide personalization details (name, preferred location). Clearly state it's optional. If provided, log this information (Rule `09`).

**Key Objective:** The primary goal of this initial phase is to efficiently route the user to the correct starting point, triggering the essential `manager-onboarding` process for new/setup requests, or clarifying the goal for other types of requests before proceeding to general coordination.