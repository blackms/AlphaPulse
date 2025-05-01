+++
# --- Basic Metadata ---
id = "KB-RC-ADV-PRIME-COORD-USAGE"
title = "Advanced Usage: Using Prime Coordinator"
status = "draft"
doc_version = "1.0" # Version of this guide
content_version = 1.0
audience = ["developers", "architects", "advanced_users", "contributors"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md" # Using general doc template structure
tags = ["roo-commander", "prime-coordinator", "advanced", "configuration", "meta-development", "direct-control", "guide", "tutorial", "workflow"]
related_docs = [
    "../README.md", # Link to the KB README
    "../../../../.ruru/modes/prime-coordinator/prime-coordinator.mode.md", # Link to Prime Coordinator mode definition
    "../../../../.ruru/modes/prime-txt/prime-txt.mode.md", # Link to Prime Text worker
    "../../../../.ruru/modes/prime-dev/prime-dev.mode.md", # Link to Prime Dev worker
    "../../../../.roo/rules-prime-coordinator/02-request-analysis-dispatch.md", # Link to dispatch rule
    "../../../../.roo/rules-prime-coordinator/03-meta-dev-workflow-simplified.md", # Link to meta-dev rule
    "../../../../.roo/rules-prime-coordinator/04-operational-delegation-simplified.md", # Link to operational delegation rule
    "../../04_Understanding_Modes/01_Mode_Roles_Hierarchy.md" # Link to hierarchy explanation
    ]
difficulty = "advanced"
estimated_time = "~15-20 minutes"
prerequisites = ["Roo Commander installed", "Deep understanding of Roo Commander architecture, modes, rules, KBs, TOML+MD format, and file system structure", "Awareness of the risks involved in modifying configuration files"]
learning_objectives = ["Understand the purpose and appropriate use cases for Prime Coordinator", "Differentiate Prime Coordinator from Roo Commander/Session Manager", "Learn how Prime Coordinator handles configuration changes vs. operational tasks", "Understand the safety mechanisms involved (worker confirmation)", "Know how to formulate effective instructions for Prime Coordinator"]
+++

# Advanced Usage: Using Prime Coordinator

## 1. Introduction / Goal 🎯

The **`prime-coordinator`** mode offers a more direct and powerful interface for interacting with the Roo Commander system compared to the standard `roo-commander` or `session-manager`. It's designed for advanced users who need to perform specific configuration changes (meta-development) or execute operational tasks with less conversational overhead.

This guide explains when and how to use `prime-coordinator` effectively and safely.

**Goal:** To enable advanced users to leverage `prime-coordinator` for direct control over configuration modifications and task delegation, while understanding its workflows and safety considerations.

## 2. Prerequisites Checklist ✅

*   [ ] You have a strong understanding of the Roo Commander architecture, including modes, rules, KBs, and the `.roo`/`.ruru` file structure.
*   [ ] You are comfortable with the TOML+MD format used for configuration and artifacts.
*   [ ] You understand the potential impact of modifying mode definitions, rules, or other configuration files.
*   [ ] You know how to formulate clear, specific, and unambiguous instructions for AI agents.

## 3. When to Use `prime-coordinator` (vs. `roo-commander`/`session-manager`) ↔️

*   **Use `session-manager` (Recommended for Daily Use):** For managing your work session, setting high-level goals, and having routine development tasks coordinated efficiently via `roo-dispatch`.
*   **Use `roo-commander`:** For complex planning, initial project onboarding, troubleshooting intricate workflow issues, or when you need more guidance and interaction.
*   **Use `prime-coordinator` (Advanced):**
    *   When you need to **directly modify specific Roo configuration files** (modes, rules, KBs, templates, workflows, processes).
    *   When you want to **delegate a specific operational task directly** to a known specialist mode with minimal planning overhead, assuming you provide clear instructions.
    *   When you need to perform **direct research** using available tools (`browser`, MCPs).

**Key Difference:** `prime-coordinator` assumes you know *exactly* what needs to be done and which file(s) to target or which specialist to use. It performs less independent analysis and asks fewer clarifying questions than `roo-commander`.

## 4. Key Capabilities ✨

*   **Meta-Development Coordination:** Manages requests to edit configuration files (`.mode.md`, `.roo/rules-*/`, `.ruru/modes/*/kb/`, etc.) by delegating to specialized Prime workers (`prime-txt` for Markdown, `prime-dev` for structured files).
*   **Operational Task Delegation:** Can delegate standard development tasks (features, bugs, tests) directly to operational specialists (e.g., `framework-react`, `dev-api`) using `new_task`.
*   **Research Execution:** Can directly use tools like `browser`, `fetch`, or MCP-based research tools to answer user queries.
*   **Mode Querying:** Can delegate read-only analysis tasks to operational modes to gather information.

## 5. Workflow Overview: How Prime Coordinator Handles Requests ⚙️

Based on its rules (`.roo/rules-prime-coordinator/`), `prime-coordinator` analyzes your request and routes it:

1.  **Configuration Change Request (Meta-Development):**
    *   Identifies the target file path.
    *   Determines the appropriate Prime worker (`prime-txt` for `.md`, `prime-dev` for `.mode.md` TOML, `.js`, `.toml`, etc.).
    *   Delegates the specific edit instruction to the worker using `new_task`.
    *   **Crucially:** The worker (`prime-txt` or `prime-dev`) is responsible for **asking the user for confirmation** before applying the change via `write_to_file` or `apply_diff`. This is the primary safety mechanism for direct edits.
    *   `prime-coordinator` awaits the worker's completion report (which includes whether the change was confirmed and applied, rejected by the user, or failed).
2.  **Operational Task Request:**
    *   Analyzes the goal and selects the appropriate *operational* specialist mode (e.g., `util-writer`, `dev-fixer`).
    *   Delegates the task using `new_task`, providing the user's instructions and context. Uses MDTM task files for complex requests if instructed.
    *   Monitors completion via `attempt_completion` from the specialist.
3.  **Research Request:**
    *   Selects and uses appropriate tools (`browser`, `fetch`, MCPs) directly.
    *   Synthesizes and reports findings.

## 6. Usage Examples ⌨️

*   **Activate:** Switch to the `prime-coordinator` mode in Roo Code.

*   **Example 1: Editing an Operational Rule**
    ```prompt
    @prime-coordinator Please update the file `.roo/rules-dev-react/01-kb-lookup-rule.md`. In the Markdown body, change the sentence "ALWAYS consult the dedicated Knowledge Base" to "ALWAYS first consult the dedicated Knowledge Base". Use apply_diff.
    ```
    *(Expected Flow: Prime Coordinator -> delegates to `prime-txt` -> `prime-txt` reads file, prepares diff -> `prime-txt` uses `ask_followup_question` to show diff and ask user "Apply this change to ...?" -> User confirms -> `prime-txt` uses `apply_diff` -> `prime-txt` reports success to Prime Coordinator -> Prime Coordinator reports to User.)*

*   **Example 2: Editing a Mode's KB File**
    ```prompt
    @prime-coordinator Add a new section titled '## Common Pitfalls' to the end of the file `.ruru/modes/framework-nextjs/kb/02-data-fetching.md` with the following content: [Paste Markdown content here].
    ```
    *(Expected Flow: Similar to Example 1, using `prime-txt` with confirmation.)*

*   **Example 3: Editing a Mode's TOML**
    ```prompt
    @prime-coordinator Modify the file `.ruru/modes/util-writer/util-writer.mode.md`. Add "rst" to the `tags` array in the TOML frontmatter.
    ```
    *(Expected Flow: Prime Coordinator -> delegates to `prime-dev` -> `prime-dev` reads file, prepares change -> `prime-dev` uses `ask_followup_question` to show proposed TOML change and ask user "Apply this change to ...?" -> User confirms -> `prime-dev` uses `apply_diff` -> `prime-dev` reports success -> Prime Coordinator reports to User.)*

*   **Example 4: Delegating an Operational Task**
    ```prompt
    @prime-coordinator Delegate this task to `test-e2e`: "Write a Playwright test script in `tests/e2e/login.spec.ts` to verify the user login flow described in FEAT-001."
    ```
    *(Expected Flow: Prime Coordinator -> delegates to `test-e2e` via `new_task` -> `test-e2e` executes -> `test-e2e` reports completion to Prime Coordinator -> Prime Coordinator reports to User.)*

*   **Example 5: Performing Research**
    ```prompt
    @prime-coordinator Use the browser tool to find the official documentation URL for the `react-hook-form` library.
    ```
    *(Expected Flow: Prime Coordinator uses `browser` tool -> Prime Coordinator reports result.)*

## 7. Safety Considerations & Best Practices ⚠️

*   **You Are in Control:** `prime-coordinator` assumes you know what you're doing. Provide clear, precise instructions.
*   **Worker Confirmation:** Rely on the mandatory confirmation step performed by `prime-txt` and `prime-dev` before they write changes to operational files. **Pay attention to these confirmation prompts.** Do not blindly approve if using auto-approve settings without understanding the change.
*   **Protected Paths:** Remember that `prime-coordinator` itself *cannot* directly modify protected core files (like `.roo/rules/` or its own mode files). It *must* use the Prime workers, which enforce confirmation.
*   **Operational Tasks:** Delegating operational tasks bypasses `roo-commander`'s planning. Ensure the task is well-defined.
*   **Review Changes:** Always review changes made via `prime-coordinator` using Git diffs or by examining the files.

## 8. Limitations 🚫

*   **Less Guidance:** Provides significantly less conversational guidance and performs less planning/analysis than `roo-commander`. Expects explicit instructions.
*   **Safety Relies on Confirmation:** The safety of direct edits to operational files hinges on the user carefully reviewing the confirmation prompts from `prime-txt`/`prime-dev`.
*   **No Direct Implementation:** It coordinates and delegates; it does not write code or documentation itself (except for simple research summaries).

## 9. Conclusion ✅

`prime-coordinator` is a powerful tool for advanced users needing direct control over Roo Commander's configuration or wanting a leaner interface for delegating well-defined operational tasks. Use it when you need precise modifications or want to bypass the standard planning phases, but always be mindful of the changes you request and carefully review the confirmation prompts from its worker modes (`prime-txt`, `prime-dev`). For general development, complex planning, or when more guidance is needed, prefer using `session-manager` or `roo-commander`.