+++
# --- Basic Metadata ---
id = "KB-RC-ADV-CUSTOM-INSTRUCTIONS"
title = "Advanced Usage: Custom Instructions & Rules"
status = "draft"
doc_version = "1.0" # Version of this guide
content_version = 1.0
audience = ["users", "developers", "architects", "contributors", "ai_modes"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md"
tags = ["roo-commander", "customization", "rules", "instructions", "configuration", "advanced", "guide", "tutorial", "ai-guidance", "modes"]
related_docs = [
    "../README.md", # Link to the KB README
    "../../02_Core_Concepts/04_Rules_Explained.md", # Link to the core concept doc
    "../../../../.ruru/docs/roo-code/custom-instructions.md", # Link to the base Roo Code doc this enhances
    "../../../../.roo/rules/", # Link to workspace rules dir
    "../../../../.ruru/templates/toml-md/16_ai_rule.md" # Link to the rule template
    ]
difficulty = "intermediate"
estimated_time = "~15-20 minutes"
prerequisites = ["Roo Commander installed", "Understanding of Roo Commander Modes & Architecture", "Basic understanding of TOML and Markdown"]
learning_objectives = ["Understand the purpose and structure of Rule files", "Differentiate between Workspace-Wide and Mode-Specific Rules", "Learn how to create and organize custom rules", "Understand how rules are loaded and applied", "Get tips for writing effective instructions for AI modes"]
+++

# Advanced Usage: Custom Instructions & Rules

## 1. Introduction / Goal 🎯

While each Roo Commander mode has a core `system_prompt` defining its primary role, **Rules** provide a powerful mechanism for adding specific, granular instructions, constraints, and procedures that further shape AI behavior. These rules allow you to customize how modes operate within your specific workspace or project context.

This guide expands on the base Roo Code documentation (`.ruru/docs/roo-code/custom-instructions.md`) and the Core Concept explanation (`02_Core_Concepts/04_Rules_Explained.md`), providing practical details on how to create, organize, and write effective rules.

**Goal:** To enable users to effectively customize and standardize AI agent behavior using workspace-wide and mode-specific rule files.

## 2. Rules vs. System Prompt vs. KB 📜 vs. 🗣️ vs. 📚

It's helpful to clarify the distinct roles:

*   **System Prompt (`.mode.md`):** Defines the mode's core **identity, primary responsibilities, and high-level operational guidelines**. It sets the overall persona and goals.
*   **Rules (`.roo/rules/...`):** Define **specific, actionable procedures, constraints, or standards** that the mode MUST follow during execution. They provide concrete "how-to" instructions or "must-do/must-not-do" constraints. Rules are **always loaded** into the active mode's context.
*   **Knowledge Base (KB) (`.ruru/modes/.../kb/`):** Contains **detailed reference information, complex procedures, or extensive examples**. It is **not** loaded automatically; modes consult it *on demand* based on instructions within their Rules (specifically KB Lookup Rules).

Think of it like this: The System Prompt is the job description, the Rules are the company handbook and standard operating procedures, and the KB is the technical library.

## 3. Rule File Structure & Format ⚙️

*   **Format:** Rules use the standard **TOML+MD** format.
*   **Template:** Use the template `.ruru/templates/toml-md/16_ai_rule.md` as a starting point.
*   **TOML Frontmatter:** Contains metadata about the rule (`id`, `title`, `scope`, `target_audience`, `status`, etc.).
*   **Markdown Body:** Contains the actual rule text, instructions, or procedure steps.

## 4. Workspace-Wide vs. Mode-Specific Rules 🌍 vs. 🏠

Rules can be applied at two levels:

### 4.1. Workspace-Wide Rules

*   **Location:** `.roo/rules/` directory at the workspace root.
*   **Applicability:** Apply globally to **ALL** modes operating within that workspace.
*   **Purpose:** Define fundamental standards, safety protocols, and procedures common to all agents (e.g., TOML+MD format, logging requirements, OS-aware commands, commit message standards, iterative execution policy).
*   **Naming:** Often prefixed with numbers (`NN-`) to influence loading order (lower numbers load earlier in the context).
*   **Example Files:**
    *   `.roo/rules/00-user-preferences.md`
    *   `.roo/rules/01-standard-toml-md-format.md`
    *   `.roo/rules/05-os-aware-commands.md`
    *   `.roo/rules/08-logging-procedure-simplified.md`

### 4.2. Mode-Specific Rules

*   **Location:** `.roo/rules-[mode_slug]/` directory (e.g., `.roo/rules-dev-react/`).
*   **Applicability:** Apply **only** when the mode matching `[mode_slug]` is active.
*   **Purpose:** Define behaviors, procedures, or constraints specific to that mode's role (e.g., instructing `dev-react` on specific component patterns, defining the KB lookup trigger for `util-writer`, outlining the staging workflow for `prime-coordinator`).
*   **Naming:** Can be prefixed with numbers or use descriptive names. A KB lookup rule (e.g., `01-kb-lookup-rule.md`) is highly recommended for modes with KBs.
*   **Example Files:**
    *   `.roo/rules-roo-commander/02-initialization-workflow-rule.md`
    *   `.roo/rules-dev-react/01-kb-lookup-rule.md`
    *   `.roo/rules-prime-txt/01-confirmation-rule.md`

## 5. Rule Loading Order & Precedence ⬇️

When a mode is activated, Roo Code loads rules and injects their content into the LLM context in the following order:

1.  Mode's Base `system_prompt` (from `.mode.md`).
2.  Workspace-Wide Rules (`.roo/rules/`) - Loaded in alphabetical/numerical order.
3.  Mode-Specific Rules (`.roo/rules-[mode_slug]/`) - Loaded in alphabetical/numerical order.

This means mode-specific rules appear later in the context and can refine or override behavior established by workspace rules if carefully worded, but generally, they should complement the base rules.

## 6. Creating Custom Rules ✍️

1.  **Identify Need:** Determine if a specific behavior, constraint, or procedure needs to be consistently enforced for one or all modes.
2.  **Determine Scope:** Decide if the rule should apply workspace-wide (`.roo/rules/`) or only to a specific mode (`.roo/rules-[mode_slug]/`).
3.  **Create File:**
    *   Copy the template: `cp .ruru/templates/toml-md/16_ai_rule.md .roo/rules/[NN-rule-name].md` (or into the mode-specific directory).
    *   Use a descriptive filename, potentially numbered for ordering.
4.  **Define Metadata (TOML):** Fill in the `id`, `title`, `scope`, `target_audience`, `status`, etc.
5.  **Write Rule Content (Markdown):** Clearly and concisely write the rule, instruction, or procedure in the Markdown body. Use formatting (lists, bolding, code blocks) for clarity.
6.  **Test:** Activate the relevant mode(s) and test if they adhere to the new rule in relevant scenarios. Refine the rule wording as needed.

## 7. Tips for Writing Effective Rules ✨

*   **Be Specific & Actionable:** Use clear, unambiguous language. Tell the AI *what* to do or *not* to do. Use imperative verbs (e.g., "MUST", "SHOULD", "AVOID").
*   **Focus:** Keep each rule file focused on a single procedure or constraint.
*   **Reference Other Rules/Docs:** Link to related rules or documentation where necessary for context.
*   **Use Formatting:** Employ Markdown lists, bold text, and code blocks to make rules easier for the AI (and humans) to parse.
*   **Consider Edge Cases:** Think about potential exceptions or alternative scenarios.
*   **Keep it Concise:** While specific, avoid unnecessary verbosity. Rules add to the context window.
*   **Test & Iterate:** The best way to ensure a rule works is to test it with the target AI mode(s) and refine the wording based on their behavior.

## 8. Conclusion ✅

Custom Instructions and Rules provide a powerful way to tailor Roo Commander's behavior to your specific needs and project standards. By understanding the difference between workspace-wide and mode-specific rules, using the standard TOML+MD format, and writing clear, actionable instructions, you can significantly enhance the consistency, safety, and effectiveness of your AI development team.