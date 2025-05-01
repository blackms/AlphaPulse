+++
# --- Basic Metadata ---
id = "KB-RC-CONCEPTS-KB"
title = "Core Concept: Knowledge Bases (KBs) Explained"
status = "draft"
doc_version = "1.0" # Version of the KB concept being described
content_version = 1.0
audience = ["users", "developers", "architects", "contributors", "ai_modes"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md"
tags = ["roo-commander", "intellimanage", "core-concept", "knowledge-base", "kb", "context", "modes", "rules", "architecture"]
related_docs = [
    "../README.md", # Link to the KB README
    "01_File_System_Structure.md",
    "04_Rules_Explained.md",
    "../../../../.roo/rules-roo-commander/99-kb-lookup-rule.md" # Example KB Lookup Rule
    ]
+++

# Core Concept: Knowledge Bases (KBs) Explained

## 1. Introduction / Overview 🎯

Within the Roo Commander framework, each specialized AI agent (mode) possesses its own dedicated **Knowledge Base (KB)**. This KB serves as a repository for detailed, mode-specific information that goes beyond the core operational instructions found in the mode's rules or system prompt.

KBs are crucial for providing deep, contextual knowledge *on demand* without overloading the primary LLM context window during every interaction.

## 2. Purpose of Knowledge Bases 🧠

Mode-specific KBs are designed to store:

*   **Detailed Procedures:** Step-by-step instructions for complex or less frequent tasks specific to the mode's function (e.g., the detailed MDTM workflow steps for `roo-commander`, specific API integration patterns for `spec-openai`).
*   **Reference Material:** Lists, tables, code snippets, configuration examples, API references, or summaries relevant to the mode's domain (e.g., list of available modes for `roo-commander`, common error codes for `dev-fixer`).
*   **Best Practices & Patterns:** Curated guidelines and recommended approaches for the mode's area of expertise (e.g., React component patterns for `dev-react`, security best practices for `lead-security`).
*   **Project-Specific Context:** Notes or guidelines tailored to how a mode should operate within a specific project (though often managed via workspace/project rules or IntelliManage artifacts).
*   **Synthesized Context:** AI-generated summaries derived from external documentation or other sources (e.g., output from `agent-context-condenser` or the Context7 enrichment workflow).

## 3. KB Structure & Location 📂

*   **Location:** Each mode's KB resides in a dedicated `kb/` subdirectory within its mode definition directory.
    *   **Path:** `.ruru/modes/[mode_slug]/kb/`
    *   *Example:* `.ruru/modes/dev-react/kb/` contains the KB for the React Specialist.
*   **Content:** KBs typically contain multiple Markdown files (`.md`), often using the TOML+MD standard for structure, especially for procedural documents. Files may be organized into subdirectories for clarity.
*   **`README.md`:** Each `kb/` directory **SHOULD** contain a `README.md` file that acts as an index, briefly describing the purpose of each document within the KB. This helps both humans and AI agents quickly find relevant information.
*   **Index Files (`index.toml`, `_index.json`):** Some KBs, particularly those populated by automated processes (like context enrichment), may contain index files (`index.toml` or `_index.json`) that provide a structured list of the KB's contents for programmatic access.

## 4. Rules vs. Knowledge Base: The Distinction 📜 vs. 📚

It's important to understand the difference between Rules (`.roo/rules-*/`) and Knowledge Bases (`.ruru/modes/*/kb/`):

*   **Rules:** Define *when* and *how* a mode should operate at a fundamental level. They are *always* loaded into the AI's context for the active mode. They often *trigger* KB lookups. Rules should be concise and focused on core logic and standard procedures. (See `04_Rules_Explained.md`).
*   **Knowledge Base:** Contains the *detailed information* referenced by the rules or needed for specific, complex tasks. KBs are *not* automatically loaded into context. They are accessed *on demand* when a rule (specifically a KB Lookup Rule) instructs the mode to consult them for specific information.

**Analogy:** Think of Rules as the main flowchart for a process, and the KB as the detailed manual referenced at specific steps in the flowchart.

## 5. KB Lookup Mechanism 🔍

*   **Trigger:** Modes don't automatically read their entire KB. Access is triggered by specific **KB Lookup Rules** (e.g., `.roo/rules-[mode_slug]/99-kb-lookup-rule.md`).
*   **Process:** When a KB lookup rule is active and triggered by the situation:
    1.  The mode identifies the relevant document(s) within its `kb/` directory based on the task and the KB's `README.md`.
    2.  It uses the `read_file` tool (or equivalent MCP tool) to access the content of the specific KB document(s).
    3.  It integrates the information read from the KB into its current task execution or response generation.

## 6. Benefits of KBs ✨

*   **Context Efficiency:** Keeps detailed reference material out of the main prompt, saving valuable LLM context window space for the immediate task.
*   **Deep Knowledge:** Allows modes to access much more detailed information than could feasibly fit in their system prompt or rules.
*   **Maintainability:** Specific procedures or reference data can be updated within the KB without modifying the core mode definition or rules.
*   **Structured Learning:** Provides a dedicated place for curated, mode-specific knowledge, improving consistency and accuracy.

## 7. Conclusion ✅

Knowledge Bases are a vital part of the Roo Commander architecture, enabling modes to access deep, specialized information precisely when needed. By separating detailed reference material from core operational rules and leveraging a rule-driven lookup mechanism, KBs allow for powerful, context-aware AI agents that remain efficient in their use of the LLM's context window. Maintaining well-structured and up-to-date KBs is essential for optimizing the performance and capabilities of Roo Commander modes.