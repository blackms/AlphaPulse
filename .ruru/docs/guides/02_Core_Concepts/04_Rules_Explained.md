+++
# --- Basic Metadata ---
id = "KB-RC-CONCEPTS-RULES"
title = "Core Concept: Rules Explained"
status = "draft"
doc_version = "1.0" # Version of the rules concept being described
content_version = 1.0
audience = ["developers", "architects", "contributors", "ai_modes"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md"
tags = ["roo-commander", "intellimanage", "core-concept", "rules", "configuration", "architecture", "ai-guidance", "modes"]
related_docs = [
    "../README.md", # Link to the KB README
    "01_File_System_Structure.md",
    "02_TOML_MD_Standard.md",
    "../../../../.roo/rules/", # Link to the actual rules directory
    "../../../../.roo/rules/01-standard-toml-md-format.md", # Example core rule
    "../../../../.roo/rules-roo-commander/01-operational-principles.md" # Example mode-specific rule
    ]
+++

# Core Concept: Rules Explained

## 1. Introduction / Overview 🎯

Rules are a fundamental component of the Roo Commander framework, providing specific instructions, constraints, and procedures that guide the behavior of AI agents (modes). They act as a configurable layer of logic that complements the core system prompt defined within each mode's `.mode.md` file.

Rules are stored as structured **TOML+MD** files within the `.roo/` directory structure.

## 2. Purpose of Rules 📜

Rules serve several critical functions:

*   **Standardization:** Enforce consistent procedures across the workspace (e.g., commit message format, logging requirements, file format standards).
*   **Guidance:** Provide step-by-step instructions for specific workflows or actions (e.g., MDTM task creation, error handling).
*   **Constraints:** Define boundaries and safety protocols (e.g., OS-aware command generation, confirmation for risky actions).
*   **Contextual Logic:** Define when and how modes should consult their Knowledge Bases (KBs).
*   **Modularity:** Allow specific behaviors to be defined and updated separately from the main mode prompts.

## 3. Rule Structure & Location 📂

Rules reside within the `.roo/` directory at the workspace root:

```
WORKSPACE_ROOT/
├── .roo/
│   ├── rules/                  # 👈 Workspace-Wide Rules
│   │   ├── 00-user-preferences.md
│   │   ├── 01-standard-toml-md-format.md
│   │   └── ... (Other global rules)
│   │
│   ├── rules-roo-commander/    # 👈 Mode-Specific Rules (Example)
│   │   ├── 01-operational-principles.md
│   │   ├── 99-kb-lookup-rule.md
│   │   └── ...
│   │
│   ├── rules-dev-react/        # 👈 Mode-Specific Rules (Example)
│   │   └── 01-kb-lookup-rule.md
│   │
│   └── ... (Other mode-specific rule directories)
│
└── .ruru/
    └── ...
```

*   **Workspace-Wide Rules (`.roo/rules/`):**
    *   These rules apply **globally** to *all* modes operating within the workspace.
    *   They define fundamental standards and procedures (e.g., file formats, logging, OS-aware commands, safety protocols).
    *   Files are typically prefixed with numbers (`NN-`) to control loading order (lower numbers load first).
*   **Mode-Specific Rules (`.roo/rules-[mode_slug]/`):**
    *   These rules apply **only** to the specific mode indicated by the directory name (e.g., rules in `.roo/rules-dev-react/` only apply when the `dev-react` mode is active).
    *   They define behavior specific to that mode's role (e.g., KB lookup triggers, specific workflow steps, error handling nuances).
    *   Typically includes a KB lookup rule (`01-kb-lookup-rule.md` or `99-kb-lookup-rule.md`).

## 4. Rule Format & Loading ⚙️

*   **Format:** Rules use the standard **TOML+MD** format (`02_TOML_MD_Standard.md`).
    *   The TOML frontmatter defines metadata like `id`, `title`, `scope`, `target_audience`.
    *   The Markdown body contains the actual rule text, instructions, or procedure steps.
*   **Loading:** When Roo Code activates a mode, it loads:
    1.  All rules from the workspace-wide `.roo/rules/` directory (in alphabetical/numerical order).
    2.  All rules from the mode-specific `.roo/rules-[mode_slug]/` directory (if it exists, also in order).
*   **Context Injection:** The content of the loaded rule files (both TOML metadata and Markdown body) is injected into the LLM's context, typically appended to the mode's base system prompt. This provides the AI with its operational instructions.

## 5. Key Rule Types (Examples)

*   **Standard Definitions:** Define core formats or conventions (e.g., `01-standard-toml-md-format.md`, `07-git-commit-standard-simplified.md`).
*   **Procedural Rules:** Outline step-by-step workflows (e.g., `04-mdtm-workflow-initiation.md`, `05-error-handling-rule.md`).
*   **Constraint Rules:** Enforce specific limitations or safety checks (e.g., `05-os-aware-commands.md`, `07-safety-protocols-rule.md`).
*   **KB Lookup Rules:** Define when and how a mode should consult its specific Knowledge Base (e.g., `99-kb-lookup-rule.md` in mode-specific directories).

## 6. Conclusion ✅

Rules are the operational backbone of Roo Commander modes. They provide explicit instructions and constraints that ensure consistent, reliable, and safe behavior from AI agents. By separating standard procedures and mode-specific logic into these structured files, the framework becomes more maintainable, customizable, and transparent. Understanding the location, format, and purpose of rules is essential for effectively using and potentially customizing Roo Commander.