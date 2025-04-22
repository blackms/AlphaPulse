# Project Structure Inventory

This document serves as a manifest for the hidden folder structure used in this workspace to organize operational artifacts, documentation, planning materials, and other non-source-code assets.

## Root-Level Hidden Folders

*   **`.tasks/`**
    *   **Purpose:** Stores active operational task files, primarily following the Markdown-Driven Task Management (MDTM) standard.
    *   **Managed By:** `project-manager`, `roo-commander`, All modes (updating assigned tasks).
*   **`.decisions/`**
    *   **Purpose:** Stores Architecture Decision Records (ADRs) documenting significant technical choices.
    *   **Managed By:** `technical-architect`, `roo-commander`.
*   **`.context/`**
    *   **Purpose:** Holds generated context summaries, the Stack Profile, user profile information, downloaded source docs for indexing, condensed indices, and other transient or derived context useful for modes.
    *   **Managed By:** `context-resolver`, `discovery-agent`, `context-condenser`, `roo-commander`.
*   **`.docs/`**
    *   **Purpose:** Central repository for relatively stable documentation, standards, guides, knowledge base articles, design artifacts, and diagrams.
    *   **Managed By:** `technical-writer`, `project-manager`, `technical-architect`, `diagramer`, `ui-designer`.
    *   **Subfolders:** `standards/`, `guides/`, `knowledge/`, `reviews/`, `diagrams/`, `designs/`, `security/`, `notes/` (for less formal meeting minutes, follow-ups), etc. (Conventions for adding/using subfolders should be further defined).
*   **`.workflows/`**
    *   **Purpose:** Defines higher-level, multi-phase sequences of activities involving multiple roles/agents (e.g., "New Feature Development Workflow"). Focuses on the overall flow and coordination.
    *   **Managed By:** `roo-commander`, `project-manager`, `technical-architect`.
*   **`.processes/`**
    *   **Purpose:** Defines specific, granular procedures, algorithms, or SOPs often executed within a workflow or by a specific role (e.g., "ACQA Process", "Mode Creation SOP"). Focuses on the detailed "how-to".
    *   **Managed By:** `roo-commander`, relevant specialists, `technical-writer`.
*   **`.templates/`**
    *   **Purpose:** Contains standardized templates for various artifacts (e.g., mode definitions, ADR formats, task files, SOPs, workflow definitions, common code snippets). Includes subfolders like `toml-md/`, `modes/`, potentially `workflows/`, `processes/`.
    *   **Managed By:** `mode-maintainer`, `technical-architect`, `project-manager`, relevant specialists.
*   **`.planning/`**
    *   **Purpose:** Holds high-level project plans, strategic documents, roadmaps, and planning artifacts (like assessment reports).
    *   **Managed By:** `roo-commander`, `project-manager`, `technical-architect`.
*   **`.logs/`**
    *   **Purpose:** Stores output logs from commands executed by modes or build/test processes. Useful for debugging.
    *   **Managed By:** Modes using `execute_command`, `cicd-specialist`, testing modes.
*   **`.reports/`**
    *   **Purpose:** Contains generated reports like test coverage, performance benchmarks, security scans, research summaries, analysis reports, and review feedback.
    *   **Managed By:** Testing modes, `performance-optimizer`, `security-specialist`, `research-context-builder`, `complex-problem-solver`, `code-reviewer`, `second-opinion`.
*   **`.ideas/`**
    *   **Purpose:** A less formal space for brainstorming, future feature ideas, quick notes, and scratchpad content.
    *   **Managed By:** All modes, User.
*   **`.archive/`**
    *   **Purpose:** Location for moving completed or obsolete items from other folders (e.g., old tasks, superseded plans, previous logs) to keep active directories clean.
    *   **Convention:** It is recommended to create subfolders within `.archive/` that mirror the top-level hidden folders (e.g., `.archive/tasks/`, `.archive/decisions/`, `.archive/docs/`) to maintain context about the origin of archived items.
    *   **Managed By:** `project-manager` (potentially automated in future).
*   **`.snippets/`**
    *   **Purpose:** Stores reusable code snippets or configuration examples identified during development.
    *   **Managed By:** All developer/specialist modes, `roo-commander`.

## Subfolder Conventions

*   While top-level hidden folders are defined here, conventions for creating and organizing subfolders within directories like `.docs`, `.context`, `.templates`, etc., should be established and documented (perhaps in `.docs/standards/directory-structure.md`) to maintain clarity as the project grows.

## Mode-Specific Context (`.roo/context/{mode-slug}/`)

*   While not part of the shared project structure above, many modes reference a potential `.roo/context/{mode-slug}/` directory.
*   **Purpose:** Intended to hold knowledge, templates, examples, or documentation *specific* to the operation of that particular mode, rather than general project context.
*   **Status:** These directories and files generally do not exist yet but represent opportunities to enhance mode capabilities by providing them with specialized knowledge bases.

This inventory provides a clear overview of where different types of information should reside within the new structure.