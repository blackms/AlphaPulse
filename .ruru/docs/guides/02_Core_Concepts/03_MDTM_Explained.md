+++
# --- Basic Metadata ---
id = "KB-RC-CONCEPTS-MDTM"
title = "Core Concept: Markdown-Driven Task Management (MDTM)"
status = "draft"
doc_version = "1.0" # Version of the MDTM concept being described
content_version = 1.0
audience = ["users", "developers", "architects", "contributors", "ai_modes"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md"
tags = ["roo-commander", "intellimanage", "core-concept", "mdtm", "tasks", "workflow", "project-management", "toml-md"]
related_docs = [
    "../README.md", # Link to the KB README
    "01_File_System_Structure.md",
    "02_TOML_MD_Standard.md",
    "../../../../.ruru/docs/standards/mdtm_standard.md", # Link to the detailed MDTM standard
    "../../../../.roo/rules/04-mdtm-workflow-initiation.md", # Link to the MDTM creation rule
    "../../../templates/toml-md/01_mdtm_feature.README.md", # Example Task Template Schema
    "../../../templates/toml-md/02_mdtm_bug.README.md" # Example Bug Template Schema
    ]
+++

# Core Concept: Markdown-Driven Task Management (MDTM)

## 1. Introduction / Overview 🎯

**Markdown-Driven Task Management (MDTM)** is the system used by Roo Commander and IntelliManage to define, track, and manage individual work items like Features, Tasks, User Stories, and Bugs. It's a file-based approach designed for seamless integration into the developer workflow.

Instead of relying on external web-based tools, MDTM uses **structured text files** stored directly within your project repository as the single source of truth for task details and status.

## 2. Core Idea: Tasks as Files 📄

The fundamental idea behind MDTM is that each significant work item (Feature, Task, Story, Bug) is represented by its own dedicated file, typically located within the project's task directory structure (e.g., `.ruru/projects/[project_slug]/tasks/` or `.ruru/projects/[project_slug]/features/` for Feature definitions).

These files utilize the **TOML+Markdown (TOML MD)** standard (see `02_TOML_MD_Standard.md`):
*   **TOML Frontmatter (`+++`):** Contains machine-readable metadata like ID, title, status, type, priority, assignee, links (parents, dependencies), tags, dates.
*   **Markdown Body:** Contains human-readable details like the description, acceptance criteria (using checklists), implementation notes, subtask checklists, diagrams, and logs.

## 3. Why MDTM? Benefits ✅

*   **Git-Native:** Tasks live alongside your code in Git, providing version history, branching capabilities, and easy diffing for changes.
*   **Context-Rich:** Task files contain all relevant details, reducing the need to switch between tools for context. AI agents can directly read task files for instructions.
*   **Human & AI Readable:** Both developers and AI assistants can easily read and parse the TOML metadata and Markdown content.
*   **Traceability:** Links between tasks, features, epics, commits, and PRs can be embedded directly in the files or commit messages.
*   **Offline Access:** Tasks are accessible even without an internet connection.
*   **Customizable:** While based on standards, the structure can be adapted with custom fields or templates.

## 4. Structure & Lifecycle 🔄

*   **Location:** Typically under `.ruru/projects/[project_slug]/tasks/` (for Tasks, Stories, Bugs) or `/features/` or `/epics/` etc. (See `01_File_System_Structure.md`).
*   **Naming:** Follows the `TYPE-ID_description.md` convention (e.g., `TASK-101_implement-login.md`).
*   **Templates:** Standard templates exist for different task types (Feature, Bug, Chore, etc.) in `.ruru/templates/toml-md/`.
*   **Status Lifecycle:** Tasks progress through defined statuses (e.g., `🟡 To Do`, `🔵 In Progress`, `🟣 Review`, `🟢 Done`, `🚧 Blocked`, `🧊 Archived`), which are updated in the TOML frontmatter. The specific allowed statuses and transitions depend on the project's configured methodology (Scrum, Kanban, Custom).
*   **Subtasks:** Granular steps are tracked using Markdown checklists (`- [ ]`, `- [x]`) within the main task file's body.

## 5. Interaction with MDTM ⌨️🤖

*   **Manual:** Users can directly create, edit, and update MDTM files using their text editor.
*   **Commands (`!pm ...`):** The IntelliManage `!pm` commands provide a structured way to interact with MDTM files (create, list, update status, add subtasks, etc.). See `DOC-UI-CMD-SPEC-001`.
*   **AI Assistance:** The AI Engine can:
    *   Generate draft MDTM files based on user goals.
    *   Suggest status updates based on context.
    *   Update TOML metadata or Markdown content based on commands.
    *   Parse acceptance criteria and subtasks.
*   **Specialist Modes:** When delegated an MDTM task (e.g., `framework-react` receives `TASK-101_implement-login.md`), the specialist mode reads the file for requirements and **updates the file directly** (checklist items, status in TOML) upon completion or if blocked.

## 6. Conclusion

MDTM provides a powerful, developer-centric approach to task management within Roo Commander and IntelliManage. By leveraging structured text files under version control, it offers unparalleled integration, traceability, and context awareness, enhanced by AI assistance for common operations. Understanding the TOML+MD structure and the standard lifecycle is key to utilizing the system effectively.