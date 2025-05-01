# --- Basic Metadata ---
id = "KB-RC-CONCEPTS-MULTIPROJECT"
title = "Core Concept: Multi-Project Workspaces"
status = "draft"
doc_version = "1.0" # Version of the multi-project concept
content_version = 1.0
audience = ["users", "developers", "architects", "project_managers", "contributors", "ai_modes"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md"
tags = ["roo-commander", "intellimanage", "core-concept", "multi-project", "workspace", "architecture", "configuration", "context"]
related_docs = [
    "../README.md", # Link to the KB README
    "01_File_System_Structure.md",
    "06_IntelliManage_Artifacts.md",
    "../../../DOC-FS-SPEC-001.md", # Link to the formal specification
    "../../../DOC-ARCH-001.md", # Link to Architecture Overview
    "../../../DOC-UI-CMD-SPEC-001.md" # Link to Command Spec
    ]
+++

# Core Concept: Multi-Project Workspaces

## 1. Introduction / Overview 🎯

A core design principle of IntelliManage and Roo Commander is the native support for managing **multiple distinct projects** within a single development workspace (e.g., a single VS Code window managing a monorepo or several related repositories).

This allows teams to handle interconnected projects (like a frontend application, a backend API, and a shared component library) using a unified management framework, all stored within the `.ruru/projects/` directory.

## 2. File System Structure for Multiple Projects 📂

As detailed in the File System Structure guide (`01_File_System_Structure.md`), the foundation for multi-project support lies in the directory organization:

```
WORKSPACE_ROOT/
└── .ruru/
    └── projects/
        ├── projects_config.toml  # Optional: Workspace-level config
        ├── frontend-app/         # 👈 Project 1
        │   ├── epics/
        │   ├── features/
        │   ├── tasks/
        │   └── project_config.toml # Config for frontend-app
        │
        ├── backend-api/          # 👈 Project 2
        │   ├── epics/
        │   ├── features/
        │   ├── tasks/
        │   └── project_config.toml # Config for backend-api
        │
        └── shared-library/       # 👈 Project 3
            ├── tasks/
            └── project_config.toml # Config for shared-library
```

*   Each project resides in its own subdirectory under `.ruru/projects/`, named with a unique `[project_slug]`.
*   Each project contains its own set of artifact directories (`epics/`, `features/`, `tasks/`, etc.) and its own `project_config.toml` file.

## 3. Project Configuration Files ⚙️

*   **`project_config.toml` (Required per Project):** Located inside each project's directory (e.g., `.ruru/projects/frontend-app/project_config.toml`). This file defines settings specific to *that project*, such as:
    *   `project_name` (Human-readable)
    *   `project_slug` (Matches directory name)
    *   `methodology` (Scrum, Kanban, Custom, None)
    *   Custom statuses, WIP limits, sprint definitions (depending on methodology)
    *   GitHub integration settings (if enabled for this project)
*   **`projects_config.toml` (Optional Workspace Level):** Located directly in `.ruru/projects/`. This file provides workspace-wide context:
    *   `managed_projects = ["slug1", "slug2", ...]` : Helps tools and AI discover all projects within the workspace.
    *   Can define workspace-level defaults (e.g., `default_methodology`) that project configs can override.

## 4. Active Project Context 📌

Since multiple projects can exist, IntelliManage needs to know which project a command or query refers to. This is managed through the **active project context**.

*   **Setting the Active Project:** Use the `!pm set-active <project_slug>` command (see `DOC-UI-CMD-SPEC-001`). This tells IntelliManage which project subsequent commands should target by default within the current session.
    *   *Example:* `!pm set-active frontend-app`
*   **Default Behavior:** If an active project is set, commands like `!pm list tasks` will operate on that project.
*   **Explicit Override:** You can always target a specific project, regardless of the active context, using the `--project <project_slug>` flag in commands.
    *   *Example:* `!pm list tasks --project backend-api --status "Blocked"` (This command targets `backend-api` even if `frontend-app` is the active project).
*   **AI Awareness:** The AI Engine should be aware of the active project context and use it when generating artifacts or interpreting ambiguous user requests.

## 5. Cross-Project Linking & Dependencies (Future Consideration) 🔗

While the core structure supports multiple projects, advanced features like explicitly linking artifacts *between* different projects (e.g., a frontend task depending on a backend feature) require a defined convention.

*   **Proposed Convention:** Use a format like `"project_slug:TYPE-ID"` within fields like `depends_on` or `related_docs`.
    *   *Example:* In `frontend-app`'s `TASK-105.md`, you might have `depends_on = ["backend-api:TASK-020"]`.
*   **Implementation:** This requires the Core Logic Engine and AI Engine to parse these cross-project references correctly when validating dependencies or gathering context. (This is noted as a future consideration in `DOC-FS-SPEC-001`).

## 6. Benefits of Multi-Project Support ✨

*   **Centralized Management:** Manage all related projects from a single interface within your IDE.
*   **Clear Separation:** Keep artifacts and configurations for different projects distinct and organized.
*   **Contextual Operations:** Easily switch context or target specific projects for commands and analysis.
*   **Foundation for Cross-Project Insights:** Enables future capabilities like cross-project roadmaps, dependency visualization, and resource allocation analysis.

## 7. Conclusion ✅

IntelliManage's multi-project architecture provides a flexible and scalable foundation for managing complex software ecosystems within a single workspace. By leveraging distinct project directories, project-specific configurations, and the concept of an active project context, users and AI agents can effectively navigate and manage work across multiple related initiatives.