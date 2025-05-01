+++
# --- Basic Metadata ---
id = "KB-RC-CONCEPTS-FS"
title = "Core Concept: IntelliManage File System Structure"
status = "draft"
doc_version = "1.0" # Version of the structure being described
content_version = 1.0
audience = ["users", "developers", "architects", "contributors", "ai_modes"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md"
tags = ["roo-commander", "intellimanage", "core-concept", "file-system", "structure", "architecture", "multi-project", "naming-convention"]
related_docs = [
    ".ruru/docs/guides/README.md", # Link to the KB README
    "../../../DOC-FS-SPEC-001.md", # Link to the formal specification
    "../../../DOC-ARCH-001.md" # Link to Architecture Overview
    ]
+++

# Core Concept: IntelliManage File System Structure

## 1. Introduction / Overview 🎯

IntelliManage uses a specific, standardized file system structure within your workspace to store all project management artifacts and configurations. This structure is designed for clarity, consistency, automated processing, multi-project support, and integration with version control (Git).

Understanding this structure is key to navigating your project data and interacting effectively with IntelliManage commands and AI agents. All IntelliManage data resides within the `.ruru/projects/` directory at the root of your workspace.

**Source of Truth:** This document summarizes the structure defined in the formal **File System Structure Specification (`DOC-FS-SPEC-001`)**.

## 2. Core Principles 💡

*   **Centralized:** All project management data lives under `.ruru/projects/`.
*   **Multi-Project:** Designed to contain multiple independent projects side-by-side.
*   **Hierarchical:** Directory structure reflects the work breakdown (Initiative > Epic > Feature > Task).
*   **Discoverable:** Consistent naming makes artifacts easy to find.
*   **Version Controlled:** The entire `.ruru/projects/` directory should be committed to Git.

## 3. Workspace Root Structure (`.ruru/projects/`) 🌳

The top level contains configuration and directories for each managed project:

```
WORKSPACE_ROOT/
├── .ruru/
│   ├── projects/             # 👈 **Main IntelliManage Directory**
│   │   ├── projects_config.toml # Optional: Workspace-level config (lists projects)
│   │   ├── [project_slug_1]/   # 👈 Project 1 Directory
│   │   │   └── ... (See Project Structure below)
│   │   ├── [project_slug_2]/   # 👈 Project 2 Directory
│   │   │   └── ...
│   │   └── ...                 # Other project directories
│   │
│   └── ...                   # (Other .ruru subdirectories like modes, templates)
│
└── ...                       # (Other workspace files like src/, docs/)
```

*   **`.ruru/projects/`**: The root for all IntelliManage data.
*   **`projects_config.toml` (Optional):** Lists managed projects (`managed_projects = ["slug1", "slug2"]`) and can define workspace-wide defaults.
*   **`[project_slug]/`**: A unique directory for each project (e.g., `frontend-app`, `backend-api`).

## 4. Project Directory Structure (`.ruru/projects/[project_slug]/`) 📂

Each project follows this standard internal structure:

```
.ruru/projects/[project_slug]/
├── initiatives/          # Stores Initiative files (.md)
│   └── INIT-001_example.md
├── epics/                # Stores Epic files (.md)
│   └── EPIC-001_example.md
├── features/             # Stores Feature files (.md)
│   └── FEAT-001_example.md
├── tasks/                # Stores Task, Story, Bug files (.md)
│   ├── TASK-001_example.md
│   └── BUG-002_example.md
├── decisions/            # Stores Architecture Decision Records (.md)
│   └── ADR-001_example.md
├── reports/              # Stores generated/manual reports (.md, .csv)
│   └── sprint-1_summary.md
├── planning/             # Stores high-level plans, roadmaps (.md)
│   └── roadmap_q3.md
├── context/              # Optional: Project-specific context files
│   └── api_conventions.md
├── archive/              # Optional: For completed/archived items (mirrors structure)
│   └── tasks/
│       └── TASK-000_archived.md
└── project_config.toml   # REQUIRED: Project-specific settings (name, methodology, etc.)
```

*   **Artifact Directories (`initiatives/`, `epics/`, `features/`, `tasks/`):** Contain the core work items as TOML+MD files.
*   **Supporting Directories (`decisions/`, `reports/`, `planning/`, `context/`):** Store related project documentation and metadata.
*   **`archive/` (Optional):** A place to move completed/closed items to keep active directories cleaner. Should mirror the main structure.
*   **`project_config.toml` (Required):** Defines the project's name, methodology (Scrum, Kanban, Custom, None), and other specific settings.

## 5. File Naming Conventions 🏷️

Consistent naming helps identify artifacts quickly.

*   **Artifact Files (`.md`):** `TYPE-ID_short-description.md`
    *   **`TYPE`:** `INIT`, `EPIC`, `FEAT`, `TASK`, `BUG` (Story uses TASK prefix typically).
    *   **`ID`:** Unique numerical ID within the project (e.g., `001`, `105`).
    *   **`short-description`:** Lowercase, hyphenated summary.
    *   *Examples:* `EPIC-001_user-auth.md`, `TASK-101_implement-login-form.md`, `BUG-042_null-pointer-exception.md`
*   **Decision Files (ADRs):** `ADR-NNN_short-description.md`
*   **Configuration Files:** `projects_config.toml` (workspace), `project_config.toml` (project).
*   **Other Documents:** Descriptive, lowercase, hyphenated names (e.g., `roadmap_q3-2025.md`).

## 6. Conclusion ✅

This standardized file system structure provides the backbone for IntelliManage. It ensures that project data is organized logically, supports multiple projects, integrates with version control, and is easily navigable by both humans and the AI agents within Roo Commander. Adhering to this structure is essential for the smooth operation of the framework.