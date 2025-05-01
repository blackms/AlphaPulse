+++
# --- Basic Metadata ---
id = "KB-RC-CONCEPTS-ARTIFACTS"
title = "Core Concept: IntelliManage Artifacts"
status = "draft"
doc_version = "1.0" # Version of the artifact definitions
content_version = 1.0
audience = ["users", "developers", "architects", "project_managers", "contributors", "ai_modes"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md"
tags = ["roo-commander", "intellimanage", "core-concept", "artifacts", "hierarchy", "tasks", "epics", "features", "stories", "bugs", "initiatives"]
related_docs = [
    "../README.md", # Link to the KB README
    "01_File_System_Structure.md",
    "02_TOML_MD_Standard.md",
    "03_MDTM_Explained.md",
    "../../../../DOC-SCHEMA-001.md", # Link to IntelliManage Schemas
    "../../../../DOC-FS-SPEC-001.md" # Link to File System Spec
    ]
+++

# Core Concept: IntelliManage Artifacts

## 1. Introduction / Overview 🎯

IntelliManage uses a structured set of **artifacts** to represent different levels of work, planning, and decision-making within a project. These artifacts are the core building blocks for managing workflows, tracking progress, and maintaining project context.

All primary artifacts are stored as individual **TOML+MD files** within the designated project directories (`.ruru/projects/[project_slug]/`) as defined in the File System Structure (`01_File_System_Structure.md`).

## 2. The Work Hierarchy ιεα

IntelliManage employs a hierarchical structure to break down work and align it with broader goals:

1.  **🎯 Initiative (Optional):** The highest level, representing a strategic business goal or theme, potentially spanning multiple projects.
2.  **🗺️ Epic:** A large body of work within a specific project, delivering significant value. Often groups multiple related Features.
3.  **🌟 Feature:** A distinct piece of user-facing functionality or capability within an Epic.
4.  **🛠️ Task / ✨ Story / 🐞 Bug:** The primary units of actionable work for the development team, typically sized for completion within an iteration or sprint. They implement Features or fix Bugs.
5.  **☑️ Subtask:** Granular steps defined within the Markdown checklist of a parent Task/Story/Bug. Not a separate file.

This hierarchy provides traceability from high-level strategy down to individual implementation steps.

## 3. Artifact Types Explained 📄

### 3.1. 🎯 Initiative

*   **Purpose:** Defines a high-level strategic objective.
*   **Location:** `.ruru/projects/[project_slug]/initiatives/INIT-ID_description.md`
*   **Key Metadata:** `id`, `title`, `status`, `type="🎯 Initiative"`, `project_name`.
*   **Links:** Can be linked *from* Epics (via `initiative_id` in Epic TOML).

### 3.2. 🗺️ Epic

*   **Purpose:** Represents a significant chunk of work within a project, often spanning multiple sprints/iterations. Groups related Features.
*   **Location:** `.ruru/projects/[project_slug]/epics/EPIC-ID_description.md`
*   **Key Metadata:** `id`, `title`, `status`, `type="🗺️ Epic"`, `project_name`, `initiative_id` (optional).
*   **Links:** Linked *from* Features (via `epic_id` in Feature TOML). May link *to* an Initiative.

### 3.3. 🌟 Feature

*   **Purpose:** Defines a specific user-facing capability or functionality. Groups related Tasks/Stories/Bugs.
*   **Location:** `.ruru/projects/[project_slug]/features/FEAT-ID_description.md`
*   **Key Metadata:** `id`, `title`, `status`, `type="🌟 Feature"`, `project_name`, `epic_id` (Required).
*   **Links:** Linked *from* Tasks/Stories/Bugs (via `feature_id`). Links *to* an Epic.

### 3.4. 🛠️ Task / ✨ Story / 🐞 Bug

*   **Purpose:** Represents the day-to-day work items:
    *   **Task:** A specific technical action needed.
    *   **Story:** A user-centric requirement (e.g., "As a user, I want to...").
    *   **Bug:** A defect or issue to be fixed.
*   **Location:** `.ruru/projects/[project_slug]/tasks/TYPE-ID_description.md` (e.g., `TASK-101_...`, `BUG-005_...`)
*   **Key Metadata:** `id`, `title`, `status`, `type` (Required: Task, Story, Bug, etc.), `project_name`, `feature_id` (Required). Optional: `priority`, `assigned_to`, `estimated_effort`, `sprint_id`, `depends_on`, `related_commits`, `related_prs`, `related_issues`.
*   **Links:** Links *to* a Feature (Required). Can depend on other Tasks/Bugs.
*   **Subtasks:** Defined within the Markdown body using checklists (`- [ ] ...`).

### 3.5. ☑️ Subtask

*   **Purpose:** Represents a small, granular step needed to complete a parent Task/Story/Bug.
*   **Location:** Defined as a checklist item (`- [ ]` or `- [x]`) within the Markdown body of the parent artifact file.
*   **Characteristics:** Does not have its own file or TOML metadata. Status is binary (checked/unchecked).

### 3.6. 🏛️ Architecture Decision Record (ADR)

*   **Purpose:** Records significant architectural or technical decisions, their context, and consequences.
*   **Location:** `.ruru/projects/[project_slug]/decisions/ADR-NNN_description.md`
*   **Format:** Follows the standard ADR template (See `.ruru/templates/toml-md/07_adr.md`).

### 3.7. 📄 Other Documents (Reports, Plans, Context)

*   **Purpose:** Store supporting project information like reports, high-level plans, or project-specific AI context.
*   **Location:** `.ruru/projects/[project_slug]/reports/`, `/planning/`, `/context/`.
*   **Format:** Typically TOML+MD, using appropriate templates (e.g., `09_documentation.md`, `12_postmortem.md`).

## 4. Relationship to Methodologies 🔄📊

While the artifact hierarchy is consistent, how these artifacts are used and transitioned depends on the project's chosen methodology (`project_config.toml`):

*   **Scrum:** Epics/Features form the Product Backlog. Stories/Tasks/Bugs are selected for Sprints (using `sprint_id`). Status transitions follow typical sprint phases.
*   **Kanban:** Focus is on the flow of Tasks/Stories/Bugs through defined `status` columns. Epics/Features provide higher-level grouping.
*   **Custom:** Workflow follows the `custom_statuses` defined in the configuration.

## 5. Conclusion ✅

The IntelliManage artifact system provides a structured yet flexible way to represent project work. Understanding the hierarchy (Initiative > Epic > Feature > Task/Story/Bug > Subtask), the purpose of each artifact type, and their storage location within the `.ruru/projects/` directory is essential for effective project management using Roo Commander and IntelliManage.