+++
# --- Basic Metadata ---
id = "KB-RC-INTRO-INTELLIMANAGE"
title = "Roo Commander: IntelliManage Integration Overview"
status = "draft"
doc_version = "1.0" # Version of the integration concept
content_version = 1.0
audience = ["users", "developers", "architects", "contributors", "ai_modes"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md"
tags = ["roo-commander", "introduction", "intellimanage", "integration", "architecture", "workflow", "project-management", "modes", "ai"]
related_docs = [
    "../README.md", # Link to the KB README
    "01_What_is_Roo_Commander.md",
    "03_Architecture_Overview.md",
    "../../02_Core_Concepts/06_IntelliManage_Artifacts.md",
    "../../05_Using_IntelliManage_Features/", # Link to the main IntelliManage usage section
    "../../../../DOC-ARCH-001.md", # IntelliManage Architecture Spec
    "../../../../DOC-AI-SPEC-001.md" # IntelliManage AI Spec
    ]
+++

# Roo Commander: IntelliManage Integration Overview

## 1. Introduction / Purpose 🎯

Roo Commander is designed to work seamlessly with the **IntelliManage** project management framework. IntelliManage provides the underlying structure for defining, organizing, and tracking project work (Initiatives, Epics, Features, Tasks, etc.) directly within the workspace's `.ruru/projects/` directory.

This document provides an overview of how Roo Commander modes **interact with and leverage** the IntelliManage system to orchestrate development workflows, manage tasks, and utilize project context effectively.

## 2. The Role of IntelliManage Artifacts 📄

IntelliManage artifacts (stored as TOML+MD files) serve as the **structured source of truth** for project work items. Roo Commander modes interact with these artifacts in various ways:

*   **Context:** Reading artifact files (`read_file`) provides detailed requirements, acceptance criteria, status, and relationships for tasks being delegated or executed.
*   **Task Definition:** Coordinating modes (`session-manager`, `roo-commander`) often trigger the creation of new IntelliManage artifacts (e.g., Features, Tasks) based on user goals.
*   **Status Tracking:** The `status` field within artifact TOML frontmatter is the primary indicator of progress, updated via IntelliManage commands (`!pm update ...`) or AI suggestions.
*   **Linking:** Hierarchical (`epic_id`, `feature_id`) and dependency (`depends_on`) links within artifact TOML enable traceability and workflow management.
*   **Reporting:** Reporting features (`!pm report ...`) aggregate data *from* these artifacts.

## 3. Interaction Patterns by Mode Type 🔄

### 3.1. Coordinating Modes (`session-manager`, `roo-dispatch`, `roo-commander`)

These modes orchestrate work and interact most directly with the IntelliManage system *as a whole*:

*   **Reading Configuration:** They read `project_config.toml` files to understand the active project's methodology (Scrum, Kanban, etc.) and custom settings, adapting their behavior accordingly.
*   **Triggering Core Logic:** They translate user requests or workflow steps into actions for the IntelliManage Core Logic Engine (CLE). This is often done by formulating and executing `!pm` commands conceptually, even if the command isn't explicitly shown to the user (e.g., deciding to create a task internally triggers the equivalent of `!pm create task ...`).
*   **Artifact Creation/Delegation:** They initiate the creation of new IntelliManage artifacts (Epics, Features, Tasks) and delegate their implementation or execution. The delegation message (`new_task`) typically includes the relevant IntelliManage artifact ID (e.g., `TASK-123`) for context.
*   **Context Gathering:** They query the CLE (or use `agent-context-resolver`) to read artifact details (`!pm show ...`, `!pm list ...`) needed for planning, decision-making, or providing context to other modes.
*   **Status Monitoring:** They monitor the `status` of delegated tasks by querying the CLE or reading task files.
*   **Reporting:** They trigger report generation (`!pm report ...`) based on user requests or workflow needs.

### 3.2. Specialist & Worker Modes (`dev-*`, `framework-*`, `util-*`, etc.)

These modes execute specific development, testing, or utility tasks and interact with IntelliManage artifacts primarily as **context providers** and **status update targets**:

*   **Receiving Context:** They receive IntelliManage artifact IDs (e.g., `TASK-123`, `FEAT-005`) as part of their task delegation message from a coordinator.
*   **Reading Artifact Details:** They **MUST** use `read_file` (or equivalent MCP tool) to read the specified artifact file (e.g., `.ruru/projects/[project_slug]/tasks/TASK-123.md`) to understand the full requirements, acceptance criteria, and any subtasks defined in the Markdown body.
*   **Updating Subtasks:** As they complete subtasks defined in the Markdown checklist (`- [ ]`), they **MUST** update the checklist item to checked (`- [x]`) using precise file editing tools (`apply_diff`, `search_and_replace`).
*   **Reporting Completion:** When their delegated work is finished, they report completion (`attempt_completion`) back to the coordinator, referencing the original artifact ID.
*   **Referencing Artifacts (e.g., in Commits):** When committing related code changes (via `dev-git`), they should include the relevant IntelliManage artifact ID in the commit message footer (e.g., `Refs: TASK-123`) to enable automated linking via the Integration Layer.

### 3.3. AI Engine (IntelliManage Component)

As specified in `DOC-AI-SPEC-001`, the AI Engine plays a crucial role in bridging natural language interaction with the structured IntelliManage system:

*   **Parses `!pm` Commands:** Understands structured commands and translates them into actions for the CLE.
*   **Understands Natural Language:** Interprets user requests like "Create a task for the login bug" and formulates the necessary `CLE.createArtifact` request.
*   **Generates Drafts:** Creates initial TOML/Markdown content for new artifacts based on user input.
*   **Suggests Links & Statuses:** Analyzes context (chat, Git events) to proactively suggest linking artifacts or updating their status (requiring user confirmation).
*   **Generates Reports:** Processes data retrieved by the CLE to create summaries, boards, and charts.

## 4. GitHub Integration Layer 🔗

The Integration Layer (`DOC-GITHUB-SPEC-001`) further connects IntelliManage artifacts with the external development workflow:

*   **Syncing:** Creates/updates GitHub Issues based on IntelliManage Tasks/Bugs/Stories (and vice-versa if configured). Syncs labels and milestones.
*   **Linking Commits/PRs:** Scans Git history for keywords referencing IntelliManage IDs and updates the `related_commits`/`related_prs` fields in the corresponding artifact files.
*   **Triggering Status Updates:** Notifies the AI Engine when specific commit keywords (e.g., "Fixes TASK-123") are detected, allowing the AI to suggest status changes.

## 5. Conclusion ✅

Roo Commander modes do not operate in isolation but are deeply integrated with the IntelliManage framework. Coordinating modes leverage IntelliManage for planning, task definition, delegation, and tracking. Specialist modes use IntelliManage artifacts as their primary source of requirements and context. The AI Engine acts as an intelligent interface, translating user intent into structured actions and providing proactive assistance. This tight integration ensures that project management remains synchronized with development activities, providing a cohesive and efficient experience within the Roo Code environment.