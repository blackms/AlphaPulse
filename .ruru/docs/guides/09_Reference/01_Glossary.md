+++
# --- Basic Metadata ---
id = "KB-RC-REF-GLOSSARY"
title = "Reference: Glossary of Terms"
status = "draft"
doc_version = "1.0" # Version of this glossary
content_version = 1.0
audience = ["users", "developers", "architects", "contributors", "ai_modes"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md" # Using general doc template structure
tags = ["roo-commander", "intellimanage", "reference", "glossary", "definitions", "terminology", "concepts"]
related_docs = [
    "../README.md" # Link to the KB README
    ]
difficulty = "beginner"
estimated_time = "~10 minutes"
prerequisites = ["None"]
learning_objectives = ["Understand the meaning of key terms used throughout the Roo Commander and IntelliManage documentation"]
+++

# Reference: Glossary of Terms

This glossary defines key terms used within the Roo Commander and IntelliManage documentation to ensure consistent understanding.

---

**A**

*   **ACQA (Adaptive Confidence-based Quality Assurance):** A standard process (`.ruru/processes/acqa-process.md`) used to determine the appropriate level of review for AI-generated artifacts based on the generating agent's confidence score and user settings. (See: `06_Advanced_Usage_Customization/05_Understanding_Workflows_Processes.md`)
*   **AFR (Adaptive Failure Resolution):** A standard process (`.ruru/processes/afr-process.md`) for handling recurring failures identified during ACQA, focusing on root cause analysis and correction. (See: `06_Advanced_Usage_Customization/05_Understanding_Workflows_Processes.md`)
*   **Agent (Mode Classification):** A type of mode that performs specific, often automated or background support tasks (e.g., context retrieval, research, summarization). (See: `04_Understanding_Modes/01_Mode_Roles_Hierarchy.md`)
*   **Artifact (IntelliManage):** A discrete unit of work, planning, or decision within the IntelliManage framework, typically stored as a TOML+MD file (e.g., Initiative, Epic, Feature, Task, Bug, ADR). (See: `02_Core_Concepts/06_IntelliManage_Artifacts.md`)

**B**

*   **Bug (IntelliManage Artifact):** An artifact representing a defect, error, or issue in the software that needs to be fixed. Stored in `.ruru/projects/[slug]/tasks/`. (See: `02_Core_Concepts/06_IntelliManage_Artifacts.md`)

**C**

*   **Classification (Mode):** The role assigned to a mode (e.g., `core`, `director`, `lead`, `worker`, `agent`, `utility`) defined in its `.mode.md` file. Determines its place in the hierarchy and typical function. (See: `04_Understanding_Modes/01_Mode_Roles_Hierarchy.md`)
*   **CLE (Core Logic Engine - IntelliManage):** The conceptual component responsible for the business logic of IntelliManage, including CRUD operations on artifacts, validation, and state management. (See: `01_Introduction/03_Architecture_Overview.md`)
*   **Context Window (LLM):** The finite amount of information (text, code, history, instructions) that a Large Language Model can process at one time. Managing this limit is crucial for performance and reliability. (See: `07_Best_Practices_Troubleshooting/02_Managing_AI_Context.md`)
*   **Coordinator (Mode Classification):** A general term for modes primarily responsible for managing workflows, delegating tasks, and interacting with the user (e.g., `roo-commander`, `session-manager`, `roo-dispatch`, `prime-coordinator`).
*   **Core (Mode Classification):** Foundational system modes essential for the framework's operation or high-level strategy. (See: `04_Understanding_Modes/01_Mode_Roles_Hierarchy.md`)

**D**

*   **Delegation:** The core process in Roo Commander where one mode assigns a specific task to another, more specialized mode using the `new_task` tool.
*   **Dependency (`depends_on`):** A sequential link between two work items (Tasks, Bugs, Stories) indicating that one must be completed before the other can start. Managed via the `depends_on` array in TOML frontmatter. (See: `05_Using_IntelliManage_Features/03_Linking_Dependencies.md`)
*   **Director (Mode Classification):** High-level management modes responsible for specific strategic areas (e.g., project management, product definition). (See: `04_Understanding_Modes/01_Mode_Roles_Hierarchy.md`)

**E**

*   **Epic (IntelliManage Artifact):** A large body of work within a project, grouping related Features. Stored in `.ruru/projects/[slug]/epics/`. (See: `02_Core_Concepts/06_IntelliManage_Artifacts.md`)

**F**

*   **Feature (IntelliManage Artifact):** A distinct piece of user-facing functionality within an Epic, grouping related Tasks/Stories/Bugs. Stored in `.ruru/projects/[slug]/features/`. (See: `02_Core_Concepts/06_IntelliManage_Artifacts.md`)
*   **File System Store:** The canonical source of truth for IntelliManage artifacts and Roo Commander configuration, residing within the `.roo/` and `.ruru/` directories. (See: `01_Introduction/03_Architecture_Overview.md`, `02_Core_Concepts/01_File_System_Structure.md`)

**H**

*   **Handover Summary:** A concise snapshot of the work session state (goals, progress, next steps) generated by `agent-session-summarizer` and used by `session-manager` to maintain context continuity. Stored in `.ruru/context/handovers/`. (See: `03_Getting_Started/04_Session_Management_Continuity.md`)
*   **Hierarchy (IntelliManage):** The structured relationship between work artifacts: Initiative > Epic > Feature > Task/Story/Bug > Subtask. (See: `02_Core_Concepts/06_IntelliManage_Artifacts.md`)
*   **Hierarchy (Modes):** The general classification and delegation flow between different types of modes (Core, Director, Lead, Worker, Agent, Utility). (See: `04_Understanding_Modes/01_Mode_Roles_Hierarchy.md`)

**I**

*   **Initiative (IntelliManage Artifact):** The highest-level artifact representing a strategic goal, potentially spanning multiple projects. Stored in `.ruru/projects/[slug]/initiatives/`. (See: `02_Core_Concepts/06_IntelliManage_Artifacts.md`)
*   **IntelliManage:** The integrated project management framework within Roo Commander, using file-based artifacts in `.ruru/projects/`.
*   **Interaction Layer:** The user interface within the IDE (e.g., Roo Code chat panel, commands) used to interact with Roo Commander modes. (See: `01_Introduction/03_Architecture_Overview.md`)

**K**

*   **KB (Knowledge Base):** A collection of detailed, mode-specific reference information, procedures, and examples stored in `.ruru/modes/[mode_slug]/kb/`. Accessed on demand via KB Lookup Rules. (See: `02_Core_Concepts/05_Knowledge_Bases_Explained.md`)

**L**

*   **Lead (Mode Classification):** Modes responsible for overseeing specific technical or functional domains, decomposing tasks, and delegating to/reviewing work from Specialists. (See: `04_Understanding_Modes/01_Mode_Roles_Hierarchy.md`)
*   **LLM (Large Language Model):** The underlying AI model (e.g., Gemini, Claude, GPT) that powers the Roo Commander modes.

**M**

*   **MCP (Model Control Protocol):** A specification allowing external tools or services (like Task Master AI, or potentially future research tools) to be controlled by AI agents within environments like Roo Code.
*   **MDTM (Markdown-Driven Task Management):** The system of using individual TOML+MD files to represent and track work items (Tasks, Features, Bugs, etc.). (See: `02_Core_Concepts/03_MDTM_Explained.md`)
*   **Mode:** An AI agent within the Roo Commander framework, defined by a `.mode.md` file, rules, and potentially a KB, designed for a specific role or task.
*   **Multi-Project Workspace:** A workspace configured to manage multiple distinct projects using IntelliManage within the `.ruru/projects/` directory. (See: `02_Core_Concepts/07_Multi_Project_Workspaces.md`)

**P**

*   **PAL (Process Assurance Lifecycle):** A standard meta-process (`.ruru/processes/pal-process.md`) for validating new Workflows and Processes before adoption. (See: `06_Advanced_Usage_Customization/05_Understanding_Workflows_Processes.md`)
*   **Preferences (User):** Configurable settings in `.roo/rules/00-user-preferences.md` that allow users to personalize Roo Commander's behavior (e.g., verbosity, skills). (See: `06_Advanced_Usage_Customization/03_User_Preferences.md`)
*   **Prime Coordinator (`prime-coordinator`):** An advanced mode for direct configuration changes (meta-development) and operational task delegation, bypassing some of the standard planning/interaction layers. (See: `06_Advanced_Usage_Customization/04_Prime_Coordinator_Usage.md`)
*   **Process (SOP):** A detailed, step-by-step Standard Operating Procedure for a specific, repeatable task, stored in `.ruru/processes/`. (See: `06_Advanced_Usage_Customization/05_Understanding_Workflows_Processes.md`)
*   **Project Slug:** A unique, filesystem-friendly identifier (lowercase, hyphens/underscores) used for project directory names under `.ruru/projects/` and in commands (e.g., `frontend-app`).

**R**

*   **Roo Code:** The VS Code extension that provides the environment for running Roo Commander modes and interacting with the framework.
*   **Roo Commander (`roo-commander`):** The core coordinating mode and the name of the overall multi-agent framework.
*   **Roo Dispatch (`roo-dispatch`):** A lean coordinating mode focused on efficiently executing specific delegated tasks by managing specialist modes. (See: `WP-INTELLIMANAGE-SESSION-DISPATCH-V1.md`)
*   **Rule:** An instruction, constraint, or procedure defined in a TOML+MD file within `.roo/rules/` (workspace-wide) or `.roo/rules-[mode_slug]/` (mode-specific), guiding AI mode behavior. (See: `02_Core_Concepts/04_Rules_Explained.md`)

**S**

*   **Session Manager (`session-manager`):** The primary mode for user interaction, managing session state, high-level goals, and context continuity using Handover Summaries. (See: `03_Getting_Started/04_Session_Management_Continuity.md`, `WP-INTELLIMANAGE-SESSION-DISPATCH-V1.md`)
*   **Slug:** A short, unique, URL/filesystem-friendly identifier (e.g., `mode_slug`, `project_slug`).
*   **Specialist (Mode Classification):** See Worker.
*   **Stack Profile:** An optional file (`.ruru/context/stack_profile.json`) defining the technologies used in a project, helping coordinators select appropriate specialist modes.
*   **Story (IntelliManage Artifact):** A user-centric requirement artifact, often following the "As a [user], I want [goal], so that [benefit]" format. Stored in `.ruru/projects/[slug]/tasks/`. (See: `02_Core_Concepts/06_IntelliManage_Artifacts.md`)
*   **Subtask:** A granular step defined as a Markdown checklist item (`- [ ]`) within a parent Task/Story/Bug file. (See: `02_Core_Concepts/06_IntelliManage_Artifacts.md`)
*   **System Prompt:** The core instruction within a mode's `.mode.md` file defining its identity, primary role, and high-level operational guidelines.

**T**

*   **Task (IntelliManage Artifact):** A specific, actionable work item required to implement a Feature or address an issue. Stored in `.ruru/projects/[slug]/tasks/`. (See: `02_Core_Concepts/06_IntelliManage_Artifacts.md`)
*   **TOML+MD:** The standard file format used for artifacts, rules, and mode definitions, combining TOML frontmatter for metadata and Markdown for content. (See: `02_Core_Concepts/02_TOML_MD_Standard.md`)
*   **Tool:** A capability provided by the Roo Code environment that AI modes can invoke (e.g., `read_file`, `write_to_file`, `execute_command`, `new_task`, `browser`).

**U**

*   **Utility (Mode Classification):** Modes providing general-purpose helper functions applicable across different domains. (See: `04_Understanding_Modes/01_Mode_Roles_Hierarchy.md`)

**W**

*   **Worker (Mode Classification):** Modes responsible for performing the core implementation work within a specific technical domain (e.g., coding, testing, infrastructure setup). Often referred to as Specialists. (See: `04_Understanding_Modes/01_Mode_Roles_Hierarchy.md`)
*   **Workflow:** A high-level, end-to-end sequence of phases and agent interactions for achieving a significant goal, documented in `.ruru/workflows/`. (See: `06_Advanced_Usage_Customization/05_Understanding_Workflows_Processes.md`)