+++
# --- Basic Metadata ---
id = "KB-RC-INTRO-ARCHITECTURE"
title = "Roo Commander: Architecture Overview"
status = "draft"
doc_version = "1.0" # Version of the architecture being described
content_version = 1.0
audience = ["users", "developers", "architects", "contributors", "ai_modes"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md"
tags = ["roo-commander", "introduction", "architecture", "overview", "multi-agent", "components", "intellimanage"]
related_docs = [
    "../README.md", # Link to the KB README
    "01_What_is_Roo_Commander.md",
    "02_Core_Principles.md",
    "../../../../DOC-ARCH-001.md" # Link to IntelliManage Architecture Spec
    ]
+++

# Roo Commander: Architecture Overview

## 1. Introduction / Overview 🎯

This document provides a high-level overview of the Roo Commander framework's architecture, building upon the concepts introduced in "What is Roo Commander?" (`01_What_is_Roo_Commander.md`) and the Core Principles (`02_Core_Principles.md`). It describes the main components and their interactions, illustrating how the multi-agent system functions within the development environment (e.g., Roo Code in VS Code) and integrates with the IntelliManage project management system.

For a more detailed technical specification of the underlying IntelliManage components, refer to the Overall Architecture document (`DOC-ARCH-001`).

## 2. Architectural Goals 🏗️

The architecture is designed to support the core principles, aiming for:

*   **Modularity:** Components (especially Modes) can be added, removed, or updated independently.
*   **Specialization:** Each component/mode has a well-defined responsibility.
*   **Scalability:** The file-based system and delegated processing allow handling complex projects and multiple agents.
*   **Traceability:** Interactions and state changes are recorded within the structured file system.
*   **Integration:** Seamless operation within the IDE and with external tools (Git, GitHub, LLMs).

## 3. Key Components & Interactions 🧩

The framework comprises several interacting components, primarily operating within the Roo Code extension environment:

```mermaid
graph TD
    subgraph Roo Code Extension / Dev Environment
        UI[Interaction Layer (Chat, Commands, UI Views)]
        RC[Roo Code Core Engine]
        MODES[Modes (Commander, SessionMgr, Dispatch, Leads, Specialists, Agents)]
        RULES[Rules Engine (.roo/)]
        KB[Knowledge Bases (.ruru/modes/*/kb/)]
        CLE[IntelliManage: Core Logic Engine]
        AIE[IntelliManage: AI Engine]
        FS[IntelliManage: File System Store (.ruru/projects/)]
        IL[IntelliManage: Integration Layer]
    end

    User --> UI;
    UI --> RC;
    RC -->|Loads/Runs| MODES;
    MODES -->|Executes Logic/Tools| RC;
    RC -->|Provides Tools/LLM Access| MODES;
    MODES -->|Consults| RULES;
    MODES -->|Consults (Lookup)| KB;

    MODES -->|PM Commands/AI Tasks via RC| CLE;
    CLE -->|Data Ops| FS;
    CLE -->|AI Tasks| AIE;
    AIE -->|LLM Calls| ExternalLLM[External LLM Service];
    AIE -->|Structured Requests| CLE;
    CLE -->|Integration Tasks| IL;
    IL -->|External API Calls| ExternalSys[GitHub API / Git CLI];
    ExternalSys --> IL;
    IL -->|Events/Data| CLE;

    style RC fill:#ddd,stroke:#333,stroke-width:2px
    style MODES fill:#cff,stroke:#333,stroke-width:2px
    style CLE fill:#cfc,stroke:#333,stroke-width:2px
    style AIE fill:#ccf,stroke:#333,stroke-width:2px
```

*   **Interaction Layer (UI):** The user's interface within the IDE (e.g., Roo Code chat panel, `!pm` commands). It sends user input to the Roo Code Core Engine.
*   **Roo Code Core Engine:** The heart of the extension. It manages the active mode, loads the mode's definition/rules/KB, provides tool access (like `read_file`, `new_task`, `execute_command`), handles communication with the LLM (via MCP or directly), and executes the logic defined by the active mode.
*   **Modes:** These are the AI agents themselves, defined by `.mode.md` files. They contain the core prompts and configurations dictating agent behavior. Key coordinating modes for IntelliManage include:
    *   `session-manager`: Manages user session state and high-level goals.
    *   `roo-dispatch`: Handles efficient execution coordination for routine tasks.
    *   `roo-commander`: Orchestrates complex planning, onboarding, and recovery.
    *   *Leads, Specialists, Agents*: Perform domain-specific tasks as delegated.
*   **Rules Engine (`.roo/`):** Loads and applies operational rules (`.roo/rules/` and `.roo/rules-*/`) defined in TOML+MD format. These rules constrain and guide the behavior of the active Mode loaded by the Roo Code Core Engine.
*   **Knowledge Bases (KBs) (`.ruru/modes/*/kb/`):** Contain detailed, mode-specific reference information, procedures, and examples. Modes consult their KB based on triggers defined in their rules.
*   **IntelliManage Core Logic Engine (CLE):** (Conceptual component managing IntelliManage) Handles the business logic for IntelliManage artifacts – CRUD operations, status transitions, linking, validation based on schemas and methodology configuration. Interacts directly with the File System Store.
*   **IntelliManage AI Engine (AIE):** (Conceptual component managing IntelliManage AI) Handles AI-specific IntelliManage functions – parsing natural language `!pm` requests, generating draft artifacts, suggesting links/status updates, generating reports, interacting with external LLMs for these tasks. Collaborates closely with the CLE.
*   **IntelliManage File System Store (`.ruru/projects/`):** The source of truth for all project management artifacts (Initiatives, Epics, Features, Tasks, etc.) and configurations (`project_config.toml`), stored as TOML+MD files.
*   **IntelliManage Integration Layer (IL):** Manages interactions with external systems like Git (monitoring commits) and GitHub (syncing issues/labels/milestones). Communicates changes back to the CLE.
*   **External Systems:** Git repositories, GitHub API, external LLM services.

## 4. Key Interaction Flows

*   **User Command (`!pm ...`):** User -> UI -> Roo Code (running `session-manager` or other mode) -> AI Engine (parses command) -> Core Logic Engine (executes PM action) -> File System Store -> CLE -> AI Engine -> Interaction Layer -> User (result/confirmation).
*   **Development Task Delegation:** User -> UI -> `session-manager` -> `roo-dispatch` (`new_task`) -> Roo Code (loads `roo-dispatch`) -> `roo-dispatch` (selects specialist) -> Specialist Mode (`new_task`) -> Roo Code (loads Specialist) -> Specialist (performs work, uses tools via RC) -> `roo-dispatch` (`attempt_completion`) -> `session-manager` (`attempt_completion`) -> UI -> User.
*   **AI Suggestion:** External Event (e.g., Git commit) -> Integration Layer -> CLE -> AI Engine (infers status change) -> Interaction Layer (prompts user) -> User (confirms) -> UI -> CLE -> File System Store.

## 5. Conclusion ✅

The Roo Commander architecture, integrated with the IntelliManage framework, provides a modular, extensible, and structured approach to AI-assisted development and project management. By separating concerns into distinct components and modes, leveraging a file-based source of truth, and defining clear interaction patterns, the system aims to deliver a powerful, traceable, and efficient experience directly within the developer's workspace.