+++
# --- Basic Metadata ---
id = "KB-RC-ADV-WORKFLOWS-PROCESSES"
title = "Advanced Usage: Understanding Workflows & Processes"
status = "draft"
doc_version = "1.0" # Version of this guide
content_version = 1.0
audience = ["developers", "architects", "contributors", "ai_modes", "project_managers"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md" # Using general doc template structure
tags = ["roo-commander", "intellimanage", "advanced", "guide", "workflow", "process", "sop", "standardization", "automation", "acqa", "afr", "pal"]
related_docs = [
    "../README.md", # Link to the KB README
    "../../../../.ruru/workflows/README.md", # Link to workflows dir README
    "../../../../.ruru/processes/README.md", # Link to processes dir README
    "../../../../.ruru/processes/acqa-process.md", # Link to ACQA
    "../../../../.ruru/processes/afr-process.md", # Link to AFR
    "../../../../.ruru/processes/pal-process.md", # Link to PAL
    "../../../../.ruru/templates/workflows/00_workflow_boilerplate.md", # Link to workflow template
    "../../../../.ruru/templates/processes/01_sop_toml_md.md" # Link to SOP template
    ]
difficulty = "intermediate"
estimated_time = "~15 minutes"
prerequisites = ["Roo Commander installed", "Understanding of Roo Commander architecture and modes"]
learning_objectives = ["Understand the distinction between Workflows and Processes", "Know the purpose and location of Workflow and Process documents", "Recognize key standard processes like ACQA, AFR, and PAL", "Understand how these documents contribute to standardization and automation"]
+++

# Advanced Usage: Understanding Workflows & Processes

## 1. Introduction / Goal 🎯

Beyond the immediate rules that guide AI mode behavior, Roo Commander utilizes **Workflows** and **Processes (SOPs)** to define standardized procedures for more complex, multi-step activities. These documented procedures ensure consistency, reliability, and provide a clear "how-to" for both human users and AI agents involved in coordination or specific quality gates.

This guide explains the purpose of the `.ruru/workflows/` and `.ruru/processes/` directories and highlights key standard processes like ACQA, AFR, and PAL.

**Goal:** To help users understand how standardized procedures are documented and used within the Roo Commander framework to manage complex operations.

## 2. Workflows vs. Processes: What's the Difference? ↔️

While related, these terms represent different levels of procedural detail:

*   **Workflows (`.ruru/workflows/`)**
    *   **Focus:** High-level, end-to-end sequences involving **multiple phases and often multiple agent roles**.
    *   **Purpose:** Define the overall orchestration for achieving a significant project goal (e.g., creating a new mode, onboarding a project, releasing a build).
    *   **Characteristics:** Describe the flow of control, key decision points, delegation patterns, and major inputs/outputs between phases and roles. May reference specific Processes (SOPs) for detailed steps within a phase.
    *   **Example:** `WF-NEW-MODE-CREATION-004.md` (describes steps from user request to final mode registration).
    *   **Template:** `.ruru/templates/workflows/00_workflow_boilerplate.md`

*   **Processes (SOPs) (`.ruru/processes/`)**
    *   **Focus:** Granular, step-by-step instructions for a **specific, repeatable task or activity**, often within a single role's domain or acting as a quality gate.
    *   **Purpose:** Define the exact "how-to" for a standard procedure to ensure consistency and correctness (e.g., validating a document, handling a specific type of error, applying QA checks).
    *   **Characteristics:** Detail specific actions, tools, inputs, outputs, and error handling for a well-defined task. Often referenced *by* Workflows.
    *   **Example:** `acqa-process.md` (defines the steps for quality assurance).
    *   **Template:** `.ruru/templates/processes/01_sop_toml_md.md` (preferred) or `00_sop_basic.md`.

**Analogy:** Think of a Workflow as the overall recipe for baking a cake, while a Process (SOP) is the detailed instruction for "how to cream butter and sugar" or "how to test if the cake is done".

## 3. How Are They Used? ⚙️

*   **Guidance for Coordinators:** Modes like `roo-commander`, `session-manager`, or Leads consult these documents (when directed by their rules or KB) to understand how to orchestrate complex tasks.
*   **Standardization:** They ensure common procedures are performed consistently across the team and by different AI agents.
*   **Training & Onboarding:** Serve as documentation for human users learning standard procedures.
*   **Automation Basis:** Provide the logical steps needed for potentially automating parts of these workflows in the future.
*   **Validation:** The Process Assurance Lifecycle (PAL) (`.ruru/processes/pal-process.md`) is used to validate the logic and completeness of new Workflows and Processes before they are activated.

## 4. Key Standard Processes 🔑

The `.ruru/processes/` directory contains critical standard procedures:

*   **`acqa-process.md` (Adaptive Confidence-based Quality Assurance):**
    *   **Purpose:** Defines how AI-generated artifacts (code, docs, config) are reviewed based on the generating agent's confidence score and the user's caution level. It standardizes the QA feedback loop.
    *   **Used By:** Coordinators (`roo-commander`, `roo-dispatch`) after receiving work from specialists.
*   **`afr-process.md` (Adaptive Failure Resolution):**
    *   **Purpose:** Defines how to handle *recurring* failures identified during the ACQA process. It focuses on identifying and fixing the root cause (e.g., flawed rule, ambiguous spec) rather than just the symptom.
    *   **Used By:** Coordinators when ACQA flags a repeating issue pattern.
*   **`pal-process.md` (Process Assurance Lifecycle):**
    *   **Purpose:** Defines the meta-process for creating, reviewing, simulating, and validating *new* Workflows and Processes (like ACQA and AFR themselves) before they are officially adopted.
    *   **Used By:** Anyone creating or significantly modifying documents in `.ruru/workflows/` or `.ruru/processes/`.

## 5. Creating New Workflows & Processes ✍️

*   **Identify Need:** Determine if a recurring or complex procedure requires standardization.
*   **Choose Scope:** Decide if it's a high-level Workflow or a granular Process (SOP).
*   **Select Template:** Use the appropriate boilerplate from `.ruru/templates/workflows/` or `.ruru/templates/processes/`.
*   **Draft & Refine:** Write the steps clearly, defining roles, inputs, outputs, tools, and error handling.
*   **Validate with PAL:** Follow the Process Assurance Lifecycle (`pal-process.md`) including conceptual review and simulation to ensure robustness.
*   **Finalize & Index:** Save the validated document to the correct directory (`.ruru/workflows/` or `.ruru/processes/`) and update the relevant index file in Roo Commander's KB (`kb/10-standard-processes-index.md` or `kb/11-standard-workflows-index.md`).

## 6. Conclusion ✅

Workflows and Processes are essential for managing complexity and ensuring consistency within the Roo Commander framework. Workflows define the high-level orchestration, while Processes (SOPs) provide the detailed "how-to" for specific activities. Understanding their purpose and location, particularly key standards like ACQA, AFR, and PAL, helps users leverage the full capabilities of the system and contribute to its improvement.