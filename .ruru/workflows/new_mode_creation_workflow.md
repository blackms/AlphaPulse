+++
# --- Basic Metadata ---
id = "WF-MODE-001"
title = "Interactive New Mode Creation Workflow (.ruru/modes/ Structure)"
status = "draft" # Reverted to draft for revision
created_date = "2025-04-16"
updated_date = "2025-04-24" # Date updated reflecting these revisions
version = "3.0" # Version remains 3.0 as per history
tags = ["workflow", "mode-creation", "interactive", "modes-structure", "naming-convention", "kb-population", "readme-enhancement", "kb-generation", "template-enforcement", "mode-registry"] # Added mode-registry, removed manifest

# --- Ownership & Context ---
owner = "Roo Commander"
related_docs = [
    "`.ruru/modes/roo-commander/kb/available-modes-summary.md`", # Standardized format
    # ".ruru/templates/modes/mode_specification.md", # Superseded by standard template
    "`.ruru/rules/00-standard-toml-md-format.md`", # Standardized format
    "`.ruru/processes/acqa-process.md`", # Standardized format
    "`.ruru/processes/afr-process.md`", # Standardized format
    "`.ruru/processes/pal-process.md`" # Standardized format
]
related_templates = [
    "`.ruru/templates/modes/00_standard_mode.md`", # Standard Mode Template (v1.1) - Standardized format
    "`.ruru/templates/toml-md/16_ai_rule.md`" # For KB lookup rule - Standardized format
    # ".ruru/templates/modes/kb_placeholders/" # Placeholders superseded by generation
]

# --- Workflow Specific Fields ---
objective = "To guide the creation of a new Roo Commander mode from scratch, using the `.ruru/modes/` structure, applying predefined naming conventions, separating knowledge base (KB) content (with option to generate basic KB if missing), creating mode-specific rules, generating an enhanced KB README, running `build_roomodes.js` to update the mode registry, and **enforcing the standard mode template structure**, maximizing delegation." # Updated manifest reference
scope = "Applies when creating a new mode not based on a direct migration. Assumes user interaction is available. Creates structure in `.ruru/modes/` and `.roo/rules-<slug>/`. Uses the naming convention document for guidance, prompts for KB population if needed, generates an enhanced KB README, **populates the standard mode template**, and triggers the mode registry update via `build_roomodes.js`." # Updated manifest reference
roles = ["User", "Coordinator (Roo Commander)", "Context Gatherer (e.g., research-context-builder)", "Context Synthesizer (e.g., context-condenser)", "Mode Structure Agent (e.g., mode-maintainer, technical-writer, toml-specialist)", "QA Agent (e.g., code-reviewer)"]
trigger = "User request to create a new mode, specifying its purpose and basic identification."
success_criteria = [
    "Mode definition file exists in `.ruru/modes/<new-slug>/<new-slug>.mode.md`, **adhering to the standard template structure** (`.ruru/templates/modes/00_standard_mode.md`) and containing gathered/generated data with the correct `id`.",
    "KB directory exists at `.ruru/modes/<new-slug>/kb/`.",
    "KB directory contains either generated content from provided sources or generated basic KB content (unless explicitly skipped).",
    "KB README (`.ruru/modes/<new-slug>/kb/README.md`) exists and contains an overview, file list with summaries and line counts (or indicates pending/skipped population).",
    "Mode-specific rule directory exists at `.roo/rules-<slug>/`.",
    "KB lookup rule file exists at `.roo/rules-<slug>/01-kb-lookup-rule.md` using the standard rule template (`.ruru/templates/toml-md/16_ai_rule.md`) with enhanced instructions.",
    "The mode registry is successfully updated after running `build_roomodes.js`.",
    "The created structure passes Quality Assurance (QA) review against specifications (including template structure).",
    "User confirms the generated mode meets initial requirements and is accessible after window reload."
]
failure_criteria = [
    "Unable to determine a valid `prefix-topic` slug with user.",
    "Unable to gather sufficient context.",
    "Failure during optional KB population/generation sub-process.",
    "Worker Agent fails to correctly populate the standard mode template.",
    "Worker Agent fails critical file/directory operations.",
    "Worker Agent fails to generate enhanced KB README.",
    "Failure during execution of `build_roomodes.js`.",
    "Generated structure repeatedly fails QA.",
    "User rejects the final mode structure or cannot access it after reload."
]

# --- Integration ---
acqa_applicable = true
pal_validated = false # Needs re-validation
validation_notes = ""

# --- AI Interaction Hints (Optional) ---
# context_type = "workflow_definition"
+++

# Interactive New Mode Creation Workflow

## 1. Objective 🎯
To guide the creation of a new Roo Commander mode from scratch. This involves:
*   Using the `.ruru/modes/` directory structure.
*   Applying predefined naming conventions (see `ai-dev-kb-repo/mode_naming_convention_v7.2.md`).
*   Separating knowledge base (KB) content into the `kb/` subdirectory, with an interactive option to generate basic KB if source material is missing or insufficient.
*   Creating mode-specific rules in the corresponding `.roo/rules-<slug>/` directory.
*   Generating an enhanced KB `README.md` file summarizing the KB contents.
*   Running the `build_roomodes.js` script to update the application's mode registry.
*   **Enforcing the use and structure of the standard mode template** (`.ruru/templates/modes/00_standard_mode.md`).
*   Maximizing delegation of tasks to specialized agents where appropriate.

## 2. Scope ↔️
This workflow applies when creating a *new* Roo Commander mode that is not derived from a direct migration of an older format. It requires user interaction for decisions and confirmations. The process results in the creation of the mode's directory structure within `.ruru/modes/`, the associated rules directory within `.roo/rules-<slug>/`, population of the standard mode template, optional KB population, generation of the KB README, creation of the KB lookup rule, and triggering the update of the mode registry via `build_roomodes.js`.

## 3. Roles & Responsibilities 👤
*   **User:** Initiates the request, provides purpose/context, approves slug/classification, reviews KB options, and confirms final mode usability.
*   **Coordinator (Roo Commander):** Orchestrates the workflow, interacts with the User, delegates tasks to Worker Agents, performs QA, and manages finalization steps.
*   **Context Gatherer:** (Worker Agent) Gathers relevant information based on the mode's purpose.
*   **Context Synthesizer:** (Worker Agent) Condenses gathered information into a structured format suitable for mode definition and KB.
*   **Mode Structure Agent:** (Worker Agent) Creates directories, populates template files (mode definition, KB README, rules) based on synthesized context and templates.
*   **QA Agent:** (Worker Agent, potentially Coordinator) Reviews generated artifacts against standards and requirements (part of ACQA process).

## 4. Preconditions🚦
*   The Roo Commander system and its delegated agents are operational.
*   The User is available for interaction and providing necessary input/confirmations.
*   Required templates (`.ruru/templates/modes/00_standard_mode.md`, `.ruru/templates/toml-md/16_ai_rule.md`) exist and are accessible.
*   Reference documents (naming convention, available modes summary) are accessible.
*   The `build_roomodes.js` script exists and is executable by the Coordinator.

## 5. Reference Documents & Tools 📚🛠️
*   Naming Convention: `ai-dev-kb-repo/mode_naming_convention_v7.2.md`
*   Existing Mode Examples: `.ruru/modes/roo-commander/kb/available-modes-summary.md`
*   Standard Mode Template: `.ruru/templates/modes/00_standard_mode.md`
*   Standard Rule Template: `.ruru/templates/toml-md/16_ai_rule.md`
*   TOML+MD Format Rule: `.ruru/rules/00-standard-toml-md-format.md`
*   QA Process: `.ruru/processes/acqa-process.md`
*   Failure Resolution Process: `.ruru/processes/afr-process.md`
*   Process Validation Lifecycle: `.ruru/processes/pal-process.md`
*   Mode Registry Build Script: `build_roomodes.js` (Assumed location accessible to Coordinator)

## 6. Workflow Steps 🪜

*   **Step 1: Initiation & Requirements Gathering (Coordinator, User)**
    *   Coordinator confirms the new mode's primary purpose, intended use cases, and target audience with the User.
    *   Coordinator consults the naming convention (`ai-dev-kb-repo/mode_naming_convention_v7.2.md`) and existing mode examples (`.ruru/modes/roo-commander/kb/available-modes-summary.md`) to propose an appropriate `prefix-topic` slug and `classification` for the new mode.
    *   Coordinator confirms the proposed slug and classification with the User. If rejected, iterate until an acceptable slug/classification is agreed upon.

*   **Step 2: Context Gathering (Coordinator delegates to Context Gatherer)**
    *   Coordinator instructs the Context Gatherer agent to find and retrieve relevant information (documents, code snippets, URLs, etc.) based on the agreed purpose and scope of the new mode.

*   **Step 3: Context Synthesis (Coordinator delegates to Context Synthesizer)**
    *   Coordinator instructs the Context Synthesizer agent to process the gathered information, structuring it logically for inclusion in the mode definition and potential KB files. This includes identifying key concepts, instructions, examples, and constraints.

*   **Step 3b: Optional KB Population Prompt (Coordinator, User)**
    *   Coordinator reviews the synthesized context.
    *   If context seems sufficient for KB population: Ask User: "I have synthesized context for the mode's Knowledge Base (KB). Do you want me to proceed with populating the KB files using this context?" (Options: Yes, No/Skip KB for now).
    *   If context seems insufficient: Ask User: "The gathered context seems limited. I can attempt to generate a *basic* KB structure with placeholders based on the mode's purpose, or we can skip KB population for now. How should I proceed?" (Options: Generate basic KB, Skip KB for now).
    *   Store the User's decision regarding KB population.

*   **Step 4: Delegate Directory Creation (Coordinator delegates to Mode Structure Agent)**
    *   Coordinator instructs the Mode Structure Agent to create the necessary directory structure:
        *   Mode directory: `.ruru/modes/<new-slug>/`
        *   KB subdirectory: `.ruru/modes/<new-slug>/kb/`
        *   Rules directory: `.roo/rules-<slug>/` (using the same `<new-slug>`)

*   **Step 5: Delegate Initial Mode File Creation using Template (Coordinator delegates to Mode Structure Agent)**
    *   Coordinator instructs the Mode Structure Agent to:
        *   Copy the standard mode template (`.ruru/templates/modes/00_standard_mode.md`) to `.ruru/modes/<new-slug>/<new-slug>.mode.md`.
        *   Populate the TOML frontmatter and relevant sections of this new `.mode.md` file using the synthesized context, agreed slug, classification, and a generated `id` (e.g., `MODE-<SLUG>`). Ensure adherence to the template's structure.

*   **Step 6: Delegate KB Content / Instruction File Creation (Coordinator delegates to Mode Structure Agent)**
    *   Based on the User's decision in Step 3b:
        *   If **Yes (Use Synthesized Context)**: Instruct the Agent to create relevant files within `.ruru/modes/<new-slug>/kb/` based on the synthesized context.
        *   If **Generate basic KB**: Instruct the Agent to create placeholder files or a single file in `.ruru/modes/<new-slug>/kb/` indicating basic structure and need for population.
        *   If **Skip KB**: No KB files are created at this stage, but the KB directory exists.

*   **Step 7: Delegate Enhanced KB README Update (Coordinator delegates to Mode Structure Agent)**
    *   Coordinator instructs the Mode Structure Agent to create/update the KB README file at `.ruru/modes/<new-slug>/kb/README.md`.
    *   This README should include:
        *   An overview of the KB's purpose (derived from mode purpose).
        *   A list of files within the `kb/` directory (if any created in Step 6).
        *   Brief summaries and line counts for each KB file (or indicate "Basic structure generated" or "KB population skipped").

*   **Step 8: Delegate KB Rule Creation (Coordinator delegates to Mode Structure Agent)**
    *   Coordinator instructs the Mode Structure Agent to:
        *   Copy the standard AI rule template (`.ruru/templates/toml-md/16_ai_rule.md`) to `.roo/rules-<slug>/01-kb-lookup-rule.md`.
        *   Populate the template, ensuring the rule correctly targets the KB directory (`.ruru/modes/<new-slug>/kb/`) and includes enhanced instructions for the AI on how to utilize the KB content effectively for this specific mode.

*   **Step 9: Quality Assurance (Coordinator applies ACQA)**
    *   Coordinator initiates the Adaptive Confidence-based Quality Assurance (ACQA) process (defined in `.ruru/processes/acqa-process.md`).
    *   This involves checks (potentially delegated to a QA Agent) for:
        *   Correct directory structure and naming.
        *   Presence and basic validity of all required files (`.mode.md`, `kb/README.md`, `01-kb-lookup-rule.md`).
        *   Adherence of the `.mode.md` file to the standard template structure.
        *   Consistency between the mode definition, KB README, and KB lookup rule.
        *   Correct population of metadata (ID, slug, classification).
    *   If issues are found, initiate corrections (potentially looping back to relevant creation steps) and re-run QA. Persistent failures may trigger the Adaptive Failure Resolution (AFR) process (see Section 8).

*   **Step 10: User Review & Refinement (Coordinator, User)**
    *   Coordinator presents the generated mode structure (key files like `.mode.md`, `kb/README.md`) to the User for review.
    *   Coordinator asks for feedback: "Does this initial structure align with your requirements for the new mode?"
    *   If the User requests refinements, the Coordinator gathers the feedback, determines necessary changes, and potentially loops back to earlier steps (e.g., Step 3 for context, Step 5/6/7/8 for file content) to implement them, followed by re-running QA (Step 9).

*   **Step 11: Build Mode Registry (Coordinator)**
    *   Once the structure passes QA and User review, the Coordinator executes the command to update the mode registry.
    *   Coordinator uses `execute_command` (or equivalent mechanism) to run: `node build_roomodes.js`.
    *   Coordinator verifies the command executed successfully (e.g., checks for exit code 0 and absence of critical errors in output). Handle script execution errors as per Section 8.

*   **Step 12: Reload Window (User Action - IMPORTANT)**
    *   Coordinator informs the user: "The mode structure is complete and the registry has been rebuilt. **Please reload the VS Code window now** for the changes to take effect. You can do this via the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`) and searching for 'Developer: Reload Window'."
    *   Coordinator waits for user confirmation that the window has been reloaded before proceeding.

*   **Step 13: Finalization (Coordinator)**
    *   Coordinator confirms with the user that the new mode is now available in the application's mode list and functions as expected at a basic level (e.g., user can switch to it).
    *   Coordinator marks the workflow task as complete.

## 7. Postconditions ✅
*   The new mode's directory structure exists under `.ruru/modes/`.
*   The mode definition file (`<new-slug>.mode.md`) exists, conforms to the standard template, and contains initial content.
*   The KB directory (`kb/`) exists, containing either populated content, basic generated content, or is empty (as per user choice), along with an updated `README.md`.
*   The mode-specific rules directory exists under `.roo/rules-<slug>/`.
*   The KB lookup rule (`01-kb-lookup-rule.md`) exists and is configured.
*   The mode registry has been updated via `build_roomodes.js`.
*   The User has confirmed the mode is accessible and meets initial requirements after reloading the window.

## 8. Error Handling & Escalation (Overall) ⚠️
*   **Invalid Slug/Classification:** If agreement cannot be reached in Step 1, escalate to the User/owner for clarification or abandon the workflow.
*   **Context Gathering/Synthesis Failure:** If agents fail (Step 2/3), retry. If persistent, notify the User and potentially proceed with minimal context or abandon.
*   **KB Population Failure:** If KB generation/population fails (Step 6), notify the User, ensure the KB README reflects the failure, and proceed if possible (mode might function without KB initially).
*   **File/Directory Operations Failure:** If the Mode Structure Agent fails critical operations (Step 4, 5, 6, 7, 8), retry. Log errors. Persistent failure requires manual intervention or abandoning the workflow.
*   **QA Failures (Step 9):** Minor issues trigger corrections and re-QA. Repeated or significant failures should trigger the Adaptive Failure Resolution (AFR) process (`.ruru/processes/afr-process.md`) to diagnose root causes.
*   **`build_roomodes.js` Failure (Step 11):** If the script fails, log the error output. Attempt to diagnose (e.g., syntax error in a mode file). If resolvable, fix and retry Step 11. If not, escalate the script error; the mode will not be available until resolved.
*   **User Rejection (Step 10/13):** If the user rejects the final structure or cannot access the mode, gather detailed feedback. Attempt refinement loop (back to Step 10 or earlier). If fundamental issues persist, escalate or document the rejection and close.

## 9. PAL Validation Record 🧪
*(Process Assurance Lifecycle - defined in `.ruru/processes/pal-process.md`)*
*   **Date Validated:** (Needs re-validation for v3.0)
*   **Method:** (e.g., Walkthrough, Test Case Execution)
*   **Test Case(s):** (e.g., Create mode 'test-basic', Create mode 'dev-complex' with KB generation)
*   **Findings/Refinements:** (Document results of validation here)

## 10. Revision History 📜
*   **v3.0 (2025-04-24):** Added Step 11 to run `build_roomodes.js` and Step 12 to remind user to reload VS Code window. Renumbered Finalization to Step 13. Updated TOML `objective` and `scope` to reflect registry build instead of manifest update. Removed duplicate v3.0 history entry. Standardized filename formatting. Enhanced clarity by removing "(No change)" placeholders and defining acronyms. Added detail to refinement (Step 10) and QA failure handling (Step 9, Section 8).
*   **v2.9 (2025-04-24):** Removed manifest steps/references. Added explicit reference to `available-modes-summary.md` for classification guidance in `related_docs` and Step 1. Corrected path for naming convention doc.
*   **v2.8 (2025-04-18):** Updated `related_templates` to reflect standard template v1.1 (containing refined KB guidance). No change to workflow steps themselves.
*   **v2.7 (2025-04-18):** Updated Step 5 to enforce population of the standard template. Updated verification check 3.
*   **v2.6 (2025-04-18):** Refined the "Generate basic KB" instruction in Step 6.
*   **v2.5 (2025-04-18):** Modified Step 3b prompt to offer "Generate basic KB content". Updated Step 6, 7, and 10.
*   **v2.4 (2025-04-18):** Enhanced Step 7 (README generation) and Step 8 (KB lookup rule). Adjusted verification check 5.
*   **v2.3 (2025-04-18):** Added Step 3b to handle insufficient initial context. Adjusted Step 6, 7, and 10.
*   **v2.2 (2025-04-18):** Updated Step 1/3 to use naming convention doc. Status back to draft.
*   **v2.1 (2025-04-18):** Incorporated conceptual review feedback. Added `domain`. Clarified manifest creation. Added `new_task`. Embedded KB rule content. Updated error handling. Added template compatibility note.
*   **v2.0 (2025-04-18):** Major revision for `.ruru/modes/` structure. Updated paths, added manifest/README/rule steps, refined naming.
*   **v1.1 (2025-04-16):** Incorporated conceptual review feedback.
*   **v1.0 (2025-04-16):** Initial draft.