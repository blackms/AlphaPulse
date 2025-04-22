+++
# --- Basic Metadata ---
id = "WF-RURU-REFACTOR-V1"               # << WORKFLOW-SCOPE-NNN >>
title = "Workflow: RURU Path Refactoring"            # << Human-readable title of the Workflow/SOP >>
status = "draft"      # << draft, active, deprecated, under-review >>
created_date = "2025-04-21"     # << YYYY-MM-DD >>
updated_date = "2025-04-21"     # << YYYY-MM-DD >>
version = "1.0"       # << Workflow document version >>
tags = ["workflow", "refactoring", "ruru", "paths", "sop"] # << Keywords >>

# --- Ownership & Context ---
owner = "Roo Commander" # Or relevant Lead Agent
related_docs = [
    ".planning/ruru-refactoring/01-overview-rationale.md",
    ".planning/ruru-refactoring/02-proposed-architecture.md",
    ".planning/ruru-refactoring/03-implementation-guide-impact.md",
    ".planning/ruru-refactoring/04-considerations-risks.md",
    "ruru.config.toml" # The configuration file itself
]     # << Paths/URLs to essential specs, guides, PAL doc >>
related_templates = [".templates/workflows/00_workflow_boilerplate.md"] # << Paths to data templates used/produced >>

# --- Workflow Specific Fields ---
objective = "To systematically refactor the Roo Commander codebase to use configurable paths defined in `ruru.config.toml` and apply placeholder substitution." # << REQUIRED: Goal of this workflow >>
scope = "Applies to all core build scripts, templates, rules files, KB files, mode definitions, and core documentation within the Roo Commander workspace that contain hardcoded paths intended to be configurable via `ruru.config.toml`."     # << REQUIRED: Applicability and boundaries >>
roles = ["Roo Commander", "prime-dev", "prime-txt"]     # << REQUIRED: List agent roles involved >>
trigger = "Manual initiation by Roo Commander based on the RURU refactoring plan."   # << How is this workflow typically initiated? >>
success_criteria = [
    "All specified files listed in the implementation guide are updated with `{{RURU_...}}` placeholders.",
    "Build scripts (`build_roomodes.js`, `build_mode_summary.js`, `create_build.js`) correctly read and utilize paths from `ruru.config.toml`.",
    "The `substitute-paths.js` script successfully parses `ruru.config.toml`, performs substitutions, and writes processed files.",
    "Core Roo Commander workflows (delegation, context lookup, etc.) function correctly using the processed files.",
    "Automated and manual tests pass for both root-level and nested (`workspace_base`) configurations specified in `ruru.config.toml`."
] # << Measurable conditions for successful completion >>
failure_criteria = [
    "Build scripts fail due to incorrect path handling.",
    "The `substitute-paths.js` script fails or produces incorrect output.",
    "Placeholders remain unsubstituted or are incorrectly substituted in critical files.",
    "Core Roo Commander functionality is broken after applying the refactoring.",
    "Tests fail, indicating regressions or incorrect implementation."
] # << Conditions indicating workflow failure >>

# --- Integration ---
acqa_applicable = true # Does the ACQA process apply to steps in this workflow? (Yes, for script and build changes)
pal_validated = false # Has this workflow been validated using PAL?
validation_notes = "" # Link to PAL validation records/notes

# --- AI Interaction Hints (Optional) ---
# context_type = "workflow_definition"
+++

# Workflow: RURU Path Refactoring

## 1. Objective 🎯
*   To systematically refactor the Roo Commander codebase to use configurable paths defined in `ruru.config.toml` and apply placeholder substitution, enabling flexible workspace structures.

## 2. Scope ↔️
*   This workflow applies to all core build scripts, templates, rules files, KB files, mode definitions, and core documentation within the Roo Commander workspace that contain hardcoded paths intended to be configurable via `ruru.config.toml`.

## 3. Roles & Responsibilities 👤
*   **Roo Commander (Coordinator):** Initiates the workflow, delegates tasks, monitors progress, coordinates testing, and handles overall error management.
*   **prime-dev:** Responsible for creating/modifying configuration files (`.toml`), Node.js scripts (`.js`), and build scripts (`.js`).
*   **prime-txt:** Responsible for modifying Markdown files (`.md`) including templates, rules, KB files, mode definitions, and documentation to replace hardcoded paths with placeholders.

## 4. Preconditions🚦
*   The RURU refactoring plan documents (Overview, Architecture, Implementation Guide, Considerations) exist in `.planning/ruru-refactoring/`.
*   The proposed structure for `ruru.config.toml` is defined (in the Architecture document).
*   Access to the Roo Commander workspace and necessary tools (`read_file`, `write_to_file`, `apply_diff`, `new_task`, `execute_command`).

## 5. Reference Documents & Tools 📚🛠️
*   `.planning/ruru-refactoring/01-overview-rationale.md`
*   `.planning/ruru-refactoring/02-proposed-architecture.md`
*   `.planning/ruru-refactoring/03-implementation-guide-impact.md`
*   `.planning/ruru-refactoring/04-considerations-risks.md`
*   `.roo/rules/` (General rules)
*   `.roo/rules-prime-dev/`
*   `.roo/rules-prime-txt/`
*   Tools: `read_file`, `write_to_file`, `apply_diff`, `new_task`, `execute_command`

## 6. Workflow Steps 🪜

*   **Step 1: Create `ruru.config.toml` (Coordinator delegates to `prime-dev`)**
    *   **Description:** Create the central configuration file at the workspace root.
    *   **Tool:** `new_task`
    *   **Inputs Provided by Coordinator:** Path to Architecture document (`.planning/ruru-refactoring/02-proposed-architecture.md`) containing the TOML structure, target path (`ruru.config.toml`).
    *   **Instructions for Delegate (`prime-dev`):**
        *   Read the structure definition from the Architecture document.
        *   Create the `ruru.config.toml` file at the workspace root.
        *   Populate it with the defined keys and default path values (e.g., `workspace_base = "."`, `modes_root = ".modes"`, `rules_root = ".roo/rules"`, etc.).
        *   Use `write_to_file` to save the file.
        *   Report completion.
    *   **Expected Output from Delegate:** Confirmation of file creation.
    *   **Coordinator Action (Post-Delegation):** Verify file existence (optional, can rely on delegate confirmation). Proceed to Step 2.
    *   **Error Handling:** If `prime-dev` reports failure, analyze the error and potentially retry or adjust instructions.

*   **Step 2: Create/Adapt Substitution Script (Coordinator delegates to `prime-dev`)**
    *   **Description:** Develop the `substitute-paths.js` Node.js script.
    *   **Tool:** `new_task`
    *   **Inputs Provided by Coordinator:** Path to Implementation Guide (`.planning/ruru-refactoring/03-implementation-guide-impact.md` - Step 2 details), target path for the script (e.g., `scripts/substitute-paths.js`).
    *   **Instructions for Delegate (`prime-dev`):**
        *   Implement a Node.js script at the specified path.
        *   The script must:
            *   Read `ruru.config.toml` from the workspace root.
            *   Identify target Markdown files (potentially based on config or arguments).
            *   Read each target file.
            *   Replace all occurrences of `{{RURU_...}}` placeholders with the corresponding values from `ruru.config.toml`. Handle potential nested structures if needed (e.g., `{{RURU_MODES_ROOT}}/{{MODE_SLUG}}/kb`). Consider how `{{MODE_SLUG}}` context is passed or handled.
            *   Write the processed content to a designated output directory (e.g., `.ruru-build/processed` or as defined in config).
        *   Use `write_to_file` to save the script.
        *   Report completion.
    *   **Expected Output from Delegate:** Confirmation of script creation.
    *   **Coordinator Action (Post-Delegation):** Acknowledge confirmation. Proceed to Step 3.
    *   **Validation/QA:** ACQA process should be applied to review the script's logic.
    *   **Error Handling:** Handle script creation failures reported by `prime-dev`.

*   **Step 3: Modify Build Scripts (Coordinator delegates to `prime-dev`)**
    *   **Description:** Update existing build scripts to use `ruru.config.toml`.
    *   **Tool:** `new_task`
    *   **Inputs Provided by Coordinator:** Paths to build scripts (`build_roomodes.js`, `build_mode_summary.js`, `create_build.js`), path to Implementation Guide (`.planning/ruru-refactoring/03-implementation-guide-impact.md` - Step 3 details), path to `ruru.config.toml`.
    *   **Instructions for Delegate (`prime-dev`):**
        *   For each specified build script:
            *   Read the script content (`read_file`).
            *   Modify the script to read relevant paths (`modes_root`, `rules_root`, `modes_file`, `mode_summary_file`, `builds_dir`, etc.) from `ruru.config.toml` instead of using hardcoded values.
            *   Adjust file reading/writing logic to use these configured paths.
            *   Ensure dynamic path construction (like `kb_path` in `build_roomodes.js`) uses the values from the config.
            *   Use `apply_diff` or `write_to_file` (confirming each change) to save modifications.
        *   Report completion for all scripts.
    *   **Expected Output from Delegate:** Confirmation of modifications for all specified build scripts.
    *   **Coordinator Action (Post-Delegation):** Acknowledge confirmation. Proceed to Step 4.
    *   **Validation/QA:** ACQA process should be applied to review the build script changes.
    *   **Error Handling:** Handle modification failures reported by `prime-dev`.

*   **Step 4: Update Templates (Coordinator delegates to `prime-txt`)**
    *   **Description:** Replace hardcoded paths in template files (`.templates/**/*.md`) with placeholders.
    *   **Tool:** `new_task`
    *   **Inputs Provided by Coordinator:** Path to templates directory (`.templates/`), list of placeholders and corresponding config keys (from Architecture doc), path to Implementation Guide (`.planning/ruru-refactoring/03-implementation-guide-impact.md` - Step 4 details).
    *   **Instructions for Delegate (`prime-txt`):**
        *   Identify all `.md` files within the `.templates/` directory (and subdirectories).
        *   For each file:
            *   Read the content (`read_file`).
            *   Replace hardcoded paths (e.g., `.tasks/`, `.docs/`, `.templates/toml-md/README.md`) in both TOML frontmatter and Markdown content with the corresponding `{{RURU_...}}` placeholders (e.g., `{{RURU_TASKS_DIR}}`, `{{RURU_DOCS_DIR}}`, `{{RURU_TEMPLATES_DIR}}/toml-md/README.md`). Pay close attention to `template_schema_doc` fields.
            *   Use `apply_diff` or `write_to_file` (confirming each change) to save modifications.
        *   Report completion for all template files modified.
    *   **Expected Output from Delegate:** Confirmation of modifications for template files.
    *   **Coordinator Action (Post-Delegation):** Acknowledge confirmation. Proceed to Step 5.
    *   **Error Handling:** Handle modification failures reported by `prime-txt`.

*   **Step 5: Update Rules Files (Coordinator delegates to `prime-txt`)**
    *   **Description:** Replace hardcoded paths in rules files (`.roo/rules/**/*.md`, `.roo/rules-<slug>/**/*.md`) with placeholders.
    *   **Tool:** `new_task`
    *   **Inputs Provided by Coordinator:** Paths to rules directories (`.roo/rules/`, `.roo/rules-*/`), list of placeholders, path to Implementation Guide (`.planning/ruru-refactoring/03-implementation-guide-impact.md` - Step 5 details).
    *   **Instructions for Delegate (`prime-txt`):**
        *   Identify all `.md` files within the specified rules directories.
        *   For each file:
            *   Read the content (`read_file`).
            *   Replace *all* hardcoded path references (e.g., `.tasks/`, `.decisions/`, `.modes/roo-commander/kb/`, `.templates/`) with corresponding placeholders (`{{RURU_TASKS_DIR}}`, `{{RURU_DECISIONS_DIR}}`, `{{RURU_MODES_ROOT}}/roo-commander/kb/`, `{{RURU_TEMPLATES_DIR}}`).
            *   Update specific fields like `kb_directory` to use placeholders, e.g., `kb_directory = "{{RURU_MODES_ROOT}}/{{MODE_SLUG}}/kb"`.
            *   Use `apply_diff` or `write_to_file` (confirming each change) to save modifications.
        *   Report completion for all rules files modified.
    *   **Expected Output from Delegate:** Confirmation of modifications for rules files.
    *   **Coordinator Action (Post-Delegation):** Acknowledge confirmation. Proceed to Step 6.
    *   **Error Handling:** Handle modification failures reported by `prime-txt`.

*   **Step 6: Update Knowledge Base Files (Coordinator delegates to `prime-txt`)**
    *   **Description:** Replace hardcoded paths in KB files (`.modes/**/kb/*.md`) with placeholders. This is expected to be extensive.
    *   **Tool:** `new_task`
    *   **Inputs Provided by Coordinator:** Path to modes root directory (`.modes/` or `{{RURU_MODES_ROOT}}` once available), list of placeholders, path to Implementation Guide (`.planning/ruru-refactoring/03-implementation-guide-impact.md` - Step 6 details).
    *   **Instructions for Delegate (`prime-txt`):**
        *   Identify all `.md` files within the `kb/` subdirectories of each mode directory under `.modes/`.
        *   For each file:
            *   Read the content (`read_file`).
            *   Systematically replace *all* hardcoded path references to standard directories, other KB files, rules, templates, etc., with the appropriate `{{RURU_...}}` placeholders. Example: `"Log the goal to {{RURU_TASKS_DIR}}/[TaskID].md"`.
            *   Use `apply_diff` or `write_to_file` (confirming each change) to save modifications.
        *   Report completion for all KB files modified.
    *   **Expected Output from Delegate:** Confirmation of modifications for KB files.
    *   **Coordinator Action (Post-Delegation):** Acknowledge confirmation. Proceed to Step 7.
    *   **Error Handling:** Handle modification failures reported by `prime-txt`. This step might require batching or careful review due to volume.

*   **Step 7: Update Mode Definitions (Coordinator delegates to `prime-txt`)**
    *   **Description:** Replace hardcoded paths in mode definition files (`.modes/**/*.mode.md`) with placeholders.
    *   **Tool:** `new_task`
    *   **Inputs Provided by Coordinator:** Path to modes root directory, list of placeholders, path to Implementation Guide (`.planning/ruru-refactoring/03-implementation-guide-impact.md` - Step 7 details).
    *   **Instructions for Delegate (`prime-txt`):**
        *   Identify all `.mode.md` files within the modes directories.
        *   For each file:
            *   Read the content (`read_file`).
            *   In the TOML frontmatter: Review `context_files`, `documentation_urls`, `related_docs`, `file_access`. Replace internal workspace paths with placeholders where appropriate. Note potential complexity with `file_access` globs. *Initially, focus on simpler path replacements; complex globs might need separate handling or `prime-dev` input.* (Coordinator note: `kb_path` and `custom_instructions_path` might be removed later if handled by build script).
            *   In the Markdown body (system prompt): Replace hardcoded paths (especially KB references) with placeholders. E.g., `KB located in {{RURU_MODES_ROOT}}/{{MODE_SLUG}}/kb/`.
            *   Use `apply_diff` or `write_to_file` (confirming each change) to save modifications.
        *   Report completion for all mode definition files modified.
    *   **Expected Output from Delegate:** Confirmation of modifications for mode definition files.
    *   **Coordinator Action (Post-Delegation):** Acknowledge confirmation. Proceed to Step 8.
    *   **Error Handling:** Handle modification failures reported by `prime-txt`.

*   **Step 8: Update Core Documentation (Coordinator delegates to `prime-txt`)**
    *   **Description:** Update core project documentation to reflect the configurable paths.
    *   **Tool:** `new_task`
    *   **Inputs Provided by Coordinator:** Paths to documentation files (`README.md`, `.docs/standards/project_structure_inventory.md`, `.roo/rules/02-workspace-default-folders.md`), path to Implementation Guide (`.planning/ruru-refactoring/03-implementation-guide-impact.md` - Step 8 details).
    *   **Instructions for Delegate (`prime-txt`):**
        *   For each specified documentation file:
            *   Read the content (`read_file`).
            *   Modify the content to:
                *   Explain the use of `ruru.config.toml`.
                *   Reference `ruru.config.toml` as the source of truth for paths.
                *   Update instructions (e.g., running the substitution script in `README.md`).
                *   Replace descriptions of fixed paths with explanations of the configurable structure.
            *   Use `apply_diff` or `write_to_file` (confirming each change) to save modifications.
        *   Report completion for all documentation files modified.
    *   **Expected Output from Delegate:** Confirmation of modifications for documentation files.
    *   **Coordinator Action (Post-Delegation):** Acknowledge confirmation. Proceed to Step 9.
    *   **Error Handling:** Handle modification failures reported by `prime-txt`.

*   **Step 9: Testing (Coordinator Task, potentially delegating specific tests)**
    *   **Description:** Thoroughly test the refactored system.
    *   **Inputs:** Modified codebase, `ruru.config.toml` (with default and potentially nested `workspace_base` settings), `substitute-paths.js` script.
    *   **Procedure:**
        *   Run the substitution script (`execute_command node scripts/substitute-paths.js`). Verify output in the designated build directory.
        *   Run the build scripts (`execute_command node build_roomodes.js`, etc.). Verify outputs (`.rurumodes`, summary file, build archive).
        *   Manually test core Roo Commander workflows using the processed files/build:
            *   Mode loading and initialization.
            *   Context lookup (KB, rules).
            *   Task delegation involving file access.
            *   Template usage.
        *   Repeat tests with a modified `ruru.config.toml` using a nested `workspace_base` (e.g., `workspace_base = ".ruru"`), ensuring the substitution script and build scripts handle it correctly.
        *   Execute any automated tests if available.
    *   **Outputs:** Test results log, list of identified issues.
    *   **Error Handling:** Log failures, create bug tasks, potentially revert changes if major issues are found. Iterate on previous steps to fix bugs.

## 7. Postconditions ✅
*   All relevant files contain `{{RURU_...}}` placeholders instead of hardcoded paths.
*   `ruru.config.toml` exists and is used by build scripts.
*   `substitute-paths.js` exists and functions correctly.
*   Build scripts produce correct outputs based on `ruru.config.toml`.
*   Roo Commander operates correctly using the processed files generated via the substitution script.
*   Tests pass for both default and nested configurations.

## 8. Error Handling & Escalation (Overall) ⚠️
*   Individual step failures should be reported by the delegate agent to the Coordinator.
*   The Coordinator will attempt to resolve issues by clarifying instructions or re-delegating.
*   Persistent failures in script execution, build processes, or core functionality after refactoring should be escalated.
*   Consider using Git branches and reverting changes if the refactoring introduces critical, hard-to-fix issues.
*   Reference the Adaptive Failure Resolution process (`.processes/afr-process.md`) if standard error handling fails.

## 9. PAL Validation Record 🧪
*   Date Validated: TBD
*   Method: TBD (Simulation, Live Testing)
*   Test Case(s): TBD (Covering core workflows, different config settings)
*   Findings/Refinements: TBD

## 10. Revision History 📜
*   v1.0 (2025-04-21): Initial draft based on RURU refactoring implementation guide.