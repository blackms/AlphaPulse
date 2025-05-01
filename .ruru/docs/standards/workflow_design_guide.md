+++
id = "STD-WORKFLOW-DESIGN-V1"
title = "Workflow Design Guide"
version = 1.0
status = "active"
effective_date = "2025-04-30"
scope = "Design and naming conventions for workflow definition files within the `.ruru/workflows/` directory."
owner = "Roo Commander Project"
template_schema_doc = ".ruru/templates/toml-md/14_standard_guideline.README.md" # Link to schema documentation
tags = ["workflow", "design", "standard", "naming-convention", "files"]
related_docs = [".ruru/workflows/WF-AUDIT-IMPROVE-V1/"]
+++

# Workflow Design Guide (v1.0)

**Status:** active | **Effective Date:** 2025-04-30 | **Owner:** Roo Commander Project

## Purpose / Goal 🎯

*   To establish consistent standards for designing and naming workflow definition files within the `.ruru/workflows/` directory.
*   To ensure clarity, maintainability, and ease of understanding for both humans and automated tools interacting with workflows.
*   To promote modularity and reusability in workflow design.

## Scope 🗺️

*   This guide applies to all workflow definition files located within the `.ruru/workflows/` directory and its subdirectories.
*   It covers the naming conventions for workflow directories and individual step files.

## Standard / Guideline Details 📜

### Workflow Directory Naming

*   Workflow directories should be named descriptively, prefixed with `WF-`, and ideally include a version number (e.g., `WF-CODE-REVIEW-V2/`).

### Workflow Step File Naming

*   Workflow step files represent individual actions or stages within a workflow.
*   They should be named using a numerical prefix to indicate order, followed by a descriptive name, and end with `.md`.
*   Example: `01_initialize_context.md`, `02_analyze_request.md`, `99_finalize_workflow.md`.

#### Guideline: Sequential Sub-Steps (NN[a-z] Convention)

*   **Description:** When a single logical phase or step within a workflow consists of multiple sequential parts, these sub-step files should be named using the primary step number followed by a lowercase letter (`a`, `b`, `c`, ...).
*   **Format:** `NN[a-z]_description.md`
*   **Rationale:**
    *   Visually groups related sub-tasks under a single primary step number in file listings.
    *   Maintains a clear sequential order within the logical phase (`a` is performed before `b`).
    *   Allows for future expansion (e.g., adding a `02c_validate_output.md`) without needing to renumber subsequent major steps (e.g., `03_...`, `04_...`).
*   **Example:** In the `WF-AUDIT-IMPROVE-V1` workflow (`.ruru/workflows/WF-AUDIT-IMPROVE-V1/`), the "Analysis" phase (Step 02) is broken down into:
    *   `02a_check_structure.md`
    *   `02b_review_logic.md`

### Standard Start and Finish Steps

*   Workflows should generally include standard start (`00_start.md`) and finish (`99_finish.md` or similar high number) steps for initialization and cleanup/reporting tasks.

## Related Links 🔗

*   Workflow Directory: `.ruru/workflows/`
*   Example Workflow: `.ruru/workflows/WF-AUDIT-IMPROVE-V1/`