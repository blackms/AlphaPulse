+++
# --- Workflow Metadata ---
id = "WF-PLANNING-PROPOSAL-V1" # (String, Required) Unique identifier (e.g., "WF-REPOMIX-COMPREHENSIVE-V2")
title = "Workflow: Planning Proposal Creation" # (String, Required)
description = """
To define a structured process for capturing, refining, and documenting planning proposals based on user input.
Applies to the creation of planning proposals initiated by a user, involving input capture, interactive refinement, whitepaper generation, and implementation document creation within the `.ruru/planning/` directory structure.
"""
version = "1.0.0" # (String, Required) Semantic version for the workflow definition.
status = "Draft" # (String, Required) Current status: "Draft", "Active", "Deprecated", "Experimental".
tags = ["workflow", "planning", "proposal", "documentation", "refinement"] # (Array of Strings, Optional) Keywords for search/categorization.

# --- Execution Control ---
entry_point = "00_start.md" # (String, Required) Filename of the first step to execute.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Describe overall inputs needed to start the workflow.
    "User request to initiate a planning proposal, providing initial text, files, or both."
]
outputs = [ # (Array of Strings, Optional) Describe the expected final artifacts or outcomes.
    "Proposal directory created at `.ruru/planning/[ProposalName]/` containing initial input, refinement notes, whitepaper, and implementation documents."
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/23_workflow_readme.md" # (String, Required) Link to this template definition.
related_docs = [".roo/rules/01-standard-toml-md-format.md"] # (Array of Strings, Optional) Links to related rules, KBs, ADRs.
+++

# Workflow: Planning Proposal Creation

## Overview

To define a structured process for capturing, refining, and documenting planning proposals based on user input.
Applies to the creation of planning proposals initiated by a user, involving input capture, interactive refinement, whitepaper generation, and implementation document creation within the `.ruru/planning/` directory structure.

## Usage

This workflow is typically initiated by the `prime-coordinator` when a user requests the creation of a planning proposal. The coordinator manages the process, delegating specific tasks like refinement and document generation to specialist modes.

## Inputs

*   **User Request:** A clear request from the user to initiate the planning proposal process.
*   **Initial Input:** Text description and/or paths to relevant files provided by the user.

## Outputs

*   **Proposal Directory:** A dedicated directory structure located at `.ruru/planning/[ProposalName]/`.
*   **Input Files:** The initial user input saved within the `input/` subdirectory.
*   **Refinement Notes:** A file (`Refinement_Notes.md`) summarizing the interactive refinement process.
*   **Whitepaper:** A formal document (`[ProposalName]_Whitepaper.md`) detailing the refined proposal.
*   **Implementation Documents:** Supporting documents like `Implementation_Plan.md` and `Concerns_Analysis.md`.