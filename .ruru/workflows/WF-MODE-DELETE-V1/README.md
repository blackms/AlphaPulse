+++
# --- Workflow Metadata ---
id = "WF-MODE-DELETE-V1" # (String, Required) Unique identifier
title = "Workflow: Delete Custom Mode" # (String, Required)
description = """
(String, Required) Defines the steps to safely delete a custom mode, its associated rules directory (if present),
and update relevant configuration files like `.ruru/config/build_collections.json`.
The `.roomodes` file is updated by running the build script.
"""
version = "1.0.0" # (String, Required) Semantic version for the workflow definition.
status = "Draft" # (String, Required) Current status: "Draft", "Active", "Deprecated", "Experimental".
tags = ["workflow", "mode", "delete", "configuration", "prime", "cleanup"] # (Array of Strings, Optional) Keywords for search/categorization.

# --- Execution Control ---
entry_point = "00_start.md" # (String, Required) Filename of the first step to execute.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Describe overall inputs needed to start the workflow.
    "User request specifying the mode slug to delete.",
    "User confirmation before deletion.",
]
outputs = [ # (Array of Strings, Optional) Describe the expected final artifacts or outcomes.
    "Confirmation of mode directory deletion.",
    "Confirmation of rules directory deletion (if applicable).",
    "Confirmation of `.ruru/config/build_collections.json` update (if applicable).",
    "Confirmation of build script execution.",
    "Final result summary.",
]

# --- Housekeeping ---
last_updated = "2025-04-30" # (String, Required) Date of last modification.
template_schema_doc = ".ruru/templates/toml-md/23_workflow_readme.md" # (String, Required) Link to this template definition.
related_docs = [
    ".ruru/modes/",
    ".roo/rules-*/",
    ".roomodes",
    ".ruru/scripts/build_roomodes.js",
    ".ruru/config/build_collections.json",
    ".roo/rules/01-standard-toml-md-format.md",
    ".roo/rules/05-os-aware-commands.md",
    ".roo/rules/08-logging-procedure-simplified.md",
] # (Array of Strings, Optional) Links to related rules, KBs, ADRs.
+++

# Workflow: Delete Custom Mode

## Overview

This workflow provides a safe procedure for removing a custom mode and its associated configuration from the workspace. It ensures that the mode's directory, its specific rules directory (if it exists), and references in configuration files are handled correctly. The process requires user confirmation before any destructive actions are taken.

## Usage

This workflow is typically initiated by a Coordinator (like `prime-coordinator`) when a user requests to delete a custom mode. The Coordinator guides the user through identifying the mode and confirming the deletion before executing the removal steps and updating configurations.

## Inputs

*   **Mode Slug:** The exact slug (identifier) of the custom mode to be deleted (e.g., `my-custom-mode`). Provided by the user.
*   **User Confirmation:** Explicit confirmation ("Yes" or "Proceed") from the user after reviewing the files and directories identified for deletion.

## Outputs

*   **Deletion Confirmations:** Messages confirming the successful removal of the mode directory (`.ruru/modes/<mode-slug>/`) and the associated rules directory (`.roo/rules-<mode-slug>/`, if it existed).
*   **Configuration Update Confirmation:** Confirmation if the mode slug was found and removed from `.ruru/config/build_collections.json`.
*   **Build Script Confirmation:** Confirmation that the `build_roomodes.js` script was executed to update `.roomodes` and related summaries.
*   **Final Summary:** A message indicating the overall success or failure of the deletion process.