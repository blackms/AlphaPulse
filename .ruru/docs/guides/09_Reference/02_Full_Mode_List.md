+++
# --- Basic Metadata ---
id = "KB-RC-REF-MODE-LIST"
title = "Reference: Full Mode List"
status = "draft"
doc_version = "1.0" # Version of this reference document
content_version = 1.0
audience = ["users", "developers", "architects", "contributors", "ai_modes"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md" # Using general doc template structure
tags = ["roo-commander", "modes", "reference", "list", "directory", "generated", "index"]
related_docs = [
    "../README.md", # Link to the KB README
    "01_Glossary.md",
    "../../04_Understanding_Modes/03_Mode_Directory_Reference.md", # Link to the reference explanation
    "../../../../.ruru/modes/roo-commander/kb/kb-available-modes-summary.md" # Link to the actual generated summary
    ]
difficulty = "beginner"
estimated_time = "~2 minutes"
prerequisites = ["None"]
learning_objectives = ["Locate the definitive list of available Roo Commander modes", "Understand that this list is automatically generated"]
+++

# Reference: Full Mode List

## 1. Introduction / Purpose 🎯

This document serves as a direct pointer to the definitive, up-to-date list of all AI agent modes currently available within your Roo Commander workspace configuration.

## 2. Authoritative Source: Generated Summary File 📌

The complete and authoritative list of modes is **automatically generated** by a build script (typically `build_roomodes.js`) and stored in Roo Commander's own Knowledge Base.

*   **Location:** **`.ruru/modes/roo-commander/kb/kb-available-modes-summary.md`**

**Please refer directly to this file for the most current list.**

## 3. Content Overview 📄

The `kb-available-modes-summary.md` file provides a concise overview of each mode, typically including:

*   Mode Slug (e.g., `dev-react`)
*   Display Name & Emoji (e.g., `⚛️ React Specialist`)
*   A brief Summary of its core purpose.

Modes are usually grouped by classification or domain for easier navigation.

## 4. Finding More Details 💡

For more detailed information on a specific mode's capabilities, workflow, rules, or KB content, please consult:

*   The mode's definition file: `.ruru/modes/[mode_slug]/[mode_slug].mode.md`
*   The enhanced Mode Selection Guide: `04_Understanding_Modes/02_Mode_Selection_Guide.md`

## 5. Keeping the List Updated 🔄

Remember that this list is generated based on the mode definition files present in `.ruru/modes/` and registered in `.roomodes`. If you add or remove custom modes, run the `node build_roomodes.js` script from your workspace root to update both the `.roomodes` registry and the `kb-available-modes-summary.md` file.

## 6. Conclusion ✅

Use the linked `kb-available-modes-summary.md` file as your primary reference for discovering the available modes within your Roo Commander setup.