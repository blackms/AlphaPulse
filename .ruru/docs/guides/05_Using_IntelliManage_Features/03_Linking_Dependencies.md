+++
# --- Basic Metadata ---
id = "KB-RC-IM-USAGE-LINKING"
title = "Using IntelliManage: Linking Artifacts & Managing Dependencies"
status = "draft"
doc_version = "1.0" # Version of IntelliManage linking features
content_version = 1.0
audience = ["users", "developers", "project_managers"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/10_guide_tutorial.README.md" # Using guide template schema
tags = ["roo-commander", "intellimanage", "guide", "tutorial", "commands", "linking", "dependencies", "hierarchy", "traceability", "usage"]
related_docs = [
    "../README.md", # Link to the KB README
    "../../../DOC-UI-CMD-SPEC-001.md", # Link to the full command spec
    "../../../DOC-SCHEMA-001.md", # Link to schemas (defines linking fields)
    "../../../DOC-FUNC-SPEC-001.md", # Link to core functionality spec
    "../../02_Core_Concepts/06_IntelliManage_Artifacts.md" # Link to artifact descriptions
    ]
difficulty = "beginner"
estimated_time = "~10-15 minutes"
prerequisites = ["IntelliManage initialized", "Basic understanding of !pm commands and artifact types"]
learning_objectives = ["Understand the difference between hierarchical and sequential links", "Learn how to create and remove parent/child links using !pm link/unlink --parent", "Learn how to create and remove dependency links using !pm link/unlink --depends-on", "Understand how dependency links affect task suggestions"]
> **🚧 Work in Progress:** The IntelliManage features described in this section are currently under development and not yet available for general use. This documentation reflects planned functionality.
+++

# Using IntelliManage: Linking Artifacts & Managing Dependencies

## 1. Introduction / Goal 🎯

Linking artifacts together is a core feature of IntelliManage that provides crucial **traceability** and helps manage **dependencies** between work items. Proper linking ensures you can see how individual tasks contribute to larger features and epics, and understand the required order of execution.

This guide explains the two main types of links and how to manage them using the `!pm link` and `!pm unlink` commands.

## 2. Types of Links 🔗

1.  **Hierarchical Links (Parent/Child):** These links define the work breakdown structure (Initiative > Epic > Feature > Task/Story/Bug). They show *what* larger goal a piece of work contributes to. These are stored in fields like `epic_id` (on Features) and `feature_id` (on Tasks/Stories/Bugs).
2.  **Sequential Dependency Links (`depends_on`):** These links define prerequisites between Tasks, Stories, or Bugs. They indicate that one item cannot be started (or sometimes considered fully 'Done') until another specific item is completed.

## 3. Managing Hierarchical Links (`--parent`) 👨‍👩‍👧‍👦

These links establish the primary structure of your project.

*   **Creating Links:**
    *   **During Creation:** The easiest way is to specify the parent when creating the child item using `--epic <EPIC-ID>` (for Features) or `--feature <FEAT-ID>` (for Tasks/Stories/Bugs).
        ```
        !pm create feature --project <slug> --epic EPIC-001 --title "..."
        !pm create task --project <slug> --feature FEAT-002 --title "..." --type "🛠️ Task"
        ```
    *   **Explicitly with `!pm link`:** If you forget during creation or need to link later.
        *   **Syntax:** `!pm link <CHILD_ID> --parent <PARENT_ID>`
        *   **Behavior:** IntelliManage automatically determines the correct field (`epic_id` or `feature_id`) based on the types of the child and parent artifacts.
        *   **Example (Link Task to Feature):** `!pm link TASK-105 --parent FEAT-002`
        *   **Example (Link Feature to Epic):** `!pm link FEAT-003 --parent EPIC-001`
*   **Viewing Links:**
    *   Use `!pm show <type> <ID>`. The relevant parent ID field (e.g., `feature_id: "FEAT-002"`) will be displayed in the TOML metadata.
*   **Removing Links:**
    *   **Syntax:** `!pm unlink <CHILD_ID> --parent`
    *   **Behavior:** Removes the value from the relevant parent ID field (`epic_id` or `feature_id`) in the child artifact's TOML.
    *   **Example:** `!pm unlink TASK-105 --parent`

## 4. Managing Sequential Dependencies (`--depends-on`) ➡️

These links define the order in which tasks need to be completed.

*   **Creating Links:**
    *   **Syntax:** `!pm link <TASK_ID> --depends-on <OTHER_TASK_ID>`
    *   **Behavior:** Adds the `<OTHER_TASK_ID>` to the `depends_on` array in `<TASK_ID>`'s TOML frontmatter. You can link multiple dependencies by repeating the command or potentially providing a comma-separated list (check `!pm help link`).
    *   **Example:** `!pm link TASK-102 --depends-on TASK-101` (TASK-102 cannot start until TASK-101 is done)
    *   **Example:** `!pm link TASK-103 --depends-on TASK-101 --depends-on TASK-102` (If supported, otherwise run twice)
*   **Viewing Links:**
    *   Use `!pm show <type> <ID>`. The `depends_on` array will be listed in the TOML metadata.
    *   The `!pm list` and `!pm next` commands often indicate if a task has unmet dependencies.
*   **Impact on Workflow:** Tasks with unmet dependencies (items listed in `depends_on` that are not "🟢 Done" or "🧊 Archived") will typically not be suggested by `!pm next` and may be visually flagged in board views.
*   **Removing Links:**
    *   **Syntax:** `!pm unlink <TASK_ID> --depends-on <OTHER_TASK_ID>`
    *   **Behavior:** Removes the specified `<OTHER_TASK_ID>` from the `depends_on` array in `<TASK_ID>`'s TOML.
    *   **Example:** `!pm unlink TASK-102 --depends-on TASK-101`

## 5. AI Assistance 🤖

The IntelliManage AI Engine can help with linking:

*   **Suggestion:** When creating new artifacts or discussing related items in chat, the AI may suggest potential parent/child or dependency links based on context and keywords.
*   **Confirmation:** Always confirm AI-suggested links before they are applied to ensure accuracy.

## 6. Best Practices 👍

*   **Link Hierarchically:** Always link Features to Epics and Tasks/Stories/Bugs to Features during creation whenever possible. This maintains the core structure.
*   **Use `depends_on` for True Blockers:** Only add sequential dependencies when one task genuinely cannot start before another is finished. Avoid creating overly complex dependency chains that hinder parallel work.
*   **Keep Links Updated:** If relationships change (e.g., a task moves to a different feature), use `!pm update` or `!pm link/unlink` to correct the links.

## 7. Conclusion ✅

Linking artifacts is essential for maintaining traceability and managing workflow dependencies in IntelliManage. Use the `--parent` option (or specify during creation) for hierarchical structure and the `--depends-on` option for sequential prerequisites. Leverage the `!pm link` and `!pm unlink` commands, along with AI suggestions, to keep your project structure accurate and dependencies clear.