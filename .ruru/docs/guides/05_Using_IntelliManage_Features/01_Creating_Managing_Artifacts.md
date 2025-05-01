+++
# --- Basic Metadata ---
id = "KB-RC-IM-USAGE-CRUD"
title = "Using IntelliManage: Creating & Managing Artifacts"
status = "draft"
doc_version = "1.0" # Version of IntelliManage commands being described
content_version = 1.0
audience = ["users", "developers", "project_managers"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/10_guide_tutorial.README.md" # Using guide template schema
tags = ["roo-commander", "intellimanage", "guide", "tutorial", "commands", "crud", "tasks", "epics", "features", "usage"]
related_docs = [
    "../README.md", # Link to the KB README
    "../../../DOC-UI-CMD-SPEC-001.md", # Link to the full command spec
    "../../../DOC-SCHEMA-001.md", # Link to schemas (defines fields)
    "../../../DOC-FS-SPEC-001.md" # Link to file system structure
    ]
difficulty = "beginner"
estimated_time = "~15-20 minutes"
prerequisites = ["IntelliManage initialized", "Understanding of IntelliManage Artifacts"]
learning_objectives = ["Learn the syntax for core !pm commands (create, list, show, update, delete, archive)", "Understand how to manage different artifact types", "Know how to specify required and optional fields", "Learn how to target specific projects"]
+++
> **🚧 Work in Progress:** The IntelliManage features described in this section are currently under development and not yet available for general use. This documentation reflects planned functionality.

# Using IntelliManage: Creating & Managing Artifacts

## 1. Introduction / Goal 🎯

This guide explains how to use the fundamental `!pm` commands within the chat interface to perform Create, Read, Update, and Delete (CRUD) operations on your IntelliManage project artifacts (Initiatives, Epics, Features, Tasks, Stories, Bugs).

Mastering these commands allows you to efficiently manage your project's work items directly from your development environment. For a complete reference of all commands and options, see the **User Interaction & Command Specification (`DOC-UI-CMD-SPEC-001`)**.

## 2. Prerequisites Checklist ✅

*   [ ] IntelliManage has been initialized in your workspace (See `DOC-SETUP-GUIDE-001`).
*   [ ] You have at least one project configured (e.g., using `!pm init project ...`).
*   [ ] You understand the basic IntelliManage artifact types (Epic, Feature, Task, etc. - see `02_Core_Concepts/06_IntelliManage_Artifacts.md`).
*   [ ] You have set your active project using `!pm set-active <project_slug>` or will use the `--project <project_slug>` flag.

## 3. Creating Artifacts (`!pm create`) ➕

This command creates a new artifact file with the specified details.

*   **Purpose:** To add new Initiatives, Epics, Features, Tasks, Stories, or Bugs to your project.
*   **Syntax:** `!pm create <type> --title "..." [options...]`
*   **Key Arguments:**
    *   `<type>` (Required): `initiative`, `epic`, `feature`, `task`, `story`, `bug`.
    *   `--title "..."` (Required): The title for the new artifact.
    *   `--project <slug>` (Required if no active project): Specifies the target project.
    *   `--epic <EPIC-ID>` (Required for `feature`): Links the new feature to an Epic.
    *   `--feature <FEAT-ID>` (Required for `task`, `story`, `bug`): Links the new item to a Feature.
    *   `--type <value>` (Required for `task`/`story`/`bug`): Sets the specific type (e.g., `--type "🛠️ Task"`).
    *   `--status <value>` (Optional): Sets initial status (defaults usually to Backlog/ToDo).
    *   `--priority <value>` (Optional): Sets priority.
    *   `--assignee <user/role>` (Optional): Assigns the item.
    *   `--tags "tag1,tag2"` (Optional): Adds tags.
    *   `--description "..."` (Optional): Adds initial Markdown description.
*   **Examples:**
    *   `!pm create epic --project backend-api --title "Refactor Authentication Module"`
    *   `!pm create task --project frontend-app --feature FEAT-001 --type "🛠️ Task" --title "Implement login form UI" --priority "🔼 High" --tags "ui,auth,react"`
    *   `!pm create bug --project backend-api --feature FEAT-010 --type "🐞 Bug" --title "Login fails with special characters" --reporter "User:Alice"`

## 4. Listing Artifacts (`!pm list`) 📋

This command displays a list of artifacts matching specified criteria.

*   **Purpose:** To find and view multiple artifacts based on filters like status, type, assignee, tags, or parent links.
*   **Syntax:** `!pm list <type> [filters...]`
*   **Key Arguments:**
    *   `<type>` (Required): `initiatives`, `epics`, `features`, `tasks` (plural).
    *   `--project <slug>` (Required if no active project).
    *   `--status <value>` (Optional): Filter by status.
    *   `--type <value>` (Optional): Filter by specific type (e.g., `--type "🐞 Bug"` when listing tasks).
    *   `--priority <value>` (Optional): Filter by priority.
    *   `--assignee <user/role>` (Optional): Filter by assignee.
    *   `--tag <tag>` (Optional): Filter by a specific tag.
    *   `--epic <EPIC-ID>` (Optional): List features or tasks linked to this Epic.
    *   `--feature <FEAT-ID>` (Optional): List tasks linked to this Feature.
    *   `--sort <field:asc|desc>` (Optional): Sort results (e.g., `--sort priority:desc`, `--sort updated_date:asc`).
    *   `--limit <number>` (Optional): Limit the number of results shown.
*   **Examples:**
    *   `!pm list tasks --project frontend-app --status "🟡 To Do"`
    *   `!pm list features --project backend-api --epic EPIC-005`
    *   `!pm list tasks --assignee "AI:dev-react" --sort created_date:desc --limit 10`

## 5. Showing Artifact Details (`!pm show`) 📄

This command displays the full details (TOML metadata and Markdown content) of a single artifact.

*   **Purpose:** To view the complete information for a specific work item.
*   **Syntax:** `!pm show <type> <ID> [--project <slug>]`
*   **Key Arguments:**
    *   `<type>` (Required): `initiative`, `epic`, `feature`, `task`, `story`, `bug`.
    *   `<ID>` (Required): The unique ID of the artifact (e.g., `TASK-101`, `EPIC-005`).
    *   `--project <slug>` (Required if ID isn't globally unique or no active project).
*   **Example:**
    *   `!pm show task TASK-101 --project frontend-app`

## 6. Updating Artifacts (`!pm update`) ✏️

This command modifies the metadata or content of an existing artifact.

*   **Purpose:** To change status, priority, assignee, title, description, tags, or manage subtasks.
*   **Syntax:** `!pm update <type> <ID> --field <new_value> [other_updates...]`
*   **Key Arguments:**
    *   `<type>` (Required): `initiative`, `epic`, `feature`, `task`, `story`, `bug`.
    *   `<ID>` (Required): The unique ID of the artifact to update.
    *   `--project <slug>` (Required if ID isn't globally unique or no active project).
    *   `--title "..."` (Optional): Change the title.
    *   `--status <value>` (Optional): Change the status (validated by methodology).
    *   `--priority <value>` (Optional): Change the priority.
    *   `--assignee <user/role>` (Optional): Change the assignee.
    *   `--add-tag <tag>` (Optional): Add a tag to the `tags` array.
    *   `--remove-tag <tag>` (Optional): Remove a tag from the `tags` array.
    *   `--description "..."` (Optional): Replace the entire Markdown description.
    *   `--add-subtask "..."` (Optional): Add a new unchecked subtask (`- [ ] ...`) to the Markdown body.
    *   `--check-subtask "..."` (Optional): Find and check off (`- [x]`) a subtask matching the description.
    *   `--uncheck-subtask "..."` (Optional): Find and uncheck (`- [ ]`) a subtask matching the description.
*   **Important:** The `updated_date` field in the TOML is automatically updated on every successful update.
*   **Examples:**
    *   `!pm update task TASK-101 --project frontend-app --status "🔵 In Progress" --assignee "User:Dev1"`
    *   `!pm update feature FEAT-001 --add-tag "refinement-needed"`
    *   `!pm update task TASK-101 --check-subtask "Implement basic layout"`

## 7. Deleting / Archiving Artifacts (`!pm delete`, `!pm archive`) 🗑️🧊

These commands remove or archive artifacts. **Use with caution!**

*   **Purpose:** To permanently remove (`delete`) or move to an archive folder (`archive`) artifacts that are no longer needed or are considered fully closed.
*   **Syntax:**
    *   `!pm delete <type> <ID> [--project <slug>]`
    *   `!pm archive <type> <ID> [--project <slug>]`
*   **Key Arguments:**
    *   `<type>` (Required): `initiative`, `epic`, `feature`, `task`, `story`, `bug`.
    *   `<ID>` (Required): The unique ID of the artifact.
    *   `--project <slug>` (Required if ID isn't globally unique or no active project).
*   **Confirmation:** Both commands **require explicit user confirmation** before execution due to their destructive/permanent nature.
*   **Archive Behavior:** The `archive` command typically moves the file to a corresponding subdirectory within `.ruru/projects/[project_slug]/archive/` and sets the status to `"🧊 Archived"`.
*   **Examples:**
    *   `!pm delete task TASK-099 --project backend-api` (Will ask for confirmation)
    *   `!pm archive epic EPIC-001 --project frontend-app` (Will ask for confirmation)

## 8. Targeting Projects (`--project`) 🏢

Remember:
*   If you only have one project configured, commands usually don't need `--project`.
*   If you have multiple projects, use `!pm set-active <slug>` to set a default for your session.
*   Use `--project <slug>` on any command to explicitly target a specific project, overriding the active setting.

## 9. Conclusion ✅

These core CRUD and listing commands form the foundation for interacting with your IntelliManage project artifacts. By using them effectively (either directly or through natural language prompts interpreted by the AI), you can keep your project data organized, up-to-date, and readily accessible within your development environment. Always refer to `!pm help` or the full command specification (`DOC-UI-CMD-SPEC-001`) for more details and advanced options.