+++
# --- Basic Metadata ---
id = "KB-RC-IM-USAGE-METHODOLOGIES"
title = "Using IntelliManage: Working with Methodologies (Scrum, Kanban, Custom)"
status = "draft"
doc_version = "1.0" # Version of IntelliManage methodology support
content_version = 1.0
audience = ["users", "developers", "project_managers"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/10_guide_tutorial.README.md" # Using guide template schema
tags = ["roo-commander", "intellimanage", "guide", "tutorial", "methodology", "scrum", "kanban", "custom", "agile", "workflow", "usage"]
related_docs = [
    "../README.md", # Link to the KB README
    "../../../DOC-METHODOLOGY-GUIDE-001.md", # Link to the formal spec
    "../../../DOC-SCHEMA-001.md", # Link to schemas (shows fields like sprint_id)
    "../../../DOC-UI-CMD-SPEC-001.md", # Link to command spec
    "../../../DOC-SETUP-GUIDE-001.md" # Link to setup guide (where methodology is set)
    ]
difficulty = "intermediate" # Assumes basic artifact knowledge
estimated_time = "~15 minutes"
prerequisites = ["IntelliManage initialized", "Basic understanding of !pm commands", "Familiarity with Scrum/Kanban concepts (optional)"]
learning_objectives = ["Understand how IntelliManage adapts to Scrum, Kanban, and Custom methodologies", "Learn how to manage sprints in Scrum projects", "Learn how to leverage Kanban features like board views and WIP limits", "Understand how to configure and use custom workflows"]
> **🚧 Work in Progress:** The IntelliManage features described in this section are currently under development and not yet available for general use. This documentation reflects planned functionality.
+++

# Using IntelliManage: Working with Methodologies (Scrum, Kanban, Custom)

## 1. Introduction / Goal 🎯

IntelliManage is designed to be flexible and adapt to your team's preferred way of working. On a **per-project basis**, you can configure it to support standard Agile methodologies like **Scrum** or **Kanban**, define a completely **Custom** workflow, or use **None** for basic tracking.

This guide explains how to work effectively within each methodology using IntelliManage commands and AI assistance. The chosen methodology primarily influences how artifact statuses are interpreted, validated, and visualized.

Refer to the **Methodology Implementation Guide (`DOC-METHODOLOGY-GUIDE-001`)** for the technical specification.

## 2. Setting Your Project's Methodology ⚙️

You define the methodology when initializing a project or by updating its configuration:

*   **Initialization:** `!pm init project <slug> --name "..." --methodology <Scrum|Kanban|Custom|None>`
*   **Update:** `!pm config project <slug> --set methodology=<Scrum|Kanban|Custom|None>`

*(See the Setup Guide (`DOC-SETUP-GUIDE-001`) for more details on configuration commands).*

## 3. Working with Scrum 🔄

If your project's `methodology = "Scrum"`:

*   **Sprints:**
    *   Define sprints (ID, start/end dates, goal) in `project_config.toml` under the `[sprints]` table.
        *   *Example Config Command:* `!pm config project <slug> --set sprints.sprint_3='{ start_date = "2025-06-01", end_date = "2025-06-14", goal = "Implement search" }'`
    *   Assign Tasks/Stories/Bugs to a sprint using the `--sprint-id <sprint_key>` flag during creation or update.
        *   *Example Create:* `!pm create task ... --sprint-id sprint_3`
        *   *Example Update:* `!pm update task TASK-101 --sprint-id sprint_3`
*   **Backlogs:**
    *   **Product Backlog:** Items without a `sprint_id` or with `status = "⚪️ Backlog"`. Use `!pm list tasks --status "⚪️ Backlog"` to view. Prioritize using the `--priority` field.
    *   **Sprint Backlog:** Items assigned to the current/specific `sprint_id`. Use `!pm list tasks --sprint-id <sprint_key>` to view.
*   **Workflow:** Use standard statuses: `⚪️ Backlog` -> `🟡 To Do` (selected for sprint) -> `🔵 In Progress` -> `🟣 Review` -> `🟢 Done`. The Core Logic Engine may enforce valid transitions.
*   **AI Assistance:**
    *   `!pm report burndown --sprint-id <sprint_key>`: Generate a textual or Mermaid burndown chart.
    *   `!pm report sprint-summary --sprint-id <sprint_key>`: Summarize completed/remaining work.
    *   Ask AI to suggest items for the next sprint based on priority and velocity (requires effort estimation field).

## 4. Working with Kanban 📊

If your project's `methodology = "Kanban"`:

*   **Workflow Visualization:** The flow is defined by the sequence of `status` values. Use `!pm board` to get a textual Kanban board view.
    *   *Example:* `!pm board --project backend-api`
    *   *Example (Swimlanes):* `!pm board --swimlane epic` (Groups rows by Epic)
*   **WIP Limits (Work In Progress):**
    *   Optionally define WIP limits per status column in `project_config.toml`.
        *   *Example Config Command:* `!pm config project <slug> --set wip_limits.'🔵 In Progress'=5`
    *   The AI or system can warn you if adding an item to a status column would exceed its WIP limit.
*   **Flow Metrics:**
    *   If status changes are timestamped (future enhancement), the AI can calculate metrics like Cycle Time and Lead Time.
    *   `!pm report cfd`: Generate a Cumulative Flow Diagram (textual/Mermaid) to visualize flow.
    *   Ask AI to identify bottlenecks (items stuck in a status for long periods).
*   **Status Updates:** Focus on accurately updating the `status` field as items move through the workflow using `!pm update task <ID> --status "..."`.

## 5. Working with Custom Methodologies 🛠️

If your project's `methodology = "Custom"`:

*   **Define Statuses:** You **MUST** define your workflow states in `project_config.toml` using the `custom_statuses` array.
    *   *Example Config Command:* `!pm config project <slug> --set custom_statuses='["⚪️ Idea", "⚫️ Refinement", "🟡 Ready", "🔵 Dev", "🟣 Review", "🟢 Done"]'`
*   **Status Updates:** Use `!pm update task <ID> --status "Your Custom Status"` to move items through your defined workflow. The system validates against the `custom_statuses` list.
*   **AI Adaptation:**
    *   The AI Engine reads your `custom_statuses` and adapts its understanding.
    *   `!pm board` and `!pm report` commands will use your custom statuses for columns and aggregation.
    *   AI guidance specific to Scrum/Kanban won't apply, but general task management assistance remains.

## 6. Working with "None" Methodology 🚫

If your project's `methodology = "None"`:

*   **Basic Tracking:** Provides simple task tracking without enforcing specific processes.
*   **Minimal Statuses:** Uses a basic set like `🟡 To Do`, `🔵 In Progress`, `🟢 Done`. Status transitions are generally not restricted.
*   **AI Assistance:** Limited to core functions like artifact creation, linking, listing, and basic status updates. Methodology-specific reports (burndown, CFD) are unavailable.

## 7. Conclusion ✅

IntelliManage offers significant flexibility by allowing you to choose and configure the project management methodology that best suits each project within your workspace. Whether you prefer the iterative structure of Scrum, the flow-based approach of Kanban, or a completely custom workflow, you can configure IntelliManage accordingly. Remember to keep your `project_config.toml` updated and leverage the `!pm` commands and AI assistance tailored to your chosen methodology for optimal project management.