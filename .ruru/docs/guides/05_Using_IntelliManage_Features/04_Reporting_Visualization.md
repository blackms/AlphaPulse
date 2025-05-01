+++
# --- Basic Metadata ---
id = "KB-RC-IM-USAGE-REPORTING"
title = "Using IntelliManage: Reporting & Visualization"
status = "draft"
doc_version = "1.0" # Version of IntelliManage reporting features
content_version = 1.0
audience = ["users", "developers", "project_managers"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/10_guide_tutorial.README.md" # Using guide template schema
tags = ["roo-commander", "intellimanage", "guide", "tutorial", "commands", "reporting", "visualization", "metrics", "kanban", "scrum", "board", "burndown", "cfd"]
related_docs = [
    "../README.md", # Link to the KB README
    "../../../DOC-UI-CMD-SPEC-001.md", # Link to the full command spec
    "../../../DOC-METHODOLOGY-GUIDE-001.md", # Link to methodology guide
    "../../../DOC-AI-SPEC-001.md" # Link to AI spec (mentions reporting)
    ]
difficulty = "intermediate" # Assumes basic artifact knowledge
estimated_time = "~10-15 minutes"
prerequisites = ["IntelliManage initialized", "Project artifacts (tasks, etc.) exist", "Understanding of project methodology (Scrum/Kanban) helpful"]
learning_objectives = ["Learn how to generate various project reports using !pm report", "Understand the purpose of different report types (burndown, CFD, summary)", "Learn how to display a textual Kanban board using !pm board", "Understand how to filter and customize report/board output"]
+++
> **🚧 Work in Progress:** The IntelliManage features described in this section are currently under development and not yet available for general use. This documentation reflects planned functionality.

# Using IntelliManage: Reporting & Visualization

## 1. Introduction / Goal 🎯

IntelliManage doesn't just store your project data; it helps you understand project status, track progress, and visualize workflow through built-in reporting and board commands. These features leverage the structured data in your artifact files and adapt to your project's configured methodology.

This guide explains how to use the `!pm report` and `!pm board` commands to gain valuable insights into your projects.

## 2. Generating Reports (`!pm report`) 📈

The `!pm report` command is your primary tool for generating various summaries and metrics based on your project data.

*   **Purpose:** To get summaries of work, track progress against goals (like sprints), and analyze workflow metrics.
*   **Syntax:** `!pm report <report_type> [--project <slug>] [options...]`
*   **Key Arguments:**
    *   `<report_type>` (Required): Specifies the type of report to generate. Common types include:
        *   `project-status`: A general overview of artifact counts by status and type for the project.
        *   `sprint-summary` (Scrum): Summarizes completed, remaining, and potentially spilled-over items for a specific sprint. Requires `--sprint-id`.
        *   `burndown` (Scrum): Generates data (or a textual/Mermaid chart) showing remaining work (effort points or item count) over the sprint duration. Requires `--sprint-id`.
        *   `cfd` (Kanban/Custom): Generates data (or a textual/Mermaid Cumulative Flow Diagram) showing the number of items in each status column over time. Useful for identifying bottlenecks.
        *   *(More report types may be added)*
    *   `--project <slug>` (Required if no active project): Specifies the target project.
    *   `[options...]`: Additional options depend on the `report_type`. Common ones include:
        *   `--sprint-id <sprint_key>` (Required for sprint reports): Specifies the sprint (e.g., `sprint_1`).
        *   `--start-date <YYYY-MM-DD>` / `--end-date <YYYY-MM-DD>` (Optional for CFD/time-based reports): Define the reporting period.
*   **AI Assistance:** The AI Engine processes the artifact data based on the report type and generates the output, often adapting to the project's methodology (Scrum, Kanban, Custom).
*   **Examples:**
    *   `!pm report project-status --project frontend-app`
    *   `!pm report sprint-summary --project backend-api --sprint-id sprint_2`
    *   `!pm report burndown --project frontend-app --sprint-id sprint_1` (May ask for format: text/mermaid)
    *   `!pm report cfd --project backend-api --start-date 2025-04-01` (May ask for format: text/mermaid)

## 3. Visualizing Workflow (`!pm board`) 📋

The `!pm board` command provides a textual representation of a Kanban-style board, showing tasks grouped by their current status.

*   **Purpose:** To visualize the current state of work items as they move through the defined workflow (standard or custom statuses).
*   **Syntax:** `!pm board [--project <slug>] [filters...] [--swimlane <epic|feature>]`
*   **Key Arguments:**
    *   `--project <slug>` (Required if no active project).
    *   `[filters...]` (Optional): Use standard list filters like `--status`, `--assignee`, `--tag`, `--priority` to narrow down which items appear on the board.
    *   `--swimlane <epic|feature>` (Optional): Group rows on the board by their parent Epic or Feature ID for better organization.
*   **Output:** A text-based board with columns representing the workflow statuses (based on the project's methodology configuration) and cards representing the tasks within each status.
*   **AI Assistance:** The AI Engine retrieves the relevant tasks using the Core Logic Engine and formats the output into the board structure, respecting the project's defined statuses.
*   **Examples:**
    *   `!pm board --project frontend-app` (Show all tasks on the board)
    *   `!pm board --project backend-api --assignee "User:Bob"` (Show Bob's tasks on the board)
    *   `!pm board --project frontend-app --swimlane feature` (Group tasks by Feature)
    *   `!pm board --status "🔵 In Progress" --status "🟣 Review"` (Show only In Progress and Review columns)

## 4. Tips for Effective Reporting & Visualization ✨

*   **Keep Statuses Updated:** The accuracy of reports and boards depends entirely on keeping artifact statuses current. Use `!pm update ... --status ...` regularly or confirm AI suggestions.
*   **Use Consistent Metadata:** Ensure artifacts have correct `type`, `priority`, `assignee`, `tags`, and parent links (`epic_id`, `feature_id`) for effective filtering and grouping.
*   **Configure Methodology:** Set the correct `methodology` in `project_config.toml` to enable relevant reports (like burndown for Scrum). Define `custom_statuses` clearly if using the Custom methodology.
*   **Define Sprints (Scrum):** If using Scrum, define your sprints in `project_config.toml` and assign tasks using `sprint_id` for accurate sprint reporting.
*   **Ask the AI:** Don't hesitate to ask the AI Engine to generate specific views or explain metrics (e.g., "Show me all blocked tasks for the backend project", "Explain the CFD report for last week").

## 5. Conclusion ✅

The `!pm report` and `!pm board` commands provide powerful tools for understanding project health, tracking progress, and visualizing workflow within IntelliManage. By keeping your artifact data up-to-date and leveraging these commands (often assisted by the AI Engine), you can gain valuable insights without leaving your development environment.