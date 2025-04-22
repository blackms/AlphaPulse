# Roo Mode Hierarchy (v7)

This document outlines the standardized hierarchical structure for Roo Code modes, designed to facilitate clear delegation pathways and organization.

## Levels Overview

The hierarchy consists of 5 main levels, identified by numeric prefixes:

1.  **`00x-executive`**: Top-level strategy, coordination, and primary user interaction.
2.  **`01x-director`**: Oversees major project phases, architecture, or lifecycle stages. Manages Leads and Workers.
3.  **`02x-lead`**: Manages and coordinates work within a specific technical department. Delegates tasks to Workers within their department.
4.  **`03x-worker`**: Executes specific tasks within a defined department/domain. Possesses specialized or general implementation skills. Reports to Leads or Directors. (Remains `03x` as it's a group prefix).
5.  **`04x-assistant`**: Provides focused support, information gathering, or automated utility functions to any level.

## Detailed Levels and Departments

### Level 0: `00x-executive`
*   **Role:** Overall project command, strategic decision-making, primary interface with the user. Initiates top-level tasks.
*   **Example Modes:** `roo-commander`

### Level 1: `01x-director`
*   **Role:** Manages broad project aspects like planning, architecture, or onboarding. Translates executive goals into actionable plans for Leads/Workers.
*   **Example Modes:** `project-manager`, `technical-architect`, `project-onboarding`

### Level 2: `02x-lead`
*   **Role:** Coordinates a team of workers within a specific technical domain. Breaks down director-level tasks into specific implementation tasks for workers. Ensures quality and consistency within the department.
*   **Example Departments (Lead Mode Examples):**
    *   `020-lead-design` (*Potential New Mode*)
    *   `020-lead-frontend` (*Potential New Mode*)
    *   `020-lead-backend` (*Potential New Mode*)
    *   `020-lead-database` (*Potential New Mode*)
    *   `020-lead-qa` (*Potential New Mode*)
    *   `020-lead-devops` (*Potential New Mode*)

### Level 3: `03x-worker`
*   **Role:** The primary execution level. Implements features, fixes bugs, performs tests, writes documentation, etc., within their designated department.
*   **Example Departments (Worker Mode Examples):**
    *   **`030-worker-design`**: `ui-designer`, `diagramer`, `one-shot-web-designer`
    *   **`031-worker-frontend`**: `react-specialist`, `vuejs-developer`, `tailwind-specialist`, `frontend-developer`, `accessibility-specialist`
    *   **`032-worker-backend`**: `api-developer`, `fastapi-developer`, `django-developer`, `php-laravel-developer`
    *   **`033-worker-database`**: `database-specialist`, `mongodb-specialist`, `elasticsearch-specialist`
    *   **`034-worker-qa`**: `e2e-tester`, `integration-tester`
    *   **`035-worker-devops`**: `cicd-specialist`, `containerization-developer`, `infrastructure-specialist`
    *   **`039-worker-cross-functional`**: `git-manager`, `technical-writer`, `refactor-specialist`, `bug-fixer`, `code-reviewer`

### Level 4: `04x-assistant`
*   **Role:** Provides specific, often automated, support functions or information retrieval. Can be called upon by any other level.
*   **Example Modes:** `discovery-agent`, `context-resolver`, `file-repair-specialist`

## Metadata Field

The `Level` field in each mode's metadata should store the full level identifier (e.g., `Level: 031-worker-frontend`).