# Roo Mode Classification Guide (v7 Hierarchy)

This guide provides criteria for classifying Roo Code modes within the v7 hierarchical structure. Assigning modes correctly is crucial for maintaining organizational clarity and enabling effective, automated delegation.

## Hierarchy Overview (v7)

Recall the defined levels:

*   **`000-executive`**: Top-level strategy & coordination.
*   **`010-director`**: Manages project phases, architecture, lifecycle.
*   **`020-lead`**: Manages a specific technical department/domain.
*   **`03x-worker`**: Executes tasks within a department/domain.
*   **`040-assistant`**: Provides focused support/utility functions.

## Classification Criteria

When classifying a new or existing mode, consider its primary **scope of responsibility**, **typical interactions**, and **core function**.

### 1. Determining the Level:

*   **Is the mode's primary function overall project strategy, top-level task initiation, and direct user interaction?**
    *   Yes -> **`000-executive`** (e.g., `roo-commander`)

*   **Is the mode's primary function managing a significant phase of the project (like planning, architecture, onboarding) or overseeing multiple technical domains? Does it translate high-level goals into plans for others?**
    *   Yes -> **`010-director`** (e.g., `project-manager`, `technical-architect`)

*   **Is the mode's primary function coordinating and managing the work of *other* modes within a specific technical department (e.g., Frontend, Backend, QA)? Does it break down director-level tasks for workers in its domain?**
    *   Yes -> **`020-lead`** (e.g., `020-lead-frontend`, `020-lead-backend`). *Note: These often need to be explicitly designed for coordination.*

*   **Is the mode's primary function *executing* specific implementation, testing, design, or documentation tasks, often focused on a particular technology or skill set?**
    *   Yes -> **`03x-worker`** (See section 2 below for determining the department/prefix).

*   **Is the mode's primary function providing a specific, often automated, support service or information-gathering capability used by other modes?**
    *   Yes -> **`040-assistant`** (e.g., `context-resolver`, `discovery-agent`)

### 2. Determining the Worker Department (`03x`):

If a mode is identified as a `worker` (Level 3), determine its department based on its core function:

*   **`030-worker-design`**: Focuses on UI/UX design, visual creation, diagramming. (e.g., `ui-designer`, `diagramer`)
*   **`031-worker-frontend`**: Focuses on implementing the client-side of applications (HTML, CSS, JS, frameworks, accessibility). (e.g., `react-specialist`, `frontend-developer`, `tailwind-specialist`)
*   **`032-worker-backend`**: Focuses on server-side logic, APIs, and application integration. (e.g., `api-developer`, `fastapi-developer`)
*   **`033-worker-database`**: Focuses specifically on database design, querying, and administration. (e.g., `database-specialist`, `mongodb-specialist`)
*   **`034-worker-qa`**: Focuses on testing, quality assurance, and validation. (e.g., `e2e-tester`, `integration-tester`)
*   **`035-worker-devops`**: Focuses on CI/CD, infrastructure, containerization, deployment. (e.g., `cicd-specialist`, `containerization-developer`)
*   **`039-worker-cross-functional`**: Focuses on tasks supporting the overall development process across multiple domains. (e.g., `git-manager`, `technical-writer`, `bug-fixer`, `code-reviewer`, `refactor-specialist`)

### 3. Assigning Metadata:

Once classified, update the mode's metadata:

*   **`Level`**: Set to the full level identifier (e.g., `031-worker-frontend`).
*   **`Categories`**: List the primary department(s) (e.g., `["Frontend"]`, `["Backend", "API"]`, `["Cross-Functional", "SCM"]`). Use consistent category names.
*   **`DelegatesTo`**: List mode slugs this mode typically delegates *sub-tasks* to (e.g., a `frontend-lead` might delegate to `react-specialist`).
*   **`EscalatesTo`**: List mode slugs this mode escalates *problems* or *broader issues* to (e.g., a `worker` might escalate to its `lead` or a `director`).
*   **`ReportsTo`**: List mode slugs this mode typically reports *completion* or *status* to (usually the level above that delegated the task).

## Considerations:

*   **Primary Focus:** Classify based on the mode's *main* purpose, even if it occasionally performs tasks overlapping with other levels/departments.
*   **Evolution:** Mode roles might evolve. Re-evaluate classification periodically.
*   **New Leads:** The `020-lead` level often requires designing specific modes for coordination within that department. Generalist modes might initially fill this role but may need dedicated Lead modes later.
*   **Consistency:** Use the defined level names and strive for consistent category naming.

This guide should help in consistently classifying modes within the v7 hierarchy.