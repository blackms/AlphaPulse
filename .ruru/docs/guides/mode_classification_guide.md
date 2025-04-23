# Roo Mode Classification Guide (v7.1 Structure)

This guide provides criteria for classifying Roo Code modes within the **v7.1 directory structure**. Assigning modes correctly is crucial for maintaining organizational clarity and enabling effective, automated delegation.

## Hierarchy Overview (v7.1)

The v7.1 structure uses semantic directory names for classification:

*   **`executive`**: Top-level strategy & coordination.
*   **`director`**: Manages project phases, architecture, lifecycle.
*   **`lead`**: Manages a specific technical domain.
*   **`worker`**: Executes tasks within a specific domain.
*   **`assistant`**: Provides focused support/utility functions.
*   **`footgun`**: Experimental/expert modes, potentially bypassing safeguards.

## Classification Criteria

When classifying a new or existing mode, consider its primary **scope of responsibility**, **typical interactions**, and **core function**.

### 1. Determining the Classification (Top-Level Directory):

*   **Is the mode's primary function overall project strategy, top-level task initiation, and direct user interaction?**
    *   Yes -> **`executive`** (e.g., `roo-commander`)

*   **Is the mode's primary function managing a significant phase of the project (like planning, architecture, onboarding) or overseeing multiple technical domains? Does it translate high-level goals into plans for others?**
    *   Yes -> **`director`** (e.g., `project-manager`, `technical-architect`)

*   **Is the mode's primary function coordinating and managing the work of *other* modes within a specific technical domain (e.g., Frontend, Backend, QA)? Does it break down director-level tasks for workers in its domain?**
    *   Yes -> **`lead`** (e.g., `frontend-lead`, `backend-lead`). *Note: These often need to be explicitly designed for coordination.*

*   **Is the mode's primary function *executing* specific implementation, testing, design, or documentation tasks, often focused on a particular technology or skill set?**
    *   Yes -> **`worker`** (See section 2 below for determining the domain).

*   **Is the mode's primary function providing a specific, often automated, support service or information-gathering capability used by other modes?**
    *   Yes -> **`assistant`** (e.g., `context-resolver`, `discovery-agent`)

*   **Is the mode an experimental or expert variant, potentially bypassing standard safeguards?**
    *   Yes -> **`footgun`** (e.g., `footgun-code`)

### 2. Determining the Domain (Second-Level Directory for Leads/Workers):

If a mode is classified as `lead` or `worker`, determine its domain based on its core function:

*   **`core`**: Foundational capabilities (often for `footgun` or core system modes).
*   **`design`**: Focuses on UI/UX design, visual creation, diagramming. (e.g., `ui-designer`, `diagramer`)
*   **`frontend`**: Focuses on implementing the client-side of applications (HTML, CSS, JS, frameworks, accessibility). (e.g., `react-specialist`, `frontend-developer`, `tailwind-specialist`)
*   **`backend`**: Focuses on server-side logic, APIs, and application integration. (e.g., `api-developer`, `fastapi-developer`)
*   **`database`**: Focuses specifically on database design, querying, and administration. (e.g., `database-specialist`, `mongodb-specialist`)
*   **`qa`**: Focuses on testing, quality assurance, and validation. (e.g., `e2e-tester`, `integration-tester`)
*   **`devops`**: Focuses on CI/CD, infrastructure, containerization, deployment. (e.g., `cicd-specialist`, `docker-compose-specialist`)
*   **`auth`**: Focuses on authentication and authorization implementation. (e.g., `firebase-auth-specialist`, `supabase-auth-specialist`)
*   **`ai-ml`**: Focuses on Artificial Intelligence and Machine Learning tasks. (e.g., `huggingface-specialist`, `openai-specialist`)
*   **`cross-functional`**: Focuses on tasks supporting the overall development process across multiple domains. (e.g., `git-manager`, `technical-writer`, `bug-fixer`, `code-reviewer`, `refactor-specialist`)
*   **`utility`**: General-purpose helper modes.

*(Note: Sub-domains can be used for further organization within a domain, e.g., `worker/frontend/react`)*

### 3. Assigning Metadata (in `{id}.mode.md`):

Once classified, ensure the mode's TOML frontmatter reflects this:

*   **`classification`**: Set to the top-level directory name (e.g., `"worker"`).
*   **`domain`**: Set to the second-level directory name (e.g., `"frontend"`).
*   **`sub_domain`**: Set to the third-level directory name if used, otherwise omit or set to `null`.
*   **`categories`**: List relevant functional areas (e.g., `["Frontend", "React", "UI"]`, `["Backend", "API", "Python"]`, `["Cross-Functional", "SCM"]`). Use consistent category names.
*   **`delegate_to`**: List mode slugs this mode typically delegates *sub-tasks* to (e.g., a `frontend-lead` might delegate to `react-specialist`).
*   **`escalate_to`**: List mode slugs this mode escalates *problems* or *broader issues* to (e.g., a `worker` might escalate to its `lead` or a `director`).
*   **`reports_to`**: List mode slugs this mode typically reports *completion* or *status* to (usually the level above that delegated the task).

## Considerations:

*   **Primary Focus:** Classify based on the mode's *main* purpose, even if it occasionally performs tasks overlapping with other classifications/domains.
*   **Evolution:** Mode roles might evolve. Re-evaluate classification periodically.
*   **Consistency:** Use the defined classification/domain names and strive for consistent category naming in metadata.

This guide should help in consistently classifying modes within the v7.1 structure.