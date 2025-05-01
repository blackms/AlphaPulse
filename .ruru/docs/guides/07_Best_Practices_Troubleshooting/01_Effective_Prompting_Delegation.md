+++
# --- Basic Metadata ---
id = "KB-RC-BESTPRACTICE-PROMPTING"
title = "Best Practices: Effective Prompting & Delegation"
status = "draft"
doc_version = "1.0" # Version of this guide
content_version = 1.0
audience = ["users", "developers", "project_managers"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/10_guide_tutorial.README.md" # Using guide template schema
tags = ["roo-commander", "intellimanage", "best-practices", "prompting", "delegation", "communication", "ai-interaction", "guide", "tips"]
related_docs = [
    "../README.md", # Link to the KB README
    "../../03_Getting_Started/03_Basic_Workflow_Example.md",
    "../../04_Understanding_Modes/02_Mode_Selection_Guide.md"
    ]
difficulty = "beginner"
estimated_time = "~10-15 minutes"
prerequisites = ["Basic interaction with Roo Commander modes"]
learning_objectives = ["Understand the importance of clear communication with AI agents", "Learn techniques for writing effective prompts and delegation messages", "Know how to provide sufficient context", "Understand how to specify constraints and desired outcomes"]
+++

# Best Practices: Effective Prompting & Delegation

## 1. Introduction / Goal 🎯

Communicating effectively with AI agents is crucial for getting the desired results from Roo Commander and its specialist modes. Just like managing a human team, providing clear, concise, and context-rich instructions leads to better outcomes, fewer errors, and faster task completion.

This guide provides practical tips and best practices for writing effective prompts when interacting with coordinating modes (`session-manager`, `roo-commander`) and for understanding how clear goals translate into effective delegation messages (`new_task`) sent to specialist modes.

**Goal:** To help users communicate their intent clearly to Roo Commander modes, leading to more accurate and efficient task execution.

## 2. Why Clear Communication Matters 💬➡️🤖

AI agents, especially LLMs, operate based on the information you provide. Ambiguous, incomplete, or contradictory instructions can lead to:

*   Incorrect results or code.
*   Wasted time and resources (API calls, compute).
*   Lengthy clarification dialogues.
*   Failure to follow project standards or constraints.
*   Delegation to the wrong specialist mode.

Clear communication minimizes these issues.

## 3. Tips for Effective Prompting & Delegation ✨

### 3.1. Be Specific About the Goal

*   **State the Objective Clearly:** What is the primary outcome you want to achieve? Start with a clear action verb.
    *   *Less Effective:* "Work on the login page."
    *   *More Effective:* "Implement the password reset functionality for the login page as described in FEAT-005."
    *   *More Effective:* "Refactor the `handleLogin` function in `src/auth.js` to improve error handling."
*   **Define Scope:** What are the boundaries of the task? What should be included, and just as importantly, what should *not* be included?
    *   *Less Effective:* "Add user profiles."
    *   *More Effective:* "Create the basic UI component for displaying user profile information (username, email). Data fetching will be handled in a separate task (TASK-102)."

### 3.2. Provide Sufficient Context

*   **Reference Artifacts:** Point to relevant IntelliManage artifacts (Tasks, Features, Epics, ADRs) by their ID. The AI can then read these files for details.
    *   *Example:* "Implement the API endpoint specified in TASK-123, ensuring it follows the guidelines in ADR-002."
*   **Mention Key Files/Locations:** Specify relevant file paths, function names, database tables, or API endpoints.
    *   *Example:* "Update the documentation in `docs/api/auth.md` to reflect the changes made to the `/login` endpoint."
*   **Explain the "Why":** Briefly explaining the reason behind a task can help the AI make better decisions.
    *   *Example:* "Refactor the database query in `getUserData()` to improve performance, as users are reporting slow load times (see BUG-055)."

### 3.3. Specify Constraints and Requirements

*   **Technical Constraints:** Mention required libraries, frameworks, language versions, or specific algorithms.
    *   *Example:* "Implement the component using React functional components and TypeScript. Use the `axios` library for API calls."
*   **Non-Functional Requirements:** Specify requirements like performance targets, security considerations, accessibility standards, or specific error handling logic.
    *   *Example:* "Ensure the image processing function completes within 500ms and handles potential file corruption errors gracefully."
*   **Output Format:** If you need output in a specific format (e.g., JSON, Markdown table, specific code structure), state it clearly.
    *   *Example:* "Provide the analysis results as a Markdown table with columns for Task ID, Complexity Score, and Recommendation."

### 3.4. Break Down Complex Requests

*   **One Goal Per Prompt:** Avoid asking for multiple unrelated things in a single prompt. It's better to have several focused interactions.
*   **Use IntelliManage Hierarchy:** For large goals, first create an Epic or Feature using `!pm create`, then ask the AI to help break it down into smaller Tasks or Stories.
*   **Iterative Approach:** Ask the AI to perform one step, review the result, then provide the instruction for the next step.

### 3.5. Define "Done" (Acceptance Criteria)

*   **Reference ACs:** If the task has Acceptance Criteria defined in its MDTM file, refer to them.
*   **State Expectations:** Clearly define what success looks like for the task. How will you know it's completed correctly?
    *   *Example:* "The task is done when the login form successfully authenticates valid users and displays an appropriate error message for invalid credentials."

### 3.6. Review and Clarify

*   **Read AI Responses:** Pay attention to how the AI coordinator (e.g., `session-manager`) rephrases your request or outlines its plan. Correct any misunderstandings early.
*   **Answer Clarifying Questions:** If the AI asks for more information, provide specific details.

## 4. How This Translates to Delegation (`new_task`) 📨

When a coordinator like `session-manager` or `roo-commander` delegates a task you provided, it uses your clear instructions and context to formulate the `new_task` message for the specialist.

*   A specific goal translates into clear instructions for the specialist.
*   Referenced artifacts allow the specialist to read the necessary details.
*   Mentioned constraints ensure the specialist uses the right tools and follows requirements.

Providing clear input to the coordinator directly results in better instructions for the specialist executing the work.

## 5. Conclusion ✅

Effective communication is key to maximizing the benefits of Roo Commander's multi-agent system. By providing specific goals, sufficient context, clear constraints, and defining success criteria, you empower the AI agents to understand your intent accurately and execute tasks efficiently. Practice these prompting techniques to improve your interactions and achieve better results.