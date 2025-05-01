+++
# --- Basic Metadata ---
id = "KB-RC-SETUP-SESSION-MGMT"
title = "Getting Started: Session Management & Continuity"
status = "draft"
difficulty = "beginner"
estimated_time = "~10 minutes"
target_audience = ["users"]
prerequisites = ["Roo Commander installed", "Basic understanding of Roo Commander workflow"]
learning_objectives = ["Understand the role of the Session Manager mode", "Learn how Handover Summaries maintain context", "Know how to resume work between sessions"]
template_schema_doc = ".ruru/templates/toml-md/10_guide_tutorial.README.md"
tags = ["roo-commander", "getting-started", "session-management", "context", "continuity", "handover", "workflow", "guide", "tutorial", "session-manager"]
related_docs = [
    "../README.md", # Link to the KB README
    "01_Installation_Setup.md",
    "02_Initial_Interaction.md",
    "03_Basic_Workflow_Example.md",
    "../../01_Introduction/03_Architecture_Overview.md", # Mentions Session Manager concept
    "../../../modes/session-manager/session-manager.mode.md", # Link to mode definition (assuming it exists)
    "../../../modes/agent-session-summarizer/agent-session-summarizer.mode.md", # Link to summarizer mode
    "../../../templates/handover_summary_template.md" # Link to the template used
    ]
+++

# Getting Started: Session Management & Continuity

## 1. Introduction / Goal 🎯

One challenge with complex AI interactions is maintaining context over time. If you close your VS Code window, restart the extension, or simply come back to a project after a break, how does the AI remember what you were working on?

Roo Commander addresses this through a dedicated **`session-manager`** mode and a **Handover Summary** mechanism. This guide explains how this system helps you maintain continuity between work sessions.

The goal is to allow you to seamlessly pause and resume your work with Roo Commander and IntelliManage, minimizing the need to repeat context or instructions.

## 2. The `session-manager` Mode 🧑‍💼

Instead of directly interacting with `roo-commander` for most day-to-day tasks, you will often interact primarily with the **`session-manager`** mode. (This might become the default mode you see when starting a chat in a configured workspace).

*   **Purpose:** The `session-manager` acts as your primary point of contact for a specific work session. Its job is to:
    *   Understand your high-level goal for the session.
    *   Keep track of the overall progress *within that session*.
    *   Load context from previous sessions (via Handover Summaries).
    *   Delegate specific execution tasks to other modes (like `roo-dispatch` or the IntelliManage Core Logic Engine).
    *   Facilitate the creation of Handover Summaries when you finish a work period.

## 3. The Handover Summary 📝➡️

Think of a Handover Summary as a concise "state snapshot" of your work at a particular point in time.

*   **Generation:** When you finish a work session, you can ask the `session-manager` to create a handover summary (e.g., "Okay, let's wrap up for now, please generate a handover summary"). The `session-manager` will likely delegate this to the `agent-session-summarizer`.
*   **Content:** The summary typically includes (based on the template `.ruru/templates/handover_summary_template.md`):
    *   The main goal you were working on.
    *   The last few significant actions completed.
    *   Any tasks currently in progress or waiting for delegation.
    *   The next planned step(s).
    *   Any known blockers or open questions.
    *   A timestamp.
*   **Storage:** Summaries are saved as timestamped Markdown files in a dedicated directory, usually `.ruru/context/handovers/`.

## 4. Resuming Your Work ⏯️

When you start a new chat session with the `session-manager`:

1.  **Automatic Check:** The `session-manager` automatically looks for the most recent Handover Summary file in `.ruru/context/handovers/`.
2.  **Context Loading:** If a summary is found, the `session-manager` reads it (`read_file`) to understand the previous state.
3.  **Confirmation Prompt:** It will typically confirm the previous goal with you:
    ```prompt
    Welcome back! It looks like last time we were working on [Goal from summary]. Shall we continue with that, or start something new?
    ```
4.  **Seamless Continuation:** If you confirm, the `session-manager` has the necessary context (last actions, pending tasks, next steps) to resume the workflow efficiently, delegating the next appropriate task to `roo-dispatch` or other modes.

## 5. How it Differs from `roo-commander` / `roo-dispatch` 👑 vs. 🚜

*   **`session-manager`:** Focuses on the **user interaction flow** and **session state** over time. Manages the *overall* goal for the work period.
*   **`roo-dispatch`:** Focuses on the **execution coordination** of *specific tasks* delegated by the `session-manager`. It's largely stateless regarding the overall session.
*   **`roo-commander`:** Reserved for **complex initial planning, onboarding, or recovery** scenarios where its comprehensive ruleset is needed.

This layered approach allows for efficient day-to-day interaction via `session-manager` and `roo-dispatch`, while preserving the power of `roo-commander` for more demanding situations.

## 6. Conclusion ✅

The `session-manager` and Handover Summary system provide a lightweight yet effective mechanism for maintaining context and continuity in your AI-assisted development workflow. By capturing the essential state at the end of a session and reloading it at the beginning of the next, IntelliManage and Roo Commander aim to make pausing and resuming complex tasks much smoother. Remember to ask the `session-manager` to generate a summary when you're stepping away!