+++
# --- Basic Metadata ---
id = "KB-RC-BESTPRACTICE-CONTEXT"
title = "Best Practices: Managing AI Context Limits"
status = "draft"
doc_version = "1.0" # Version of this guide
content_version = 1.0
audience = ["users", "developers", "architects"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/10_guide_tutorial.README.md" # Using guide template schema
tags = ["roo-commander", "intellimanage", "best-practices", "context-window", "llm", "performance", "troubleshooting", "knowledge-base", "summarization", "modes"]
related_docs = [
    "../README.md", # Link to the KB README
    "../../02_Core_Concepts/05_Knowledge_Bases_Explained.md",
    "../../03_Getting_Started/04_Session_Management_Continuity.md",
    "../../04_Understanding_Modes/01_Mode_Roles_Hierarchy.md",
    "../../06_Advanced_Usage_Customization/03_User_Preferences.md" # Verbosity settings
    ]
difficulty = "intermediate"
estimated_time = "~10-15 minutes"
prerequisites = ["Basic interaction with Roo Commander modes", "Understanding of LLM concepts helpful"]
learning_objectives = ["Understand what constitutes the AI's context window", "Recognize the limitations of context windows", "Learn the strategies Roo Commander uses to manage context", "Know user best practices for minimizing context issues"]
+++

# Best Practices: Managing AI Context Limits

## 1. Introduction / Goal 🎯

Large Language Models (LLMs), like the ones powering Roo Commander modes, have a finite "memory" known as the **context window**. This window represents the maximum amount of information (text, code, instructions, history) the AI can consider at any one time when generating a response or performing an action.

Understanding context limits and how Roo Commander manages them is crucial for ensuring reliable performance, preventing errors, and getting accurate results, especially during long or complex interactions.

**Goal:** To explain the concept of the LLM context window, how it applies to Roo Commander, the system's built-in management strategies, and user best practices to work effectively within these limits.

## 2. What Makes Up the Context Window? 🤔

When you interact with a Roo Commander mode, the information sent to the underlying LLM (e.g., Gemini via Vertex AI) typically includes:

1.  **Mode's System Prompt:** The core instructions from the `.mode.md` file.
2.  **Loaded Rules:** Content from applicable workspace-wide (`.roo/rules/`) and mode-specific (`.roo/rules-[mode_slug]/`) rule files.
3.  **Chat History:** Recent messages exchanged between you and the AI mode(s) in the current chat session.
4.  **Tool Calls:** The XML-like representation of actions taken by the AI (e.g., `<new_task>`, `<read_file>`, `<execute_command>`).
5.  **Tool Results:** The output returned from those tool calls (e.g., `<result>File content here...</result>`).
6.  **File Content (Sometimes):** Content explicitly read using `read_file` or provided in prompts.

This combined information forms the context the LLM uses to understand the current situation and generate its next response or action.

## 3. The Context Limit Problem ⚠️

Every LLM has a maximum context window size, measured in tokens (roughly equivalent to words or parts of words).

*   **Exceeding the Limit:** Sending more tokens than the model supports results in an API error, and the AI cannot process the request.
*   **Approaching the Limit:** Even before hitting the hard limit, performance can degrade:
    *   **Forgetting:** The AI may "forget" information or instructions from earlier in the conversation/context.
    *   **Reduced Quality:** Responses may become less coherent, relevant, or accurate.
    *   **Slower Responses:** Processing larger contexts takes more time.

In long, complex development tasks involving multiple files, extensive chat history, and numerous tool calls, managing the context window becomes critical.

## 4. Roo Commander's Context Management Strategies 🛠️

The Roo Commander framework employs several strategies to mitigate context window limitations:

1.  **Specialized Modes:** Using modes with focused roles (`roo-dispatch`, `dev-react`, `util-writer`) means their system prompts and loaded rules are often much smaller than the comprehensive `roo-commander`, reducing the base context size for specific tasks.
2.  **Knowledge Bases (KBs):** Detailed procedures, reference materials, and examples are stored in mode-specific KBs (`.ruru/modes/[mode_slug]/kb/`). This information is **not** loaded into the main context automatically. Modes access specific KB files *on demand* using `read_file` when instructed by their rules, retrieving only the necessary information for the current step. (See `02_Core_Concepts/05_Knowledge_Bases_Explained.md`).
3.  **Session Summarization (Handover Summaries):** The `session-manager` mode uses Handover Summaries (`.ruru/context/handovers/`) to maintain continuity between sessions *without* needing to reload the entire chat history from previous days. It loads a concise summary of the last known state. (See `03_Getting_Started/04_Session_Management_Continuity.md`).
4.  **Structured Data (TOML+MD):** Using TOML frontmatter for key metadata allows agents to quickly extract essential information (like status, ID, links) from artifact files without needing the LLM to parse large amounts of Markdown text every time.
5.  **Iterative Execution:** Breaking down large goals into smaller, delegated tasks (often tracked via MDTM) means each individual interaction with a specialist mode typically involves a smaller, more focused context related only to that specific sub-task.

## 5. User Best Practices for Managing Context 👍

You can significantly help manage context by following these practices:

*   **Be Concise:** Write clear but brief prompts. Avoid unnecessary conversational filler.
*   **Break Down Requests:** Submit complex goals as a series of smaller, distinct requests rather than one massive prompt.
*   **Start New Sessions:** For completely unrelated tasks, consider starting a new chat session. This clears the history and allows the `session-manager` to start fresh (or load the relevant handover summary if applicable).
*   **Use Artifact IDs:** Instead of pasting large code blocks or document sections into the chat, reference the relevant file path or IntelliManage artifact ID (e.g., "Refactor the function in `src/utils.js` based on the requirements in TASK-125"). Let the AI use `read_file`.
*   **Manage IntelliManage Artifacts:** Keep artifact descriptions focused. Archive or close completed tasks (`!pm archive TASK-ID`) so they don't clutter lists requested by `!pm list`.
*   **Adjust Verbosity:** If you consistently run into issues, consider setting `verbosity_level = "concise"` in your User Preferences (`06_Advanced_Usage_Customization/03_User_Preferences.md`) to reduce conversational output from coordinators.

## 6. Troubleshooting Context Issues ❓

*   **Signs of Problems:**
    *   API errors mentioning token limits or context length.
    *   The AI repeatedly asking for information you've already provided.
    *   The AI seemingly ignoring previous instructions or constraints.
    *   Nonsensical or irrelevant responses.
*   **What to Do:**
    *   Try simplifying your last request.
    *   Start a new chat session to clear the history.
    *   Break the task down into smaller steps.
    *   Ensure you are referencing files/artifacts instead of pasting large amounts of text.
    *   Check your `verbosity_level` setting.

## 7. Conclusion ✅

While LLMs have context limits, the Roo Commander framework is designed with strategies like specialized modes, KBs, summarization, and iterative execution to manage these constraints effectively. By understanding these mechanisms and applying user best practices for clear and focused communication, you can minimize context-related issues and maintain a productive workflow even during complex development tasks.