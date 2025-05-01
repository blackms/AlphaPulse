+++
# --- Basic Metadata ---
id = "KB-RC-BESTPRACTICE-TROUBLESHOOTING"
title = "Troubleshooting: Common Issues & Debugging Steps"
status = "draft"
doc_version = "1.0" # Version of this guide
content_version = 1.0
audience = ["users", "developers"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/10_guide_tutorial.README.md" # Using guide template schema
tags = ["roo-commander", "intellimanage", "troubleshooting", "debugging", "errors", "faq", "guide", "best-practices", "support"]
related_docs = [
    "../README.md", # Link to the KB README
    "01_Effective_Prompting_Delegation.md",
    "02_Managing_AI_Context.md",
    "../../06_Advanced_Usage_Customization/02_Custom_Instructions_Rules.md",
    "../../06_Advanced_Usage_Customization/03_User_Preferences.md",
    "../../08_Community_Contributing/02_Contribution_Guide.md" # How to report bugs
    ]
difficulty = "intermediate"
estimated_time = "~15-20 minutes"
prerequisites = ["Basic interaction with Roo Commander modes"]
learning_objectives = ["Identify common types of errors when using Roo Commander", "Learn basic troubleshooting steps for mode failures, context errors, file access issues, and configuration problems", "Know where to look for error messages and logs", "Understand how to report issues effectively"]
+++

# Troubleshooting: Common Issues & Debugging Steps

## 1. Introduction / Goal 🎯

While Roo Commander aims for smooth operation, you might occasionally encounter errors or unexpected behavior. This guide outlines common issues faced by users and provides basic troubleshooting steps to help you diagnose and potentially resolve them.

**Goal:** To equip users with the knowledge to identify common problems and perform initial debugging steps when Roo Commander or its modes don't behave as expected.

## 2. General Troubleshooting Approach 🤔

Before diving into specific issues, try these general steps:

1.  **Read the Error Message Carefully:** The AI or Roo Code often provides specific error messages in the chat or output panels. Look for keywords related to files, permissions, context limits, or specific tools.
2.  **Check Roo Code Output:** Look at the Roo Code Output channel in VS Code (View -> Output, then select "Roo Code" from the dropdown). It may contain more detailed logs or stack traces.
3.  **Simplify Your Request:** Try breaking down your last prompt or action into smaller, simpler steps. Does a simpler version work? This helps isolate the problem.
4.  **Reload the Window:** Sometimes, temporary glitches can be resolved by reloading the VS Code window (`Developer: Reload Window` from the Command Palette).
5.  **Check Recent Changes:** Did the issue start after you modified a rule, mode definition, or configuration file? Revert the change temporarily to see if it resolves the problem.

## 3. Common Issues & Solutions ❓➡️✅

### 3.1. Mode Failures / Errors

*   **Symptoms:** The active mode stops responding, outputs an error message (e.g., "An error occurred...", "Failed to process request"), produces nonsensical output, or fails to complete a delegated task.
*   **Possible Causes:**
    *   Ambiguous or contradictory instructions in your prompt.
    *   A bug within the mode's logic, rules, or system prompt.
    *   Failure of a tool used by the mode (see section 3.5).
    *   An error returned by the underlying LLM API.
    *   Context window issues (see section 3.2).
*   **Troubleshooting Steps:**
    1.  Review your last prompt for clarity (See `01_Effective_Prompting_Delegation.md`). Try rephrasing.
    2.  Check the Roo Code Output channel for specific error details.
    3.  Try the same task with a different, potentially more generalist mode (e.g., if `dev-react` fails, try `util-senior-dev`). Does it succeed? This might indicate an issue specific to the specialist mode.
    4.  If the error seems LLM-related (e.g., mentions API issues), wait a few moments and try again.
    5.  If the issue persists, consider reporting it (see Section 4).

### 3.2. Context Window Errors

*   **Symptoms:** API errors mentioning "context length exceeded", "token limit", or similar. The AI might forget previous instructions, repeat itself, or give incoherent responses during long conversations.
*   **Possible Causes:** The total amount of text (prompt, history, rules, tool results) sent to the LLM exceeds its limit.
*   **Troubleshooting Steps:**
    1.  **Start a New Chat Session:** This is often the quickest fix, clearing the history. Use Handover Summaries (`03_Getting_Started/04_Session_Management_Continuity.md`) to resume context.
    2.  **Reference Files:** Instead of pasting large code blocks or text, use file paths or artifact IDs and ask the AI to read them (`read_file`).
    3.  **Break Down Tasks:** Address complex goals in smaller, sequential steps.
    4.  **Reduce Verbosity:** Set `verbosity_level = "concise"` in User Preferences (`06_Advanced_Usage_Customization/03_User_Preferences.md`).
    5.  Consult the Context Management guide (`07_Best_Practices_Troubleshooting/02_Managing_AI_Context.md`) for more details.

### 3.3. File Access / Permission Errors

*   **Symptoms:** Errors like "Permission denied", "EACCES", "File not found" (when you know it exists), or failure to write/read specific files.
*   **Possible Causes:**
    *   Actual operating system file permissions preventing VS Code (and thus Roo Code) from accessing the file/directory.
    *   Incorrect file path provided in the prompt or used internally by the mode.
    *   The mode has `file_access` restrictions defined in its `.mode.md` file that prevent it from accessing the required path.
*   **Troubleshooting Steps:**
    1.  **Verify Path:** Double-check that the file path mentioned in the error or your prompt is correct.
    2.  **Check OS Permissions:** Ensure the user running VS Code has read/write permissions for the target file and its parent directories.
    3.  **Check Mode Restrictions:** Look at the active mode's `.mode.md` file. Does it have a `[file_access]` section? Do the `read_allow` or `write_allow` patterns permit access to the required path? (Consider using `prime-coordinator` to check/modify if needed, carefully).
    4.  **Ensure File Exists:** If the error is "File not found", manually verify the file exists at the expected location.

### 3.4. Configuration Errors

*   **Symptoms:** Roo Code fails to load modes, modes behave unexpectedly, errors mentioning TOML parsing, rules not being applied.
*   **Possible Causes:**
    *   Syntax errors (invalid TOML) in `.mode.md`, `.roo/rules/`, `.ruru/projects/project_config.toml`, or `.roomodes`.
    *   Incorrect file paths referenced in configurations.
    *   Failure to run `build_roomodes.js` after adding/removing modes.
*   **Troubleshooting Steps:**
    1.  **Validate TOML:** Carefully check any recently edited configuration files for valid TOML syntax. Use an online TOML validator if needed. Pay attention to quotes, brackets, commas, and data types.
    2.  **Check `.roomodes`:** Ensure the `.roomodes` file at the workspace root correctly lists the modes you expect to be loaded.
    3.  **Run Build Script:** If you added/removed modes, run `node build_roomodes.js` from the workspace root.
    4.  **Reload Window:** Always reload the VS Code window after significant configuration changes.
    5.  **Check Roo Code Output:** Look for specific parsing errors during startup.

### 3.5. Tool Failures (`execute_command`, `browser`, etc.)

*   **Symptoms:** Errors specifically mentioning a tool (e.g., "Failed to execute command", "Browser tool error").
*   **Possible Causes:**
    *   For `execute_command`: The command doesn't exist in the system's PATH, invalid arguments passed, command failed with an error code.
    *   For `browser`/`fetch`: Network connectivity issues, invalid URL, website blocking requests.
    *   Incorrect usage of the tool by the AI mode.
*   **Troubleshooting Steps:**
    1.  **Verify Command/URL:** If possible, try running the command manually in your terminal or accessing the URL in your browser. Does it work?
    2.  **Check Arguments:** Review the arguments passed to the tool (often visible in the AI's `<thinking>` block or the Roo Code Output). Are they correct?
    3.  **Check Network:** Ensure you have internet connectivity if using network-dependent tools.
    4.  **Consider Alternatives:** Could a different tool achieve the same goal?

### 3.6. IntelliManage (`!pm`) Command Errors

*   **Symptoms:** Errors after issuing a `!pm` command, such as "Artifact not found", "Invalid status transition", "Schema validation failed", "Missing required field".
*   **Possible Causes:** Incorrect artifact ID, trying an invalid status change (e.g., Backlog -> Done), missing required TOML fields (like `feature_id` for tasks), invalid data types in TOML.
*   **Troubleshooting Steps:**
    1.  **Verify ID & Project:** Double-check the artifact ID and ensure you're targeting the correct project (using `!pm set-active` or `--project`).
    2.  **Check Methodology:** Review the allowed status transitions for your project's methodology (`project_config.toml`).
    3.  **Check Schema:** Ensure you provided all required fields for the command (e.g., `--type` for `!pm create task`). Refer to `DOC-UI-CMD-SPEC-001` or `!pm help [command]`.
    4.  **Inspect Artifact File:** If updating/showing an existing artifact, check its TOML frontmatter for correctness.

## 4. Reporting Issues 🐞

If you encounter a persistent issue that you believe is a bug in Roo Commander or one of its modes/rules:

1.  Gather information: Note the mode used, the exact prompt given, the full error message, relevant logs from the Roo Code Output channel, and steps to reproduce the issue.
2.  Consult the Contribution Guide (`08_Community_Contributing/02_Contribution_Guide.md`) for instructions on where and how to report bugs (e.g., GitHub Issues).

## 5. Conclusion ✅

Troubleshooting AI interactions requires a methodical approach. By carefully reading error messages, checking logs, simplifying requests, verifying configurations, and understanding common pitfalls like context limits and file permissions, you can often diagnose and resolve issues encountered while using Roo Commander and IntelliManage.