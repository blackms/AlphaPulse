+++
# --- Basic Metadata ---
id = "KB-RC-ADV-USER-PREFS"
title = "Advanced Usage: Configuring User Preferences"
status = "draft"
doc_version = "1.0" # Version of this guide
content_version = 1.0
audience = ["users", "developers"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md" # Using general doc template structure
tags = ["roo-commander", "customization", "preferences", "configuration", "user-profile", "advanced", "guide", "settings"]
related_docs = [
    "../README.md", # Link to the KB README
    "../../../../.roo/rules/00-user-preferences.md", # Link to the actual preferences file
    "../../../../.roo/rules/00-user-preferences.README.md", # Link to the schema for the preferences file (if it exists)
    "04_Prime_Coordinator_Usage.md" # How to edit config files
    ]
difficulty = "intermediate"
estimated_time = "~10 minutes"
prerequisites = ["Roo Commander installed", "Basic understanding of TOML"]
learning_objectives = ["Locate the user preferences file", "Understand the purpose of each configuration field", "Know how different settings affect Roo Commander's behavior", "Understand how to safely modify preferences"]
+++

# Advanced Usage: Configuring User Preferences

## 1. Introduction / Goal 🎯

Roo Commander allows you to personalize certain aspects of its behavior and provide context about yourself through a dedicated **User Preferences** file. This file helps tailor interactions and can inform how AI agents approach tasks or communicate results.

This guide explains the purpose and structure of the user preferences file (`.roo/rules/00-user-preferences.md`) and details the effect of each available setting.

**Goal:** To enable users to understand and configure their personal preferences to optimize their experience with Roo Commander.

## 2. Location and Format 📂

*   **Location:** The user preferences file is located at the workspace root: `.roo/rules/00-user-preferences.md`.
*   **Format:** It uses the standard **TOML+MD** format. All configuration settings are defined within the TOML frontmatter block (`+++`). The Markdown body below the TOML block is typically minimal and just confirms the purpose of the file.

```markdown
+++
# --- Basic Metadata ---
id = "user-preferences"
title = "User Preferences"
# ... other standard rule metadata ...

# --- User Information ---
user_name = "Your Name" # << Example >>
skills = ["Python", "React", "AWS"] # << Example >>

# --- Roo Usage Preferences ---
[roo_usage_preferences]
preferred_modes = ["dev-python", "util-writer"] # << Example >>
verbosity_level = "normal" # Options: "concise", "normal", "verbose"
auto_execute_commands = false # Options: true, false
preferred_language = "en" # ISO 639-1 code
preferred_input_language = "en" # ISO 639-1 code
preferred_output_language = "en" # ISO 639-1 code
+++

# User Preferences Data (Defined in TOML)
# Settings here influence Roo Commander behavior.
```

## 3. Configuration Fields Explained ⚙️

### 3.1. User Information

*   `user_name` (String, Required):
    *   **Purpose:** Your name, used for personalization in communication or potentially assigning tasks if using user-specific identifiers.
    *   **Effect:** AI modes may refer to you by this name.
    *   **Example:** `user_name = "Alice Wonderland"`
*   `skills` (Array of Strings, Optional):
    *   **Purpose:** List your primary technical skills, frameworks, or areas of expertise.
    *   **Effect:** Helps AI agents understand your technical background, potentially tailoring explanations or suggesting relevant tasks. Can inform mode selection if certain skills imply familiarity with specific tools.
    *   **Example:** `skills = ["React", "TypeScript", "Node.js", "AWS", "API Design"]`

### 3.2. Roo Usage Preferences (`[roo_usage_preferences]` table)

*   `preferred_modes` (Array of Strings, Optional):
    *   **Purpose:** List the slugs of modes you frequently use or prefer for certain types of tasks.
    *   **Effect:** AI coordinators *may* consider this list when suggesting modes, but task requirements and specialist capabilities generally take precedence. Primarily informational.
    *   **Example:** `preferred_modes = ["dev-react", "design-tailwind", "util-writer"]`
*   `verbosity_level` (String, Optional, Default: `"normal"`):
    *   **Purpose:** Control the level of detail in conversational output from coordinating modes like `roo-commander` or `session-manager`.
    *   **Options:**
        *   `"concise"`: Minimal explanations, focus on actions and results.
        *   `"normal"`: Balanced explanations and status updates (Default).
        *   `"verbose"`: More detailed explanations of reasoning, steps taken, and potential options.
    *   **Effect:** Influences how much explanatory text coordinating modes generate. Does not typically affect specialist modes' core output (like code or documentation).
*   `auto_execute_commands` (Boolean, Optional, Default: `false`):
    *   **Purpose:** Control whether commands suggested by AI modes (within `<execute_command>`) require explicit user confirmation before running.
    *   **Effect:**
        *   `false` (Default & Recommended): Roo Code will prompt you ("Run this command?") before executing any command suggested by an AI mode.
        *   `true`: Roo Code will attempt to execute commands suggested by AI modes automatically **without** prompting. **Use with extreme caution!** This bypasses a critical safety check.
*   `preferred_language` (String, Optional, Default: `"en"`):
    *   **Purpose:** A general preference for language, primarily used as a fallback if input/output preferences aren't set. Uses ISO 639-1 codes.
    *   **Effect:** Minor influence, mainly a hint for default behavior.
*   `preferred_input_language` (String, Optional, Default: `"en"`):
    *   **Purpose:** Informs the AI about the language you are most likely to use when writing prompts. Uses ISO 639-1 codes.
    *   **Effect:** Can help the AI better interpret your requests, especially for nuanced language.
*   `preferred_output_language` (String, Optional, Default: `"en"`):
    *   **Purpose:** Specifies the preferred language for the AI's conversational responses, code comments, and generated documentation. Uses ISO 639-1 codes.
    *   **Effect:** AI modes will attempt to generate relevant output in this language where feasible. Code itself will remain in the appropriate programming language.

## 4. How to Modify Preferences ✏️

Modifying `.roo/rules/00-user-preferences.md` involves changing configuration files and should be done carefully.

1.  **Recommended Method: Via `roo-commander` Initial Options:**
    *   Activate `roo-commander`.
    *   Select option **`13: ⚙️ Update my preferences / profile`**.
    *   Follow the prompts to specify the changes you want to make.
    *   Commander will delegate the update to `prime-coordinator`, which uses `prime-txt` (requiring your confirmation before saving).
2.  **Advanced Method: Via `prime-coordinator`:**
    *   Activate `prime-coordinator`.
    *   Provide an explicit instruction:
        ```prompt
        @prime-coordinator Please update the file `.roo/rules/00-user-preferences.md`. Change the `verbosity_level` to "verbose" and add "Docker" to the `skills` array.
        ```
    *   `prime-coordinator` will delegate to `prime-txt`, which will prompt you for confirmation before saving the changes.
3.  **Direct Edit (Use Caution):**
    *   You can directly open and edit the `.roo/rules/00-user-preferences.md` file in VS Code.
    *   **Ensure you maintain valid TOML syntax** within the `+++` block. Mistakes can prevent Roo Commander from loading rules correctly.
    *   Save the file. You may need to reload the VS Code window (`Developer: Reload Window`) for changes to take full effect immediately.

## 5. Conclusion ✅

The `00-user-preferences.md` file provides a simple way to personalize your Roo Commander experience and provide helpful context to the AI agents. By configuring your name, skills, and usage preferences, you can tailor interactions to better suit your workflow and background. Remember to modify this file carefully, preferably using the guided options.