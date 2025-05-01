+++
# --- Basic Metadata ---
id = "KB-RC-ADV-CUSTOM-MODES"
title = "Advanced Usage: Creating & Configuring Custom Modes"
status = "draft"
doc_version = "1.0" # Version of this guide
content_version = 1.0
audience = ["developers", "architects", "users", "contributors", "ai_modes"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md"
tags = ["roo-commander", "customization", "modes", "configuration", "advanced", "guide", "tutorial", "ai-agents"]
related_docs = [
    "../README.md", # Link to the KB README
    "../../04_Understanding_Modes/01_Mode_Roles_Hierarchy.md",
    "../../../../.ruru/docs/roo-code/custom-modes.md", # Link to the base Roo Code doc this enhances
    "../../../../.ruru/templates/modes/00_standard_mode.md", # Link to the mode template
    "../../../../.roo/rules/01-standard-toml-md-format.md", # Link to TOML+MD standard
    "../../02_Core_Concepts/04_Rules_Explained.md",
    "../../02_Core_Concepts/05_Knowledge_Bases_Explained.md"
    ]
difficulty = "intermediate"
estimated_time = "~20-30 minutes"
prerequisites = ["Roo Commander installed", "Understanding of Roo Commander Modes & Hierarchy", "Basic understanding of TOML and Markdown"]
learning_objectives = ["Understand the benefits of creating custom modes", "Learn how to define a new mode using the standard template", "Understand the key configuration fields in a .mode.md file", "Learn how to add mode-specific rules and knowledge bases", "Differentiate between global and project-specific modes"]
+++

# Advanced Usage: Creating & Configuring Custom Modes

## 1. Introduction / Goal 🎯

Roo Commander's power lies in its multi-agent system, where specialized modes handle specific tasks. While the framework comes with a suite of built-in modes, you can create **custom modes** to tailor Roo's behavior precisely to your project's needs, specific workflows, or unique technological requirements.

This guide provides a detailed walkthrough on how to define, configure, and manage custom modes within the Roo Commander ecosystem, enhancing the information provided in the base Roo Code documentation (`.ruru/docs/roo-code/custom-modes.md`).

**Goal:** To empower users to create new AI agents (modes) with specific roles, capabilities, instructions, and knowledge bases, extending the core Roo Commander framework.

## 2. Why Create Custom Modes? ✨

*   **Specialization:** Create modes hyper-focused on a niche task, tool, or internal API (e.g., "Internal Billing API Specialist", "Legacy System Refactorer").
*   **Workflow Automation:** Define modes that execute specific, multi-step internal workflows.
*   **Safety & Constraints:** Build modes with restricted tool access or file permissions for specific tasks (e.g., a "Read-Only Auditor" mode).
*   **Team Standards:** Encode team-specific coding standards, documentation formats, or review processes directly into a mode's instructions and KB.
*   **Experimentation:** Test new prompts, rules, or AI configurations in an isolated mode without affecting standard workflows.

## 3. The Anatomy of a Mode: `.mode.md` File 🧬

Every mode, whether built-in or custom, is defined by a single file following the **TOML+MD standard**.

*   **Location:** `.ruru/modes/[mode_slug]/[mode_slug].mode.md`
    *   *Example:* `.ruru/modes/dev-react/dev-react.mode.md`
*   **Template:** Always start with the standard template: `.ruru/templates/modes/00_standard_mode.md`.
*   **Structure:**
    *   **TOML Frontmatter (`+++`):** Defines the mode's core configuration, metadata, and pointers.
    *   **Markdown Body (`# ...`):** Provides human-readable documentation *about* the mode (Description, Capabilities, Workflow, Limitations, Rationale). This section is primarily for human understanding and **is not typically loaded directly into the AI's context** (unlike Rules or KBs).

## 4. Key TOML Configuration Fields ⚙️

*(Refer to the standard mode template for the full list)*

*   **Core Identification (Required):**
    *   `id`: The unique slug used internally and in commands (e.g., `dev-react`, `util-my-custom-tool`). Must match the directory and filename slug.
    *   `name`: Human-readable name with emoji (e.g., `⚛️ React Specialist`).
    *   `version`: Semantic version for the mode definition (e.g., `"1.0.0"`).
*   **Classification & Hierarchy (Required):**
    *   `classification`: The mode's role (e.g., `worker`, `lead`, `agent`, `utility`). See `04_Understanding_Modes/01_Mode_Roles_Hierarchy.md`.
    *   `domain`: Primary area of operation (e.g., `frontend`, `backend`, `devops`, `utility`).
    *   `sub_domain` (Optional): More specific area (e.g., `react`, `terraform`, `testing`).
*   **Description (Required):**
    *   `summary`: A concise, one-sentence description of the mode's core purpose. Used in mode lists.
*   **Base Prompting (Required):**
    *   `system_prompt`: The core instruction defining the mode's persona, primary responsibilities, and high-level operational guidelines. This is the foundation of the AI's behavior for this mode. **Crucially, it should include the standard operational guidelines, especially the KB lookup instruction.**
*   **Tool Access (Optional):**
    *   `allowed_tool_groups`: An array specifying which tool categories the mode can use (`"read"`, `"edit"`, `"browser"`, `"command"`, `"mcp"`). If omitted, defaults to all standard groups.
*   **File Access Restrictions (Optional):**
    *   `[file_access]`: Contains `read_allow` and `write_allow` arrays with glob patterns to restrict file system access. If omitted, defaults to allowing all access.
*   **Metadata (Optional but Recommended):**
    *   `tags`: Keywords for selection and discovery.
    *   `categories`: Broader functional areas.
    *   `delegate_to`, `escalate_to`, `reports_to`: Define typical collaboration pathways.
    *   `documentation_urls`: Links to relevant external docs.
*   **Custom Instructions Pointer (Recommended):**
    *   `custom_instructions_dir = "kb"`: Points to the mode-specific Knowledge Base directory. **This is essential for loading detailed context.**

## 5. Adding Mode-Specific Rules & Knowledge 🧠

While the `.mode.md` defines the *what*, the *how* is often detailed in rules and the KB:

*   **Mode-Specific Rules:**
    *   **Location:** `.roo/rules-[mode_slug]/` (e.g., `.roo/rules-dev-react/`)
    *   **Purpose:** Define specific operational procedures, constraints, or KB lookup triggers *only* for this mode.
    *   **Requirement:** At minimum, should usually contain a KB lookup rule (e.g., `01-kb-lookup-rule.md`) instructing the mode to consult its KB.
*   **Mode-Specific Knowledge Base (KB):**
    *   **Location:** `.ruru/modes/[mode_slug]/kb/` (e.g., `.ruru/modes/dev-react/kb/`)
    *   **Purpose:** Store detailed procedures, reference material, code snippets, best practices, examples, or synthesized context relevant to the mode's function.
    *   **Structure:** Contains `.md` files, potentially organized into subdirectories. Should include a `README.md` index file.

## 6. Creating a New Custom Mode: Step-by-Step 🛠️

1.  **Plan:** Define the mode's purpose, classification, domain, key capabilities, and a unique `slug`.
2.  **Create Directories:**
    *   Create the mode directory: `.ruru/modes/[your-slug]/`
    *   Create the KB directory: `.ruru/modes/[your-slug]/kb/`
    *   Create the rules directory: `.roo/rules-[your-slug]/`
3.  **Create Mode Definition File:**
    *   Copy the standard template: `cp .ruru/templates/modes/00_standard_mode.md .ruru/modes/[your-slug]/[your-slug].mode.md`
    *   Edit the new `.mode.md` file:
        *   Fill in all **Required** TOML fields (`id`, `name`, `version`, `classification`, `domain`, `summary`, `system_prompt`). Ensure `id` matches the slug.
        *   Customize the `system_prompt` to accurately reflect the mode's role and responsibilities. **Include the standard operational guidelines, especially the KB lookup pointer.**
        *   Configure `allowed_tool_groups` and `file_access` if restrictions are needed.
        *   Add relevant `metadata` (tags, categories, collaboration links).
        *   Ensure `custom_instructions_dir = "kb"` is present.
        *   Fill in the Markdown body documentation sections (Description, Capabilities, etc.).
4.  **Create KB Lookup Rule:**
    *   Copy the rule template: `cp .ruru/templates/toml-md/16_ai_rule.md .roo/rules-[your-slug]/01-kb-lookup-rule.md` (or use `99-` prefix if preferred).
    *   Edit the new rule file:
        *   Fill in the TOML metadata (`id`, `title`, `scope`, `target_audience=[your-slug]`, etc.).
        *   Add the rule text instructing the mode to consult its KB at `.ruru/modes/[your-slug]/kb/`.
5.  **Populate Initial KB (Optional but Recommended):**
    *   Create a `README.md` inside `.ruru/modes/[your-slug]/kb/` outlining the planned KB structure.
    *   Add any initial context files (e.g., `01-core-principles.md`).
6.  **Update Mode Registry:**
    *   Run the build script from the workspace root: `node build_roomodes.js`
    *   This scans the `.ruru/modes/` directory and updates the `.roomodes` file, making your new mode available to Roo Code.
7.  **Reload VS Code:** Reload the window (`Developer: Reload Window`) for Roo Code to recognize the new mode.
8.  **Test & Refine:** Interact with your new mode, test its behavior, and refine its system prompt, rules, and KB content as needed.

*(Alternatively, use the interactive "Create New Custom Mode" workflow initiated via Roo Commander - Option 17 in the initial prompt)*

## 7. Global vs. Project-Specific Modes 🌍 vs. 🏠

*   **Global Modes:** Defined in Roo Code's global configuration (less common for user customization). Available in all workspaces.
*   **Project-Specific Modes:** Defined within the workspace's `.ruru/modes/` directory and registered in the root `.roomodes` file. Only available within that specific workspace. This is the standard way to add custom modes for a project.
*   **Precedence:** If a project-specific mode has the same `slug` as a global or built-in mode, the project-specific version typically overrides the global one within that workspace.

## 8. Conclusion ✅

Custom modes are a powerful feature for tailoring Roo Commander to specific needs. By following the standard `.mode.md` structure, leveraging mode-specific rules and knowledge bases, and using the provided templates, you can create specialized AI agents that enhance your development workflow and enforce project standards effectively.