+++
# --- Basic Metadata ---
id = "KB-RC-MODES-SELECTION-GUIDE"
title = "Understanding Modes: Mode Selection Guide (Enhanced)"
status = "draft"
doc_version = "1.0" # Version of this guide
content_version = 1.0
audience = ["users", "developers", "architects", "contributors", "ai_modes"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md"
tags = ["roo-commander", "intellimanage", "core-concept", "modes", "delegation", "selection", "hierarchy", "guide", "best-practices"]
related_docs = [
    "../README.md", # Link to the KB README
    "01_Mode_Roles_Hierarchy.md", # Link to the hierarchy explanation
    "03_Mode_Directory_Reference.md", # Link to the mode list
    "../../../../.ruru/docs/standards/mode_selection_guide.md", # Link to the original standard this enhances
    "../../../../.roo/rules-roo-commander/03-delegation-simplified.md", # Commander's delegation rule
    "../../../../.ruru/modes/roo-commander/kb/kb-available-modes-summary.md" # Generated mode summary
    ]
+++

# Understanding Modes: Mode Selection Guide (Enhanced)

## 1. Purpose 🎯

This guide provides structured information and principles to assist all users and AI modes (especially coordinators like `session-manager`, `roo-dispatch`, and `roo-commander`) in selecting the most appropriate specialist mode for a given task. Effective delegation is key to efficient project execution within the Roo Commander framework.

This guide enhances the original standard (`.ruru/docs/standards/mode_selection_guide.md`) by incorporating hierarchy concepts and explaining both automatic selection by coordinators and manual selection by users.

## 2. General Selection Principles 💡

When deciding which mode should handle a task, consider these principles:

1.  **Specificity First:** Always prefer a specialist mode whose core purpose and `tags` directly match the task's domain and technology over a generalist mode (e.g., use `framework-react` for React component implementation over `lead-frontend` or `util-senior-dev`).
2.  **Match Keywords & Context:** Use the task description's keywords, required technologies, and the project's Stack Profile (`.ruru/context/stack_profile.json`, if available) to find modes with matching `tags` or capabilities described in their summary/documentation.
3.  **Consider Hierarchy & Task Level:** Align the task's nature with the mode classifications (see `01_Mode_Roles_Hierarchy.md`).
    *   **Strategic/Planning:** Directors (`manager-*`, `core-architect`).
    *   **Domain Oversight/Coordination:** Leads (`lead-*`).
    *   **Implementation/Execution:** Workers/Specialists (`dev-*`, `framework-*`, `spec-*`, `util-*`).
    *   **Support/Automation:** Agents (`agent-*`).
    Delegate "down" the hierarchy where appropriate.
4.  **Review Capabilities:** If unsure between similar modes, consult their specific capabilities listed in the Mode Directory Reference (`03_Mode_Directory_Reference.md`) or their individual `.mode.md` documentation.
5.  **Use MDTM Appropriately:** For complex, stateful, or high-risk tasks requiring detailed tracking, initiate the MDTM workflow (see `.roo/rules/04-mdtm-workflow-initiation.md`) and specify the `assigned_to` mode in the task file's TOML. For simpler tasks, direct `new_task` delegation is sufficient.
6.  **Provide Clear Context:** The clearer your task description and context, the better the automatic selection (by coordinators) or your manual selection will be.
7.  **When in Doubt, Ask/Default:** If unsure which specialist to delegate to, ask a higher-level coordinator (`roo-commander`) or delegate to the relevant Lead mode to handle the breakdown and further delegation.

## 3. How Coordinators Select Modes (Automatic Selection) 🤖⚙️

Coordinating modes like `roo-commander` and `roo-dispatch` use a systematic process when automatically selecting a specialist for a delegated task:

1.  **Analyze Task Input:** Parse the task description, identifying keywords, required actions (e.g., "implement", "test", "refactor", "document", "configure"), and specific technologies mentioned (e.g., "React", "PostgreSQL", "Terraform", "FastAPI").
2.  **Consult Stack Profile:** Check the project's Stack Profile (`.ruru/context/stack_profile.json`) for preferred technologies and team familiarity, which can influence mode selection.
3.  **Filter Available Modes:** Review the list of available modes (from `.roomodes` or internal knowledge).
4.  **Match Metadata:** Compare task requirements against mode metadata:
    *   **`tags`:** Look for direct matches between task keywords/technologies and mode tags.
    *   **`summary` / `roleDefinition`:** Analyze for alignment between the task goal and the mode's core purpose.
    *   **`domain` / `sub_domain`:** Filter modes based on the task's domain (e.g., "backend", "frontend", "devops").
    *   **`classification`:** Ensure the mode's role (e.g., "worker", "agent") is appropriate for the task type (implementation vs. support).
5.  **Prioritize Specificity:** Give preference to modes with highly specific matching tags or sub-domains over more generalist modes.
6.  **Consider Hierarchy:** Generally avoid delegating implementation tasks directly to Leads if appropriate Workers/Specialists exist.
7.  **Select & Log:** Choose the best-matching mode. Log the chosen mode and the rationale for the selection in the coordination log (e.g., the Commander's MDTM task file).
8.  **Fallback:** If no suitable specialist is found, select the most relevant Lead or a capable generalist (like `util-senior-dev`) and provide clear instructions, or escalate back to the user/requestor for guidance.

## 4. User Influence & Direct Selection 🧑‍💻👉

While coordinators aim for optimal automatic delegation, users have several ways to influence or directly control mode selection:

1.  **Influencing Automatic Selection:**
    *   **Clear Prompts:** Provide specific, unambiguous task descriptions to the coordinator (`session-manager`, `roo-commander`).
    *   **Use Keywords:** Include relevant technology names, action verbs, and domain keywords in your request.
    *   **Reference Artifacts:** Point to specific requirements documents, designs, or existing code files that provide context.
2.  **Direct Mode Interaction (Bypassing Coordinators):**
    *   **Mode Switcher:** Use the mode selection dropdown in the Roo Code UI to activate a specific mode directly.
    *   **Chat Mention:** Address a mode directly in chat using `@<mode-slug>` (e.g., `@dev-react Please implement this component...`). *Note: The mode still operates under the coordination framework if initiated by Commander/Dispatch.*
3.  **Specifying Assignee in MDTM:**
    *   When creating or updating an MDTM task file (`.ruru/projects/[proj]/tasks/TASK-ID.md`), you can manually set the `assigned_to` field in the TOML frontmatter to the desired mode slug. The `manager-project` or `roo-commander` will then delegate to that specified mode when the task becomes active.

**Caution:** When you directly select a mode or specify an assignee, you take on more responsibility for ensuring that mode is the *correct* choice for the task's complexity and requirements. Coordinators provide a layer of analysis that is bypassed with direct selection.

## 5. Mode Details 📖

*(Note: The detailed information for each specific mode, including Core Purpose, Key Capabilities, and Hierarchy/Collaboration details, is intended to be automatically generated and maintained by a build script (e.g., `build_mode_selection_guide_data.js`) parsing individual `.mode.md` files. Manual updates to the detailed list below this point should generally be avoided.)*

---

### `roo-commander` (👑 Roo Commander)

*   **Core Purpose:** Highest-level coordinator for software development projects, managing goals, delegation, and project state.
*   **Key Capabilities:** Implementation
*   **Hierarchy & Collaboration:**
    *   **Typical Delegators:** `user`
    *   **Typical Reports To:** `user`
    *   **Frequent Collaborators:** `prime-coordinator`, `lead-*`, `manager-*`, `core-architect`

---

*(... Auto-generated details for all other modes would follow here ...)*

---

## 6. Maintaining This Guide 🔄

The general principles and explanations in Sections 1-4 are maintained manually. The detailed mode information in Section 5 should be kept up-to-date automatically by running the relevant build script (e.g., `build_mode_selection_guide_data.js` or `build_roomodes.js` if it also handles this) whenever modes are added, removed, or significantly changed.