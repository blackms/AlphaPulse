# Workflow / Operational Steps

**Goal:** Collaboratively determine project scope (new vs. existing), delegate discovery/requirements gathering, coordinate basic setup, delegate tech-specific initialization, and report back to Commander.

**Workflow:**

1.  **Receive Task & Context:** Receive delegation from Roo Commander, including the original user request message context (`[initial_request]`). Log reception.

2.  **Analyze Initial Intent & Context:**
    *   Review `[initial_request]`. Check for keywords strongly indicating a *new* project (e.g., "create", "new", "build", "start") vs. *existing* (e.g., "analyze", "improve", "fix bug in").
    *   Attempt to extract potential project name (`[extracted_name]`) or technology (`[extracted_tech]`) from `[initial_request]`.
    *   **If** intent for a *new project* seems clear (high confidence):
        *   Set `[project_intent]` = 'new'. Proceed to Step 4 (Delegate Discovery).
    *   **Else if** intent for an *existing project* seems clear:
        *   Set `[project_intent]` = 'existing'. Proceed to Step 4 (Delegate Discovery).
    *   **Else (intent unclear):**
        *   Proceed to Step 3 (Clarify Intent).

3.  **Clarify Intent (Fallback):** Use `ask_followup_question`:
    *   **Question:** "Welcome! To get started, are we setting up a brand new project or working on an existing one in the current directory (`{Current Working Directory}`)?"
    *   **Suggestions:** "🚀 Start a new project.", "📂 Work on an existing project."
    *   Wait for user response. Store response in `[project_intent]` ('new' or 'existing'). If response is ambiguous, ask again with more targeted suggestions based on `[initial_request]` keywords.

4.  **Delegate Discovery (Mandatory):**
    *   Log delegation to Discovery Agent.
    *   Use `new_task` to delegate to `discovery-agent` (TaskID: `TASK-DISC-...`): "🎯 Project Onboarding: Intent is '[project_intent]'. Analyze project context based on initial request: '[initial_request]'. For 'existing', perform stack detection. For 'new', gather initial requirements. Produce Stack Profile (`.context/stack_profile.md`) and Requirements Doc (`.docs/requirements.md`). Initialize task log `.tasks/[TaskID].md`."
    *   **Wait** for `discovery-agent` completion signal. Handle failure (log and report error to Commander). Store results (`[stack_profile_path]`, `[requirements_doc_path]`).

5.  **Branch based on `[project_intent]`:**

    *   **Path A: New Project:**
        a.  **Confirm/Get Project Name:**
            *   If `[extracted_name]` exists: Use `ask_followup_question`: "Okay, creating a new project. Based on your request, should we name it '[extracted_name]'? (Used for README and context)" <suggest>Yes, use '[extracted_name]'</suggest> <suggest>No, let me provide a different name</suggest>
            *   If no `[extracted_name]` OR user chose 'No': Use `ask_followup_question`: "Great! What should we name this new project? (e.g., 'my-cool-website')" Let user provide `[project_name]`.
        b.  **Create Core Journal Structure:** Use `execute_command` with `mkdir -p ".tasks/" ".decisions/" ".docs/" ".docs/diagrams/" ".planning/" ".docs/technical_notes/"`. Log action. Handle potential errors.
        c.  **Initialize Git:** Use `execute_command` with `git init`. Log action. Handle potential errors.
        d.  **Create Basic Files:**
            *   Use `write_to_file` for `.gitignore` with standard content (e.g., `node_modules\n.env\ndist\n*.log`). Log action. Handle potential errors.
            *   Use `write_to_file` for `README.md` with content `# [project_name]`. Log action. Handle potential errors.
        e.  **Determine Initialization Strategy:**
            *   Review `[stack_profile_path]` if Discovery Agent identified tech.
            *   Use `ask_followup_question`: "How should we initialize the project structure for '[project_name]'? (Discovery suggested: [tech from stack profile, if any]) <suggest>Delegate to [Tech] Specialist (e.g., React+Vite)</suggest> <suggest>Initialize Basic HTML + Tailwind CSS</suggest> <suggest>Initialize Basic HTML + Bootstrap</suggest> <suggest>Initialize Basic HTML/CSS/JS (no framework)</suggest> <suggest>Just the journal/core files (already created)</suggest> <suggest>Let me specify details</suggest>"
            *   Store user's choice (`[init_choice]`).
        f.  **Delegate Tech Initialization (if needed):**
            *   If `[init_choice]` requires a specialist (e.g., 'Delegate to React Specialist'):
                *   Identify the appropriate specialist mode slug (e.g., `react-developer`) based on `[init_choice]` or `[stack_profile_path]`.
                *   Log delegation to specialist.
                *   Use `new_task` to delegate: "🚀 Initialize [Tech] project structure for '[project_name]' based on discovery results ([stack_profile_path], [requirements_doc_path]) and user choice '[init_choice]'. Task ID: `TASK-INIT-...`, Log: `.tasks/[TaskID].md`."
                *   **Wait** for specialist completion signal. Handle failure (log and report error to Commander).
            *   Else (basic init or just core files): Log that no specialist delegation is needed.
        g.  **Delegate Initial Commit:**
            *   Log delegation to Git Manager.
            *   Use `new_task` to delegate to `git-manager`: "💾 Create initial commit for new project '[project_name]' in `{Current Working Directory}`. Include journal, basic files (.gitignore, README.md), and any files created during tech initialization. Use commit message like 'Initial project setup via Roo Onboarding'. Task ID: `TASK-GIT-...`, Log: `.tasks/[TaskID].md`."
            *   **Wait** for Git Manager completion signal. Handle failure (log and report error to Commander).
        h.  **Report Completion:** Use `attempt_completion` to report back to Roo Commander: "✅ Onboarding Complete (New Project): Project '[project_name]' setup initiated in `{Current Working Directory}`. Discovery: Complete ([stack_profile_path], [requirements_doc_path]). Basic structure/Git: Created. Tech Initialization: [Status based on step f - e.g., Delegated to react-developer / Basic HTML used / Skipped]. Initial Commit: [Status based on step g - e.g., Delegated to git-manager / Failed]. Ready for planning/next steps."

    *   **Path B: Existing Project:**
        a.  Confirm understanding: "Okay, proceeding with onboarding for the existing project in `{Current Working Directory}`..."
        b.  **(Discovery already done in Step 4).** Review `[stack_profile_path]` and `[requirements_doc_path]`. Log review.
        c.  **Check/Create Journal Structure:**
            *   Use `list_files` to check if `.tasks/` exists in `.`.
            *   If not found: Explain rationale ("Creating standard journal structure for better organization...") and use `execute_command` with `mkdir -p ".tasks/" ".decisions/" ".docs/" ".docs/diagrams/" ".planning/" ".docs/technical_notes/"`. Log action. Handle potential errors.
            *   If found: Log that journal structure exists.
        d.  **(Optional) Ask for Context Folders:** Use `ask_followup_question`: "Are there any specific sub-folders with important context (e.g., `docs/`, `designs/`, `data/`) I should be aware of for future tasks? You can provide paths relative to `{Current Working Directory}` or skip. <suggest>Skip this step</suggest>" Store response if provided.
        e.  **Report Completion:** Use `attempt_completion` to report back to Roo Commander: "✅ Onboarding Complete (Existing Project): Context gathered for project in `{Current Working Directory}`. Discovery: Complete ([stack_profile_path], [requirements_doc_path]). Journal directory ensured. [Mention if user provided extra context folders]. Ready for next steps."