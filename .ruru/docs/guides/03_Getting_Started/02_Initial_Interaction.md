+++
# --- Basic Metadata ---
id = "KB-RC-SETUP-INITIAL-INTERACTION"
title = "Getting Started: Your First Interaction with Roo Commander"
status = "draft"
difficulty = "beginner"
estimated_time = "~5 minutes"
target_audience = ["users"]
prerequisites = ["Roo Commander installed (See 01_Installation_Setup.md)"]
learning_objectives = ["Understand the purpose of Roo Commander's initial prompt", "Recognize the different starting options and their general function", "Know how Roo Commander routes requests to specific workflows"]
template_schema_doc = ".ruru/templates/toml-md/10_guide_tutorial.README.md"
tags = ["roo-commander", "getting-started", "initialization", "workflow", "user-interface", "guide", "tutorial", "options"]
related_docs = [
    "../README.md", # Link to the KB README
    "01_Installation_Setup.md",
    "../../../../.roo/rules-roo-commander/02-initialization-workflow-rule.md", # Link to the governing rule
    "../../../../.ruru/modes/roo-commander/kb/initial-actions/README.md" # Link to the KB actions index
    ]
+++

# Getting Started: Your First Interaction with Roo Commander

## 1. Introduction / Goal 🎯

When you first activate the **👑 Roo Commander** mode in the Roo Code chat interface, or sometimes when your initial request is broad, Roo Commander will present you with a standard set of starting options.

This guide explains the purpose of this initial interaction and what happens when you select one of the options. The goal is to help you quickly direct Roo Commander towards the most relevant workflow for your immediate needs.

## 2. The Initial Prompt: Why So Many Options? 🤔

You'll typically see a prompt similar to this:

```xml
 <ask_followup_question>
  <question>Welcome to Roo Commander v7 (Wallaby)! How can I assist you today?</question>
  <follow_up>
    <suggest>🔌 Install/Manage MCP Servers</suggest> <!-- Option 0 -->
    <suggest>🎩 Start a NEW project from scratch</suggest> <!-- Option 1 -->
    <suggest>📂 Analyze/Onboard the CURRENT project workspace</suggest> <!-- Option 2 -->
    <suggest>🌐 Clone a Git repository & onboard</suggest> <!-- Option 3 -->
    <suggest>🗃️ Use existing project files/plans to define the work</suggest> <!-- Option 4 -->
    <suggest>📑 Plan/Design a new feature or project</suggest> <!-- Option 5 -->
    <suggest>🐞 Fix a specific bug</suggest> <!-- Option 6 -->
    <suggest>♻️ Refactor or improve existing code</suggest> <!-- Option 7 -->
    <suggest>✍️ Write or update documentation</suggest> <!-- Option 8 -->
    <suggest>📟 Review project status / Manage tasks (MDTM)</suggest> <!-- Option 9 -->
    <suggest>❓ Research a topic / Ask a technical question</suggest> <!-- Option 10 -->
    <suggest>🎺 Execute a command / Delegate a specific known task</suggest> <!-- Option 11 -->
    <suggest>🪃 Manage Roo Configuration (Advanced)</suggest> <!-- Option 12 -->
    <suggest>🖲️ Update my preferences / profile</suggest> <!-- Option 13 -->
    <suggest>🦘 Learn about Roo Commander capabilities</suggest> <!-- Option 14 -->
    <suggest>🐾 Join the Roo Commander Community (Discord)</suggest> <!-- Option 15 -->
    <suggest>🤔 Something else... (Describe your goal)</suggest> <!-- Option 16 -->
    <suggest>🧑‍🎨 Mode Management</suggest> <!-- Option 17 -->
  </follow_up>
 </ask_followup_question>
```

**Purpose:** Instead of trying to guess your exact intent from a potentially vague initial prompt (like "Help me with my project"), Roo Commander presents these common starting points. Each option acts as a **shortcut** to a specific, predefined workflow designed for that common scenario.

This structured approach helps Roo Commander:
*   **Clarify Intent:** Quickly understand your primary goal.
*   **Activate Correct Workflow:** Initiate the most relevant sequence of actions and delegate to the right specialist modes from the start.
*   **Save Time:** Avoid lengthy back-and-forth clarification dialogues for common starting tasks.

## 3. How It Works: Routing to KB Procedures 🗺️

This initial interaction is governed by a specific rule: **`.roo/rules-roo-commander/02-initialization-workflow-rule.md`**.

When you select an option (e.g., "🐞 Fix a specific bug"), this rule instructs Roo Commander to execute a detailed procedure defined in a corresponding Knowledge Base (KB) file located within `.ruru/modes/roo-commander/kb/initial-actions/`.

*   **Example:** Selecting Option 6 ("🐞 Fix a specific bug") triggers the execution of the procedure defined in `.ruru/modes/roo-commander/kb/initial-actions/06-fix-bug.md`. This KB file tells Commander to ask you for bug details and then delegate the task to the `dev-fixer` mode.

Each option (0-16, plus Mode Management) has a corresponding KB file in the `initial-actions` directory that outlines the specific steps Commander will take, including any necessary user prompts or delegations to other modes.

## 4. The "Something Else" Option 🤔

If your goal doesn't neatly fit into the predefined options, choose **"🤔 Something else... (Describe your goal)"**. This will prompt you to provide a more detailed description. Roo Commander will then analyze your description and attempt to route you to the most appropriate workflow or specialist mode, potentially falling back to its general coordination capabilities if the goal is complex or unique.

## 5. Conclusion ✅

The initial prompt with its multiple options is a core part of Roo Commander's strategy for efficient workflow initiation. By selecting the option that best matches your immediate goal, you help Commander quickly understand your intent and activate the most relevant procedures and specialist modes, streamlining the start of your development task.