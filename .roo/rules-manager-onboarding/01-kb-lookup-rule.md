+++
id = "KB-LOOKUP-MANAGER-ONBOARDING"
title = "KB Lookup Rule: Project Onboarding"
context_type = "rules"
scope = "Knowledge Base Lookup Configuration" # Added a reasonable scope based on title
target_audience = ["manager-onboarding"] # Added target audience based on mode slug
target_mode_slug = "manager-onboarding" # Added field
kb_directory = ".modes/manager-onboarding/kb" # Added field
granularity = "ruleset"
status = "active"
last_updated = "2025-04-18" # Updated date based on current time
# version = ""
# related_context = []
tags = ["kb", "lookup", "onboarding", "manager"] # Added relevant tags
# relevance = ""
+++

# Rule: Knowledge Base Lookup for manager-onboarding Mode

This rule defines how the `manager-onboarding` mode should access its dedicated Knowledge Base (KB).

**Configuration:**

*   **KB Directory:** The system will look for relevant KB articles within the `.modes/manager-onboarding/kb` directory relative to the project root.
*   **Lookup Trigger:** This rule is typically triggered when the `manager-onboarding` mode needs specific procedural information, guidelines, or context relevant to its onboarding tasks.
*   **Matching:** The system will use semantic search or keyword matching (based on configuration) against the content of the files within the specified `kb_directory`.

**Purpose:**

To ensure the `manager-onboarding` mode has access to the necessary structured knowledge to effectively guide users through project onboarding processes, answer common questions, and provide relevant context.