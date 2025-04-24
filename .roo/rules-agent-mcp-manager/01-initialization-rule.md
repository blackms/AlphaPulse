+++
id = "AGENT-MCP-MANAGER-RULE-INIT-V1"
title = "MCP Manager Agent: Rule - Initial Request Processing"
context_type = "rules"
scope = "Initial user interaction handling for MCP installation and management and management"
target_audience = ["agent-mcp-manager"]
granularity = "procedure"
status = "active"
last_updated = "2025-04-24" # Use current date
tags = ["rules", "workflow", "initialization", "mcp", "install", "manage", "update", "agent-mcp-manager"]
related_context = [
    ".ruru/modes/agent-mcp-manager/agent-mcp-manager.mode.md",
    # Specific KB Action files
    ".ruru/modes/agent-mcp-manager/kb/install-vertex-ai.md",
    ".ruru/modes/agent-mcp-manager/kb/install-unsplash.md",
    ".ruru/modes/agent-mcp-manager/kb/install-custom-url.md",
    ".ruru/modes/agent-mcp-manager/kb/remove-mcp-server.md" # Added remove KB
    ]
template_schema_doc = ".ruru/templates/toml-md/16_ai_rule.README.md"
relevance = "Critical: Defines the entry point for the MCP installer mode"
+++

# Rule: Initial Request Processing (MCP Manager Agent)

This rule governs how you handle the initial interaction when activated, presenting MCP server installation and management options and routing to specific KB procedures.

**Standard Initial Prompt:**

```xml
 <ask_followup_question>
  <question>Welcome to the MCP Manager Agent! What would you like to do?</question>
  <follow_up>
    <suggest>🔌 Install Vertex AI Server</suggest> <!-- Option 1 -->
    <suggest>🖼️ Install Unsplash Image Server (Placeholder)</suggest> <!-- Option 2 -->
    <suggest>🌐 Install Custom Server from Git URL (Placeholder)</suggest> <!-- Option 3 -->
    <suggest>🗑️ Remove an existing MCP Server</suggest> <!-- Option 4 -->
    <suggest>🔄 Check for MCP Server Updates (Placeholder)</suggest> <!-- Option 5 -->
    <suggest>❌ Cancel</suggest> <!-- Option 6 -->
  </follow_up>
 </ask_followup_question>
```

**Procedure:**

1.  **Present Options:** Present the **Standard Initial Prompt** (defined above) using `<ask_followup_question>`.
2.  **Await User Selection:** Wait for the user's choice.
3.  **Handle User Selection:**
    *   Once the user selects an option:
        1.  Identify the selected option.
        2.  Log the chosen installation path.
        3.  **Execute the detailed procedure defined in the corresponding KB file (adding placeholders for new options) (adding placeholders for new options)**:
            *   Option 1 (Vertex AI): Execute KB `.ruru/modes/agent-mcp-manager/kb/install-vertex-ai.md`.
            *   Option 2 (Unsplash): Execute KB `.ruru/modes/agent-mcp-manager/kb/install-unsplash.md`. (Currently a placeholder).
            *   Option 3 (Custom URL): Execute KB `.ruru/modes/agent-mcp-manager/kb/install-custom-url.md`. (Currently a placeholder).
            *   Option 4 (Remove): Execute KB `.ruru/modes/agent-mcp-manager/kb/remove-mcp-server.md`.
            *   Option 5 (Update): Execute KB `.ruru/modes/agent-mcp-manager/kb/check-updates.md`. (Placeholder).
            *   Option 6 (Cancel): Report cancellation to the coordinator (`roo-commander`) using `<attempt_completion>`. **Stop.**
        4.  Follow the steps within that specific KB procedure. **End this initialization workflow** upon completion or cancellation of the KB procedure.

**Key Objective:** To provide clear installation and management options and route the user interaction to the precise, detailed procedure stored in the relevant Knowledge Base file for the selected MCP server or management task.