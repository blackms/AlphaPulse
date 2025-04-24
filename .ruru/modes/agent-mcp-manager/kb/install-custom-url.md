+++
id = "KB-MCP-MANAGER-CUSTOM-URL-V0.1"
title = "KB: Install Custom MCP Server from Git URL (Placeholder)"
status = "placeholder"
created_date = "2025-04-24"
updated_date = "2025-04-24"
version = "0.1"
tags = ["kb", "agent-mcp-manager", "workflow", "mcp", "install", "custom", "git", "placeholder"]
owner = "agent-mcp-manager"
related_docs = [".roo/rules-agent-mcp-manager/01-initialization-rule.md"]
objective = "Placeholder for the generic custom MCP server installation procedure via Git URL."
scope = "This procedure is not yet fully implemented. It requires defining a standard process for handling arbitrary MCP servers."
roles = ["Agent (agent-mcp-manager)", "User"]
trigger = "User selection of 'Install Custom Server from Git URL'."
success_criteria = ["N/A"]
failure_criteria = ["Always fails (not implemented)."]
+++

# KB Procedure: Install Custom MCP Server from Git URL (Placeholder)

**This installation procedure is not yet fully implemented.**

Installing a custom MCP server requires a more defined process to handle potential variations in build steps, configuration requirements, and execution commands.

**Future Steps (Conceptual):**

1.  Prompt user for the Git repository URL.
2.  Prompt user for the desired local directory name (e.g., `.ruru/mcp-servers/my-custom-server`).
3.  Clone the repository.
4.  Attempt standard dependency installation (e.g., `bun install`, `npm install`, `pip install -r requirements.txt` - may need user guidance).
5.  Prompt user for the main execution script path within the repo.
6.  Prompt user for any necessary environment variables.
7.  Attempt to create `.env` file.
8.  Update `.roo/mcp.json` with the provided details.

*(Agent: Report this limitation to the coordinator using `<attempt_completion>` and indicate the task is blocked.)*