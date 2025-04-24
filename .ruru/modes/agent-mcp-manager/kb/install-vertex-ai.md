+++
# --- Basic Metadata ---
id = "KB-MCP-MANAGER-VERTEX-AI-V1"
title = "KB: Install Vertex AI MCP Server (Interactive)"
status = "active"
created_date = "2025-04-24"
updated_date = "2025-04-24"
version = "1.0"
tags = ["kb", "agent-mcp-manager", "workflow", "mcp", "install", "vertex-ai", "configuration", "setup", "interactive"]

# --- Ownership & Context ---
owner = "agent-mcp-manager" # Changed
related_docs = [
    ".roo/rules-agent-mcp-manager/01-initialization-rule.md", # Changed
    ".roo/mcp.json"
]
related_templates = []

# --- Workflow Specific Fields ---
objective = "Interactively guide the user through installing and configuring the Vertex AI MCP server (`https://github.com/shariqriazz/vertex-ai-mcp-server`), handling prerequisites, cloning, dependencies, environment configuration, and updating Cline's `.roo/mcp.json`."
scope = "Executed when user selects 'Install Vertex AI Server'. Performs checks, file system operations, user prompts, and configuration updates."
roles = ["Agent (agent-mcp-manager)", "User"] # Changed
trigger = "User selection of 'Install Vertex AI Server' from the initial prompt."
success_criteria = [
    "Prerequisites (git, bun, GCP Auth method) confirmed/checked.",
    "Vertex AI MCP server repository is successfully cloned into `.ruru/mcp-servers/vertex-ai` (or skipped if existing and confirmed).",
    "Dependencies are successfully installed using `bun install`.",
    "User provides necessary GCP configuration details (Project ID, Location, Auth method/path).",
    "`.ruru/mcp-servers/vertex-ai/.env` file is created with user-provided details.",
    "Existing `.roo/mcp.json` is backed up (if present).",
    "`.roo/mcp.json` is created or updated with valid JSON for the server, including dynamic `alwaysAllow` list.",
    "User is instructed to reload Cline."
]
failure_criteria = [
    "User cancels.",
    "Prerequisites not met or check fails.",
    "Clone, directory creation, install, or file copy/write commands fail.",
    "User provides invalid or incomplete configuration details.",
    "Agent fails to read/write/parse `.roo/mcp.json` correctly.", # Changed
    "Dynamic tool list generation fails."
]

# --- Integration ---
acqa_applicable = false
pal_validated = false
validation_notes = "Needs thorough testing for different scenarios (new install, existing dir, existing mcp.json, different OS, ADC vs Key auth)."

# --- AI Interaction Hints (Optional) ---
context_type = "workflow_step_details"
target_audience = ["agent-mcp-manager"] # Changed
granularity = "detailed"
+++

# KB Procedure: Install Vertex AI MCP Server (Interactive)

This procedure interactively installs and configures the Vertex AI MCP server.

## 1. Objective 🎯
*   Interactively guide the user to install and configure the Vertex AI MCP server, ensuring correct setup and configuration in `.roo/mcp.json`.

## 2. Roles & Responsibilities 👤
*   **Agent (agent-mcp-installer):** Manages the flow, checks prerequisites, executes commands, prompts user, handles configuration files.
*   **User:** Confirms actions, provides GCP details.

## 3. Procedure Steps 🪜

*   **Step 1: Prerequisite Check (Agent Task)**
    *   **Description:** Inform the user and check for `git`, `bun`, and GCP auth readiness.
    *   **Inputs:** Triggered by user selecting Option 1.
    *   **Tool:** `ask_followup_question`, `execute_command`
    *   **Procedure:**
        1.  Inform: "This flow will install the Vertex AI MCP server from GitHub. Requires `git` and `bun` installed, plus Google Cloud access."
        2.  Ask Auth Method:
            ```xml
            <ask_followup_question>
             <question>First, how will you authenticate with Google Cloud?
             1. Application Default Credentials (ADC - requires `gcloud auth application-default login` run previously)
             2. Service Account Key file (JSON)
             </question>
             <follow_up>
               <suggest>1. Use Application Default Credentials (ADC)</suggest>
               <suggest>2. Use a Service Account Key file</suggest>
               <suggest>Help me check git/bun prerequisites first</suggest>
               <suggest>Cancel installation</suggest>
             </follow_up>
            </ask_followup_question>
            ```
        3.  **Handle Response:**
            *   If "Cancel": Report cancellation to coordinator using `<attempt_completion>`. **Stop.**
            *   If "Help": Execute `git --version` and `bun --version` (check OS Rule 05). Report output. If fails, guide user to install manually and restart flow. If succeeds, re-ask the Auth Method question above.
            *   If "ADC" chosen: Store `[Auth Method]` = "ADC". Proceed to Step 2.
            *   If "Service Account Key" chosen: Store `[Auth Method]` = "Key". Inform: "Okay, you'll need the full path to your key file later. Here's a video guide on creating one if needed: https://vimeo.com/1075028909/098f77b209?share=copy (Note: link may need updating for specific MCP setup)". Proceed to Step 2.
    *   **Outputs:** User confirmation to proceed, `[Auth Method]` stored.

*   **Step 2: Check/Create Target Directory & Clone (Agent Task)**
    *   **Description:** Ensure the target directory exists or clone into it.
    *   **Inputs:** User confirmation. Workspace root path (`{Current Working Directory}`).
    *   **Tool:** `list_files`, `ask_followup_question`, `execute_command`
    *   **Procedure:**
        1.  Set `target_dir_rel = ".ruru/mcp-servers/vertex-ai"`
        2.  Set `parent_dir_rel = ".ruru/mcp-servers"`
        3.  Check if `target_dir_rel` exists: `<list_files><path>[target_dir_rel]</path></list_files>`
        4.  **If** `list_files` succeeds (directory exists):
            *   Use `<ask_followup_question>`: "Directory `[target_dir_rel]` already exists. Skip cloning and proceed assuming it contains the server code, or remove and re-clone?" Options: "Skip cloning", "Remove and re-clone", "Cancel".
            *   If "Skip cloning": Proceed to Step 3.
            *   If "Remove and re-clone":
                *   Confirm: Use `<ask_followup_question>` "⚠️ **Confirm:** Really remove directory `[target_dir_rel]` and all its contents?" Options: "Yes, remove and clone", "Cancel".
                *   If "Yes": Execute `rm -rf [target_dir_rel]` (Check OS Rule 05!). Handle errors. If successful, proceed to clone step below.
                *   If "Cancel": Report cancellation to coordinator. **Stop.**
            *   If "Cancel": Report cancellation to coordinator. **Stop.**
        5.  **Else (If `list_files` fails - directory likely doesn't exist):**
            *   Explain: "Creating parent directory `[parent_dir_rel]` if it doesn't exist..."
            *   Execute `mkdir -p [parent_dir_rel]` (Check OS Rule 05!). Handle errors.
            *   Explain: "Cloning repository into `[target_dir_rel]`..."
            *   Execute: `<execute_command><command>git clone https://github.com/shariqriazz/vertex-ai-mcp-server [target_dir_rel]</command><cwd>{Current Working Directory}</cwd></execute_command>`
            *   Await result. If `exit_code` != 0: Report specific error to coordinator. **Stop.**
    *   **Outputs:** Cloned repository in `.ruru/mcp-servers/vertex-ai`.

*   **Step 3: Install Dependencies (Agent Task)**
    *   **Description:** Install Node.js dependencies using Bun.
    *   **Inputs:** Successful clone/skip in Step 2.
    *   **Tool:** `execute_command`
    *   **Procedure:**
        1.  Explain: "Installing dependencies using Bun (this includes a build step)..."
        2.  Execute: `<execute_command><command>bun install</command><cwd>.ruru/mcp-servers/vertex-ai</cwd></execute_command>`
        3.  Await result.
    *   **Outputs:** Dependencies installed.
    *   **Error Handling:** If `exit_code` != 0 or `stderr` shows errors: Report specific error to coordinator. **Stop.**

*   **Step 4: Prompt for GCP Config & Write `.env` (Agent Task)**
    *   **Description:** Get user's GCP details and write them to the `.env` file directly.
    *   **Inputs:** Successful install in Step 3, `[Auth Method]` from Step 1.
    *   **Tool:** `ask_followup_question`, `write_to_file`
    *   **Procedure:**
        1.  Ask for details:
            ```xml
            <ask_followup_question>
             <question>Please provide your Google Cloud configuration:
             - Google Cloud Project ID?
             - Google Cloud Location (e.g., us-central1)?
             - { If [Auth Method] == 'Key': "The FULL, ABSOLUTE path to your Service Account Key JSON file?" Else: "" }
             </question>
             <follow_up>
               <suggest>Project ID: [Your-Project-ID], Location: [Your-Location]{ If [Auth Method] == 'Key': ", Key Path: [/full/path/to/your/key.json]" Else: "" }</suggest>
             </follow_up>
            </ask_followup_question>
            ```
        2.  Await response. Parse and store `[Project ID]`, `[Location]`, and optional `[Credentials Path]`. Validate inputs (e.g., path looks like a path if provided). If invalid, re-prompt.
        3.  Construct `.env` content string:
            ```env
            # Required
            GOOGLE_CLOUD_PROJECT="[Project ID]"
            GOOGLE_CLOUD_LOCATION="[Location]"

            # Optional Authentication (set if using Service Account Key)
            { If [Auth Method] == 'Key': 'GOOGLE_APPLICATION_CREDENTIALS="[Credentials Path]"' Else: '# GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"' }
            ```
            (Replace placeholders, ensure path is quoted if it contains spaces, handle commenting correctly based on auth method).
        4.  Explain: "Creating/updating the environment file at `.ruru/mcp-servers/vertex-ai/.env`..."
        5.  Execute: `<write_to_file><path>.ruru/mcp-servers/vertex-ai/.env</path><content>[Constructed .env content]
