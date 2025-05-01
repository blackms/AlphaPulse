+++
# --- Basic Metadata (Auto-Generated) ---
id = "WF-AUDIT-IMPROVE-V1_01_gather_context" # Unique ID for this step
step_number = 1 # Step sequence number
# --- Workflow Step Configuration ---
title = "Gather Audit Context"
description = "Reads the target workflow files, relevant project rules, and standard workflow templates needed for analysis."
step_type = "standard" # start, standard, finish, error
# --- Execution Details ---
# mode = "self" # Optional: Mode responsible for executing this step (defaults to workflow executor)
# delegate_to = "discovery-agent" # Optional: Delegate execution to another mode
# --- Input/Output ---
inputs = ["target_workflow_dir"] # List of input variables required by this step
outputs = ["workflow_files_content", "rules_content", "templates_content"] # List of output variables produced by this step
# --- Control Flow ---
next_step = "02a_check_structure.md" # The next step file to execute
error_step = "EE_audit_error.md" # Optional: Step file to execute on error
# --- Logging & Auditing ---
# log_level = "info" # Default logging level for this step
# --- Tags & Categorization ---
tags = ["context", "read", "rules", "templates", "workflow-files"]
# --- Notes ---
# - Reads all *.md files in the target workflow directory.
# - Reads specific rules and templates defined in the workflow README.
# - Consider using `read_multiple_files_content` MCP tool if available for efficiency.
+++

# Step 01: Gather Audit Context

## Description

This step gathers all necessary information required to perform the workflow audit. It reads the content of the target workflow's definition files, relevant project-wide rules (like formatting standards), and the standard workflow step templates used for comparison.

## Actions

1.  **List Target Workflow Files:** Use `list_files` to get all `.md` files within the `target_workflow_dir`.
2.  **Read Workflow Files:** Use `read_file` (or `read_multiple_files_content` via MCP if available and efficient) to read the content of all identified `.md` files from the target workflow directory. Store this content (e.g., in a dictionary mapping path to content) in the `workflow_files_content` output variable.
3.  **Identify Relevant Rules/Templates:** Determine the paths of required rules and templates by parsing the `required_rules` and `required_templates` fields from the *target* workflow's `README.md` content (which is part of the `workflow_files_content` gathered in Action 2).
4.  **Read Rules:** Use `read_file` (or `read_multiple_files_content`) to read the content of the identified rule files. Store in `rules_content`.
5.  **Read Templates:** Use `read_file` (or `read_multiple_files_content`) to read the content of the identified template files. Store in `templates_content`.
6.  **Log Context Gathering:** Record that context gathering is complete.

## Inputs

*   `target_workflow_dir` (from step 00)

## Outputs

*   `workflow_files_content`: A collection (e.g., dictionary/map) containing the paths and content of all files within the target workflow directory.
*   `rules_content`: A collection containing the paths and content of relevant project rules.
*   `templates_content`: A collection containing the paths and content of standard workflow templates.

## Next Step

*   Proceed to `02a_check_structure.md`.

## Error Handling

*   If listing or reading files fails, transition to `EE_audit_error.md`.