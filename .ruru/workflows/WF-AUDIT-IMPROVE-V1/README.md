+++
# --- Basic Metadata (Auto-Generated) ---
id = "WF-AUDIT-IMPROVE-V1" # Workflow ID
title = "Workflow Audit & Improvement" # Human-readable title
description = "Audits an existing workflow definition for correctness, compliance, efficiency, and suggests/applies improvements iteratively." # Brief description
version = "1.0" # Version of the workflow definition
status = "draft" # draft, active, deprecated, retired
created_date = "2025-04-30" # Date workflow was created
last_updated = "2025-04-30" # Date workflow was last updated
author = "technical-architect" # Mode or user who created/maintains it

# --- Execution & Triggering ---
trigger_type = "manual" # manual, event-driven, scheduled
# trigger_condition = "" # If event-driven/scheduled, describe condition
# default_inputs = { target_workflow_dir = ".ruru/workflows/WF-TARGET-V1/" } # Example default inputs

# --- Context & Dependencies ---
required_modes = ["util-workflow-manager", "util-reviewer", "agent-research", "technical-architect"] # Modes needed for execution
required_mcps = ["vertex-ai-mcp-server"] # MCPs needed (optional)
required_rules = [
    ".roo/rules/01-standard-toml-md-format.md",
    ".roo/rules/04-mdtm-workflow-initiation.md",
    ".roo/rules/06-iterative-execution-policy.md",
    ".roo/rules/08-logging-procedure-simplified.md",
    ".roo/rules/10-vertex-mcp-usage-guideline.md"
    # Add specific rules for workflow structure/quality if they exist
]
required_templates = [
    ".ruru/templates/toml-md/23_workflow_readme.md",
    ".ruru/templates/toml-md/24_workflow_step_start.md",
    ".ruru/templates/toml-md/25_workflow_step_standard.md",
    ".ruru/templates/toml-md/26_workflow_step_finish.md"
]
# related_workflows = [] # Other workflows this one interacts with

# --- Input/Output & State ---
# Define expected inputs and outputs for the workflow as a whole
inputs_schema = """
{
  "type": "object",
  "properties": {
    "target_workflow_dir": {
      "type": "string",
      "description": "Path to the root directory of the workflow to be audited (e.g., '.ruru/workflows/WF-TARGET-V1/')."
    },
    "max_iterations": {
        "type": "integer",
        "description": "Maximum number of improvement iterations allowed.",
        "default": 3
    },
    "audit_report_path": {
        "type": "string",
        "description": "Path where the final audit report will be saved.",
        "default": ".ruru/docs/audits/WF-AUDIT-IMPROVE-V1/audit_report.md"
    },
    "consolidated_findings_path": {
        "type": "string",
        "description": "Path to store the consolidated findings before approval.",
        "default": ".ruru/temp/audit_findings.json"
    },
    "approved_changes_path": {
        "type": "string",
        "description": "Path to store the approved changes before application.",
        "default": ".ruru/temp/approved_changes.json"
    },
    "available_modes": {
        "type": "array",
        "items": { "type": "string" },
        "description": "List of available mode slugs in the current environment. Required for validating delegation targets."
    }
  },
  "required": ["target_workflow_dir", "available_modes"]
}
"""
outputs_schema = """
{
  "type": "object",
  "properties": {
    "audit_result": {
      "type": "string",
      "description": "Summary status of the audit (e.g., 'Completed with improvements', 'Completed, no changes needed', 'Failed')."
    },
    "final_report_path": {
      "type": "string",
      "description": "Path to the final generated audit report."
    },
    "iterations_performed": {
        "type": "integer",
        "description": "Number of improvement iterations performed."
    }
  },
  "required": ["audit_result", "final_report_path", "iterations_performed"]
}
"""
state_variables = ["target_workflow_dir", "iteration_count", "max_iterations", "audit_report_path", "consolidated_findings_path", "approved_changes_path", "workflow_files_content", "rules_content", "templates_content", "structure_analysis_results", "logic_analysis_results", "simulation_results", "proceed_with_changes", "available_modes"] # Key variables managed across steps (Commented out, for documentation)

# --- Diagram (Optional) ---
# mermaid_diagram = """
# graph TD
#     A[Start] --> B(Step 1);
#     B --> C{Decision?};
#     C -- Yes --> D[Step 2a];
#     C -- No --> E[Step 2b];
#     D --> F[End];
#     E --> F;
# """

# --- Tags & Categorization ---
tags = ["workflow", "audit", "improvement", "quality", "compliance", "refactor", "meta"]

# --- Notes ---
# Free-form notes about the workflow
# - This workflow is designed to be run by a coordinator mode like roo-commander or technical-architect.
# - It leverages specialist modes for detailed analysis and modification.
# - Iteration limit prevents infinite loops.
+++

# Workflow: Workflow Audit & Improvement (WF-AUDIT-IMPROVE-V1)

## 1. Purpose

This workflow provides a systematic process for auditing existing workflow definitions (`.ruru/workflows/*`). It checks for structural correctness, compliance with project standards (TOML+MD, naming), logical flow, efficiency, error handling, and overall quality. It facilitates iterative improvement by suggesting changes, seeking approval, applying them via delegation, and re-evaluating.

## 2. Workflow Steps

The workflow proceeds through the following high-level steps:

1.  **Start:** Initialize the audit process, receiving the target workflow directory path.
2.  **Gather Context:** Read all files from the target workflow, relevant project rules, and standard workflow templates.
3.  **Analyze Structure:** Delegate structural and format validation (TOML+MD, naming, required fields) to a specialist mode (e.g., `util-workflow-manager`).
4.  **Review Logic & Quality:** Delegate a deeper review of the workflow's logic, efficiency, error handling, and best practice adherence to appropriate modes (e.g., `util-reviewer`, `agent-research` + Vertex AI).
5.  **Consolidate Findings:** Aggregate results and actionable suggestions from the analysis steps into a structured format.
6.  **Request Approval:** Present the consolidated findings and suggestions to the initiating user/coordinator for approval. Determine if changes are requested.
7.  **Apply Changes (Conditional):** If changes are approved and the iteration limit is not reached, delegate the application of specific changes to `util-workflow-manager`. Loop back to re-analyze.
8.  **Simulate Flow:** Perform static checks on step connections (`next_step`, `error_step`, `delegate_to`, input/output consistency).
9.  **Finish:** Generate a final audit report summarizing the process, findings, changes made, and simulation results.

## 3. Inputs

*   `target_workflow_dir` (string, required): Path to the workflow directory to audit.
*   `max_iterations` (integer, optional, default: 3): Maximum improvement loops.
*   `available_modes` (list[string], required): List of available mode slugs provided by the execution environment (needed for validation steps).
*   `available_modes` (list[string], required): List of available mode slugs provided by the execution environment (needed for validation steps).
*   `available_modes` (list[string], required): List of available mode slugs provided by the execution environment (needed for validation steps).
*   `available_modes` (list[string], required): List of available mode slugs provided by the execution environment.

## 4. Outputs

*   `audit_result` (string): Summary status of the audit.
*   `final_report_path` (string): Path to the generated audit report.
*   `iterations_performed` (integer): Number of improvement iterations done.

## 5. Error Handling

*   Errors during context gathering, delegation, or file operations trigger the `EE_audit_error.md` step.
*   The error step logs the failure and terminates the workflow gracefully, reporting the failure state.

## 6. Key Considerations

*   Relies heavily on delegation to specialist modes. Ensure these modes are available and correctly configured.
*   The quality of the audit depends on the capabilities of the delegated review modes and the clarity of project standards/rules.
*   The static simulation is a basic check and does not guarantee runtime correctness.