+++
id = "PRIME-RULE-METADEV-SIMPLE-V1"
title = "Prime Coordinator: Rule - Meta-Development Workflow Trigger"
context_type = "rules"
scope = "Routing requests to modify Roo Commander configuration files"
target_audience = ["prime"]
granularity = "procedure"
status = "active"
last_updated = "2025-04-22"
tags = ["rules", "workflow", "meta-development", "configuration", "staging", "safety", "prime"]
related_context = [
    "01-operational-principles.md",
    "02-request-analysis-dispatch.md",
    "07-logging-confirmation-simplified.md",
    ".ruru/modes/prime-coordinator/kb/03-meta-dev-staging-workflow.md", # Points to KB
    ".ruru/modes/prime-coordinator/kb/04-meta-dev-direct-workflow.md", # Points to KB
    "prime-txt", "prime-dev"
    ]
template_schema_doc = ".ruru/templates/toml-md/16_ai_rule.README.md"
relevance = "High: Core workflow for configuration changes"
+++

# Rule: Meta-Development Workflow Trigger

This rule defines how to handle requests (Type A from Rule `02`) to modify Roo Commander configuration files.

**Procedure:**

1.  **Receive Request:** Obtain the target file path (`TARGET_PATH`) and desired changes.
2.  **Define Protected Paths:** `PROTECTED_PATHS` = patterns matching: `.roo/rules/**`, `.ruru/modes/roo-commander/**`, `.roo/rules-roo-commander/**`, `.ruru/modes/prime*/**`, `.roo/rules-prime*/**`, `.roomodes*`, `build_*.js`, `create_build.js`.
3.  **Check Path:** Determine if `TARGET_PATH` matches any pattern in `PROTECTED_PATHS`.
4.  **Execute Appropriate Workflow:**
    *   **IF Protected:** Initiate the **Staging Workflow**. Consult **KB `.ruru/modes/prime-coordinator/kb/03-meta-dev-staging-workflow.md`** for the detailed procedure (Copy -> Delegate Edit to Staging -> Diff -> Present -> Cleanup).
    *   **ELSE (Operational Config):** Initiate the **Direct Edit Workflow**. Consult **KB `.ruru/modes/prime-coordinator/kb/04-meta-dev-direct-workflow.md`** for the detailed procedure (Assess Risk -> Delegate Edit -> Worker Confirmation -> Report). Remember workers (`prime-txt`/`prime-dev`) require user confirmation before writing.

**Key Objective:** Ensure protected core files are modified via the safer Staging Workflow, while allowing direct (but confirmed) edits for operational configuration files via Prime workers.