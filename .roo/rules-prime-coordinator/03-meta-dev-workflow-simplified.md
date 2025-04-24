+++
id = "PRIME-RULE-METADEV-SIMPLE-V3" # Incremented version
title = "Prime Coordinator: Rule - Meta-Development Workflow Trigger (Simplified)"
context_type = "rules"
scope = "Routing requests to modify Roo Commander configuration files"
target_audience = ["prime"]
granularity = "procedure"
status = "active"
last_updated = "2025-04-24" # Updated date
tags = ["rules", "workflow", "meta-development", "configuration", "staging", "safety", "prime", "confirmation", "auto-apply", "generated-files"] # Added tags
related_context = [
    "01-operational-principles.md",
    "02-request-analysis-dispatch.md",
    "07-logging-confirmation-simplified.md",
    ".ruru/modes/prime-coordinator/kb/05-meta-dev-staging-confirm-auto-apply.md", # Staging KB ref
    ".ruru/modes/prime-coordinator/kb/07-meta-dev-direct-auto-apply-operational.md", # Operational KB ref
    "prime-txt", "prime-dev",
    "build_roomodes.js" # Related build script
    ]
template_schema_doc = ".ruru/templates/toml-md/16_ai_rule.README.md"
relevance = "High: Core workflow for configuration changes"
+++

# Rule: Meta-Development Workflow Trigger (Simplified)

This rule defines how to handle requests (Type A from Rule `02`) to modify Roo Commander configuration files, implementing a simplified workflow based on path sensitivity.

**Procedure:**

1.  **Receive Request:** Obtain the target file path (`TARGET_PATH`) and desired changes.
2.  **Define Path Patterns:**
    *   `PROTECTED_CORE_PATHS` = patterns matching: `.roo/rules/**`, `.ruru/modes/roo-commander/**`, `.roo/rules-roo-commander/**`, `.ruru/modes/prime*/**`, `.roo/rules-prime*/**`, `build_*.js`, `create_build.js`. (These require staging and confirmation).
    *   `GENERATED_CONFIG_PATHS` = patterns matching: `.roomodes*`. (These are generated files, editing directly is discouraged).
3.  **Check Path & Execute Appropriate Workflow:**
    *   **IF `TARGET_PATH` matches `GENERATED_CONFIG_PATHS`:** Initiate **Generated File Handling**.
        *   *Brief:* **Do not proceed with direct edit.** Ask the user for confirmation, explaining that this file is auto-generated. Strongly suggest running the appropriate build script (e.g., `node build_roomodes.js`) instead of manual editing. If the user insists on manual editing *after* the warning, proceed with the "Staging + Confirmation + Auto-Apply Workflow" below as if it were a `PROTECTED_CORE_PATHS` file.
    *   **ELSE IF `TARGET_PATH` matches `PROTECTED_CORE_PATHS`:** Initiate the **Staging + Confirmation + Auto-Apply Workflow**.
        *   *Brief:* Copy to staging -> Delegate edit to worker -> Generate diff -> Present diff to user & ask for confirmation -> If confirmed, Coordinator applies change to original -> Cleanup staging.
        *   Consult **KB `.ruru/modes/prime-coordinator/kb/05-meta-dev-staging-confirm-auto-apply.md`** for the detailed procedure.
    *   **ELSE (Operational Config - not protected):** Initiate the **Direct Auto-Apply Workflow (Operational)**.
        *   *Brief:* Delegate edit directly to the appropriate worker (`prime-txt`/`prime-dev`) -> Worker applies change directly (worker's internal confirmation rule still applies) -> Worker reports completion/failure.
        *   Consult **KB `.ruru/modes/prime-coordinator/kb/07-meta-dev-direct-auto-apply-operational.md`** for the detailed procedure.

**Key Objective:** Ensure maximum safety (staging + user confirmation) for all core configuration, rules, modes, and build scripts (`PROTECTED_CORE_PATHS`). Discourage direct editing of generated files (`GENERATED_CONFIG_PATHS`) by suggesting build scripts first. Allow direct application by workers for general operational configuration, while consistently respecting the worker's own confirmation rules before any write action.