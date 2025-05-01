+++
# --- Workflow Metadata ---
id = "WF-RELEASE-NOTES-HYBRID-V3" # Updated ID and Version
title = "Workflow: Generate Release Notes (Hybrid Approach - Automated)" # Updated title
description = """
Generates release notes incrementally by analyzing Git history since the last release tag.
Determines the next version, reads GitHub config from `.ruru/config/project.toml`.
Saves incremental updates to `.ruru/docs/release-notes/next/`.
Can finalize notes into a single version file and optionally push to GitHub.
"""
version = "3.0.0" # Incremented version
status = "Draft" # Start as Draft
tags = ["workflow", "release-notes", "changelog", "git", "github", "mcp", "automation", "hybrid"] # Kept original tags

# --- Execution Control ---
entry_point = "00_start.md" # Standard entry point

# --- Interface ---
# Detailed parameter descriptions are in the Markdown body below.
# These correspond to parameters defined in 00_start.md
inputs = [
    "action: Action to perform ('add' or 'finalize', optional, default 'add').",
    "output_dir: Base output directory (optional, defaults to .ruru/docs/release-notes/). Incremental notes go into 'next/' subdirectory.",
    "push_to_github: Flag to attempt GitHub Release push during 'finalize' action (optional, default false).",
    "mark_as_draft: Flag to mark GitHub Release as draft during 'finalize' (optional, default true).",
    "mark_as_prerelease: Flag to mark GitHub Release as pre-release during 'finalize' (optional, default false)."
]
outputs = [
    "If action='add': An incremental Markdown file saved in the 'next/' subdirectory.",
    "If action='finalize': A final aggregated Markdown file for the new version.",
    "If action='finalize' and push_to_github=true: Optional GitHub Release created or updated.",
    "Final summary report of actions taken."
]

# --- Housekeeping ---
last_updated = "{{DATE}}" # Placeholder for update timestamp
template_schema_doc = ".ruru/templates/toml-md/23_workflow_readme.md" # Link to the template used
related_docs = [
    ".ruru/docs/planning/github-deeper-integration/PLAN-RELEASE-NOTES-WHITEPAPER.md",
    ".ruru/docs/planning/github-deeper-integration/PLAN-RELEASE-NOTES-MCP-WORKFLOW.md",
    ".ruru/docs/planning/github-deeper-integration/PLAN-RELEASE-NOTES-LOCAL-WORKFLOW.md",
    ".ruru/docs/planning/github-deeper-integration/PLAN-RELEASE-NOTES-SOURCE-OF-TRUTH.md",
    ".ruru/docs/planning/github-deeper-integration/PLAN-RELEASE-NOTES-TRIGGERS.md",
    ".roo/rules/07-git-commit-standard-simplified.md",
    ".ruru/templates/toml-md/13_release_notes.md" # Template for the generated local release notes file
]
+++

# Workflow: Generate Release Notes (Hybrid Approach)

## Overview

Generates release notes incrementally by analyzing Git history since the last release tag found in the `output_dir`. It automatically determines the next version based on semantic versioning (patch increment) and reads GitHub configuration from `.ruru/config/project.toml`.

This workflow supports two main actions:
*   **`add` (default):** Analyzes commits since the last tag, generates notes for *new* changes, and saves them as a timestamped Markdown file in the `[output_dir]/next/` directory.
*   **`finalize`:** Aggregates all incremental notes from the `[output_dir]/next/` directory into a single release notes file named after the determined next version (e.g., `[output_dir]/vX.Y.Z.md`). Optionally pushes these notes to a GitHub Release.

## Usage

This workflow can be run multiple times with the default `add` action as development progresses. When ready to release, run it once with `action="finalize"`.

*   **Adding Notes:** Initiate the workflow without the `action` parameter (or `action="add"`).
*   **Finalizing Release:** Initiate the workflow with `action="finalize"`. Set `push_to_github=true` to also create the GitHub release.

## Inputs

The workflow accepts the following inputs (defined in the TOML `[parameters]` section of the `00_start.md` step):

*   **`action`** (String, Optional, Default: `"add"`): Specifies the operation mode.
    *   `"add"`: Generate and save incremental notes for new commits since the last tag.
    *   `"finalize"`: Aggregate incremental notes, create the final version file, and optionally push to GitHub.
*   **`output_dir`** (String, Optional, Default: `.ruru/docs/release-notes/`): The base directory for release notes. Incremental notes are stored in a `next/` subdirectory within this path (e.g., `.ruru/docs/release-notes/next/`). Final notes are saved directly in this path (e.g., `.ruru/docs/release-notes/vX.Y.Z.md`).
*   **`push_to_github`** (Boolean, Optional, Default: `false`): If `true` and `action="finalize"`, the workflow will attempt to create or update a GitHub Release using the GitHub MCP server and details from `.ruru/config/project.toml`.
*   **`mark_as_draft`** (Boolean, Optional, Default: `true`): If pushing to GitHub during `finalize`, marks the release as a draft.
*   **`mark_as_prerelease`** (Boolean, Optional, Default: `false`): If pushing to GitHub during `finalize`, marks the release as a pre-release.

*(Note: `target_tag`, `previous_tag`, `github_owner`, `github_repo` are no longer direct inputs; they are determined automatically).*

## Outputs

The outputs depend on the `action` performed:

*   **If `action="add"`:**
    *   A new Markdown file saved in the `[output_dir]/next/` directory (e.g., `.ruru/docs/release-notes/next/update_20250430215500.md`) containing notes for commits since the last run or the last tag.
*   **If `action="finalize"`:**
    *   A final, aggregated Markdown file saved in `output_dir` named after the new version (e.g., `.ruru/docs/release-notes/v7.1.5.md`).
    *   If `push_to_github` was `true` and successful, a GitHub Release corresponding to the new version tag.
*   **Always:**
    *   A final report summarizing the outcome (path to created file(s), GitHub release status/link if applicable).