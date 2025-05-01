+++
# --- Workflow Metadata ---
id = "WF-CREATE-ROO-CMD-BUILD-V1" # (String, Required) Unique identifier
title = "Workflow: Create Roo Commander Build" # (String, Required)
description = """
(String, Required) Defines the full process for creating a Roo Commander build,
including running build scripts, verifying artifacts, packaging, generating release notes,
updating documentation (README, CHANGELOG), committing changes, tagging the release,
and optionally creating a GitHub release.
"""
version = "1.0.0" # (String, Required) Semantic version for the workflow definition.
status = "Draft" # (String, Required) Current status: "Draft", "Active", "Deprecated", "Experimental".
tags = ["workflow", "build", "roo-commander", "release", "documentation", "git", "github"] # (Array of Strings, Optional) Keywords for search/categorization.

# --- Execution Control ---
entry_point = "00_start.md" # (String, Required) Filename of the first step to execute.

# --- Interface ---
inputs = [ # (Array of Strings, Optional) Describe overall inputs needed to start the workflow.
    "build_parameters: Object containing details like version, platform, flags, create_github_release (boolean).",
]
outputs = [ # (Array of Strings, Optional) Describe the expected final artifacts or outcomes.
    "workflow_result: Summary object containing overall status and paths/URLs to artifacts, logs, release notes, commit, and GitHub release.",
]

# --- Housekeeping ---
owner = "lead-devops" # (String, Optional) Added owner field
maintainer = "lead-devops" # (String, Optional) Added maintainer field
last_updated = "{{DATE}}" # (String, Required) Date of last modification. Use placeholder.
template_schema_doc = ".ruru/templates/toml-md/23_workflow_readme.md" # (String, Required) Link to this template definition.
related_docs = [] # (Array of Strings, Optional) Links to related rules, KBs, ADRs. # Consider adding links later
+++

# Workflow: Create Roo Commander Build

## Overview

Defines the full process for creating a Roo Commander build, including running build scripts,
verifying artifacts, packaging, generating release notes, updating documentation (README, CHANGELOG),
committing changes, tagging the release, and optionally creating a GitHub release.

## Workflow Diagram

```mermaid
graph TD
    A[00_start] --> B(01_validate_params);
    B -- Success --> C(02_setup_environment);
    C -- Success --> D(03a_run_main_build);
    D -- Success --> D2(03b_run_kilocode_build);
    D2 -- Success --> D3(03c_run_collection_builds);
    D3 -- Success --> E(04_verify_artifacts);
    E -- Success --> F(05_package_artifacts);
    F -- Success --> G(06_determine_prev_tag);
    G -- Success --> H(07_query_git_history);
    H -- Success --> I(08_generate_release_notes);
    I -- Success --> J(09_save_release_notes);
    J -- Success --> K(10_update_readme);
    K -- Success --> L(11_update_changelog);
    L -- Success --> M(12_commit_docs);
    M -- Success --> N(13_push_tag);
    N -- Success --> O(14_create_github_release);
    O -- Success / Skipped --> P(99_finish);

    A -- Error --> EE_Start(EE_handle_start_error);
    B -- Error --> EE_Val(EE_handle_validation_error);
    C -- Error --> EE_Env(EE_handle_env_error);
    D -- Error --> EE_Build(EE_handle_build_error);
    D2 -- Error --> EE_Build;
    D3 -- Error --> EE_Build;
    E -- Error --> EE_Verify(EE_handle_verify_error);
    F -- Error --> EE_Pkg(EE_handle_packaging_error);
    G -- Error --> EE_GitTag(EE_handle_git_tag_error);
    H -- Error --> EE_GitQuery(EE_handle_git_query_error);
    I -- Error --> EE_NotesGen(EE_handle_notes_error);
    J -- Error --> EE_NotesSave(EE_handle_notes_save_error);
    K -- Error --> EE_Readme(EE_handle_readme_update_error);
    L -- Error --> EE_Changelog(EE_handle_changelog_update_error);
    M -- Error --> EE_GitCommit(EE_handle_git_commit_error);
    N -- Error --> EE_GitTagPush(EE_handle_git_tag_push_error);
    O -- Error --> EE_GHRelease(EE_handle_github_release_error);
    P -- Error --> EE_Fin(EE_handle_finish_error);

    EE_Start --> Z((End Failure));
    EE_Val --> Z;
    EE_Env --> Z;
    EE_Build --> Z;
    EE_Verify --> Z;
    EE_Pkg --> Z;
    EE_GitTag --> Z;
    EE_GitQuery --> Z;
    EE_NotesGen --> Z;
    EE_NotesSave --> Z;
    EE_Readme --> Z;
    EE_Changelog --> Z;
    EE_GitCommit --> Z;
    EE_GitTagPush --> Z;
    EE_GHRelease --> Z;
    EE_Fin --> Z;
    P -- Success --> Y((End Success));

    classDef errorNode fill:#f9f,stroke:#333,stroke-width:2px;
    class EE_Start,EE_Val,EE_Env,EE_Build,EE_Verify,EE_Pkg,EE_GitTag,EE_GitQuery,EE_NotesGen,EE_NotesSave,EE_Readme,EE_Changelog,EE_GitCommit,EE_GitTagPush,EE_GHRelease,EE_Fin errorNode;
```

---
This workflow orchestrates the necessary actions to compile, package, document, tag, and optionally release a new version of Roo Commander.

## Usage

This workflow is typically initiated by a coordinator or release manager when a new build of Roo Commander is required. Provide the necessary build parameters as input.

## Inputs

*   **Build Parameters:** An object containing:
    *   `version`: The semantic version string for the build (e.g., "v1.2.3").
    *   `platform`: Target platform (optional, defaults might apply).
    *   `build_flags`: Any specific flags for the build scripts (optional).
    *   `create_github_release`: Boolean flag (default `false`) indicating whether to create a GitHub release.

## Outputs

*   **Workflow Result:** A summary object containing:
    *   `overall_status`: 'Success' or 'Failure'.
    *   Status flags for individual critical steps (build, verify, package, commit, tag push).
    *   `github_release_status`: 'Success', 'Failure', or 'Skipped'.
    *   `artifact_path`: Path to the final packaged build artifact.
    *   `release_notes_path`: Path to the generated release notes file.
    *   `commit_hash`: SHA hash of the documentation commit.
    *   `github_release_url`: URL of the GitHub release (if created).