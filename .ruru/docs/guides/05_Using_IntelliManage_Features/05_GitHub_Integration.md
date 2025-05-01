+++
# --- Basic Metadata ---
id = "KB-RC-IM-USAGE-GITHUB"
title = "Using IntelliManage: Setting Up & Using GitHub Integration"
status = "draft"
doc_version = "1.0" # Version of IntelliManage GitHub integration
content_version = 1.0
audience = ["users", "developers", "project_managers"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/10_guide_tutorial.README.md" # Using guide template schema
tags = ["roo-commander", "intellimanage", "guide", "tutorial", "commands", "github", "integration", "setup", "usage", "sync", "issues", "labels", "milestones"]
related_docs = [
    "../README.md", # Link to the KB README
    "../../../DOC-GITHUB-SPEC-001.md", # Link to the full GitHub spec
    "../../../DOC-UI-CMD-SPEC-001.md", # Link to command spec
    "../../../DOC-SETUP-GUIDE-001.md" # Link to general setup guide
    ]
difficulty = "intermediate" # Requires GitHub knowledge and PAT handling
estimated_time = "~15-20 minutes (excluding PAT creation)"
prerequisites = ["IntelliManage initialized for the target project", "Project is a Git repository", "GitHub account access", "Ability to create GitHub Personal Access Tokens (PATs)"]
learning_objectives = ["Enable GitHub integration for an IntelliManage project", "Configure repository details and authentication securely", "Understand how artifacts sync with GitHub Issues/Labels/Milestones", "Learn how to link commits/PRs to IntelliManage items", "Know how to check sync status and troubleshoot common issues"]
+++
> **🚧 Work in Progress:** The IntelliManage features described in this section are currently under development and not yet available for general use. This documentation reflects planned functionality.

# Using IntelliManage: Setting Up & Using GitHub Integration

## 1. Introduction / Goal 🎯

IntelliManage can integrate with GitHub to synchronize project artifacts (Tasks, Stories, Bugs, Epics) with GitHub Issues, Labels, and Milestones. This allows teams to manage work seamlessly across both platforms, providing visibility in GitHub while leveraging IntelliManage's features within the development environment.

This guide walks you through setting up the integration for a specific project and explains how to use its features effectively. For detailed technical specifications, refer to the **GitHub Integration Specification (`DOC-GITHUB-SPEC-001`)**.

## 2. Prerequisites Checklist ✅

*   [ ] The IntelliManage project you want to integrate is already initialized (e.g., `.ruru/projects/my-web-app/` exists with `project_config.toml`).
*   [ ] The project directory is initialized as a Git repository and ideally pushed to the target GitHub repository.
*   [ ] You have a GitHub Personal Access Token (PAT) with the necessary scopes (typically `repo` for accessing repository issues, labels, milestones). **Important:** Create a fine-grained PAT if possible, limiting its access. See [GitHub Docs on PATs](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
*   [ ] You have securely stored your GitHub PAT as an **environment variable** accessible by your development environment (e.g., in your shell profile like `.bashrc`, `.zshrc`, or system environment variables). **Do NOT paste the token directly into configuration files or chat.**

## 3. Step 1: Enabling Integration in `project_config.toml` ⚙️

Configuration is done on a per-project basis within its `project_config.toml` file. Use the `!pm config project` command.

*   **Action:** Set the required fields within the `[github_integration]` table.
*   **Command:**
    ```
    !pm config project <project_slug> --set github_integration.enable_sync=true --set github_integration.repo_owner="your-github-username-or-org" --set github_integration.repo_name="your-repo-name" --set github_integration.pat_env_var_name="YOUR_GITHUB_PAT_ENV_VAR_NAME"
    ```
*   **Explanation:**
    *   Replace `<project_slug>` with your project's slug (e.g., `frontend-app`).
    *   Replace `"your-github-username-or-org"` with the owner of the target GitHub repository.
    *   Replace `"your-repo-name"` with the name of the target GitHub repository.
    *   Replace `"YOUR_GITHUB_PAT_ENV_VAR_NAME"` with the **exact name** of the environment variable where you stored your GitHub PAT (e.g., `GITHUB_PAT`, `GH_TOKEN`).
    *   `enable_sync=true`: Turns the integration on for this project.
*   **Verification:** After running the command, you can check the contents of `.ruru/projects/<project_slug>/project_config.toml` to confirm the `[github_integration]` section is present and correct.

## 4. Step 2: Understanding Default Mappings 🗺️

By default, IntelliManage attempts to map artifacts and metadata as follows (these can often be customized - see `DOC-GITHUB-SPEC-001`):

*   **IntelliManage Tasks/Stories/Bugs** <-> **GitHub Issues**
*   **IntelliManage Epics** (default) <-> **GitHub Milestones**
*   **IntelliManage Status** <-> **GitHub Labels** (e.g., `PM:Status:ToDo`, `PM:Status:InProgress`)
*   **IntelliManage Type** <-> **GitHub Labels** (e.g., `PM:Type:Bug`, `PM:Type:Story`)
*   **IntelliManage Priority** <-> **GitHub Labels** (e.g., `PM:Priority:High`)
*   **IntelliManage Tags** <-> **GitHub Labels** (without prefix)

Labels prefixed with `PM:` (or your configured prefix) are generally managed by IntelliManage.

## 5. Step 3: Using the Integration 🔄

Once enabled, the integration works in several ways:

*   **Initial Sync:** The first time sync is enabled or triggered (`!pm sync github`), IntelliManage may attempt to create corresponding Issues/Milestones/Labels in GitHub for existing artifacts, and vice-versa (depending on configuration). This can take time for large projects.
*   **Automatic Suggestions (AI Driven):**
    *   When you create/update an artifact in IntelliManage, the AI may prompt you to confirm creating/updating the corresponding GitHub Issue/Milestone.
    *   When changes are detected on linked GitHub items (e.g., an Issue is closed, a label is added), the AI may suggest updating the corresponding IntelliManage artifact's status or metadata. **User confirmation is typically required.**
*   **Manual Sync Trigger:** You can force a synchronization check using:
    *   **Command:** `!pm sync github [--project <slug>]`
    *   This is useful if you suspect things are out of sync or after making bulk changes on either platform.
*   **Linking via Commits/PRs:**
    *   Include IntelliManage artifact IDs in your Git commit messages or Pull Request descriptions using keywords configured in `project_config.toml` (defaults include `Fixes`, `Closes`, `Refs`, etc.).
    *   **Example Commit Message:**
        ```
        feat: Implement user login endpoint

        Adds the POST /login endpoint as described in the spec.
        Connects to the auth service.

        Refs: TASK-101
        Fixes: BUG-042
        ```
    *   When these commits/PRs are pushed/merged, the Integration Layer detects the keywords and IDs.
    *   It updates the `related_commits` or `related_prs` field in the linked IntelliManage artifacts (TASK-101, BUG-042).
    *   If keywords like `Fixes` or `Closes` are used, the AI will likely suggest marking the corresponding artifact (BUG-042) as "Done".

## 6. Step 4: Checking Status & Troubleshooting ✅❓

*   **Check Status:** Use the status command to verify connectivity and last sync time.
    *   **Command:** `!pm status github [--project <slug>]`
*   **Common Issues:**
    *   **Authentication Errors:** Double-check your PAT is correct, stored in the *exact* environment variable name specified in `pat_env_var_name`, and has the necessary `repo` scope. Restart your IDE/terminal after setting environment variables.
    *   **Repo Not Found:** Verify `repo_owner` and `repo_name` are correct in `project_config.toml`.
    *   **Sync Conflicts:** If `conflict_resolution` is set to `"manual_flag"`, items with conflicting updates will be flagged. You'll need to manually review and update either the IntelliManage artifact or the GitHub item to resolve the conflict, then potentially run `!pm sync github` again.
    *   **Label/Milestone Errors:** Ensure labels/milestones exist in GitHub or that `create_missing_labels`/`create_missing_milestones` is enabled (if available/default). Check for typos in mappings if customized.

## 7. Conclusion ✅

Integrating IntelliManage with GitHub provides a powerful bridge between project planning/tracking and your code repository. By configuring the sync, understanding the mappings, and utilizing commit message linking, you can maintain consistency and improve visibility across both systems, streamlining your development workflow. Remember to handle your GitHub PAT securely using environment variables.