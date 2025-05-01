+++
# --- Basic Metadata ---
id = "KB-RC-COMMUNITY-CONTRIBUTING"
title = "Community & Contributing: Contribution Guide"
status = "draft"
doc_version = "1.0" # Version of this guide
content_version = 1.0
audience = ["developers", "users", "contributors"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/10_guide_tutorial.README.md" # Using guide template schema
tags = ["roo-commander", "community", "contributing", "development", "guide", "github", "issues", "pull-requests", "modes", "rules", "documentation"]
related_docs = [
    "../README.md", # Link to the KB README
    "01_Joining_the_Community.md",
    "../../06_Advanced_Usage_Customization/01_Custom_Modes.md", # Guide on creating modes
    "../../06_Advanced_Usage_Customization/02_Custom_Instructions_Rules.md" # Guide on creating rules
    ]
difficulty = "intermediate"
estimated_time = "~15-20 minutes"
prerequisites = ["Familiarity with Git and GitHub", "Basic understanding of Roo Commander architecture"]
learning_objectives = ["Understand the different ways to contribute to Roo Commander", "Learn the process for reporting bugs and suggesting features", "Know how to contribute documentation, rules, and new modes", "Understand the Pull Request workflow"]
+++

# Community & Contributing: Contribution Guide

## 1. Introduction / Goal 🎯

Thank you for your interest in contributing to the Roo Commander framework! Contributions from the community are vital for improving the system, adding new capabilities, and ensuring it meets the diverse needs of its users.

This guide outlines the various ways you can contribute, from reporting issues and improving documentation to developing new modes and rules.

**Goal:** To provide clear guidelines for contributing to the Roo Commander project effectively.

## 2. Ways to Contribute 🤝

There are many ways to contribute, including:

*   Reporting Bugs
*   Suggesting Features or Enhancements
*   Improving Documentation (like this Knowledge Base!)
*   Adding or Refining Rules
*   Developing New Modes
*   Submitting Code Fixes or Improvements (via Pull Requests)

## 3. Reporting Bugs 🐞

If you encounter a reproducible error or unexpected behavior that you believe is a bug:

1.  **Search Existing Issues:** Check the [GitHub Issues tab](https://github.com/jezweb/roo-commander/issues) on the main repository to see if the bug has already been reported.
2.  **Gather Information:** Collect details needed to reproduce the bug:
    *   Roo Commander version (if known).
    *   Steps to reproduce the issue reliably.
    *   The exact prompt(s) you used.
    *   The full error message(s) observed (from chat and the Roo Code Output channel).
    *   Your operating system and VS Code version.
    *   Any relevant configuration details (e.g., modified rules, custom modes involved).
3.  **Create a New Issue:** If the bug hasn't been reported, create a new issue on GitHub.
    *   Use a clear and descriptive title.
    *   Provide the gathered information in the issue description.
    *   Use appropriate labels (e.g., `bug`, `mode:dev-react`, `area:intellimanage`) if possible.

## 4. Suggesting Features & Enhancements ✨

Have an idea for a new feature, a new mode, or an improvement to an existing one?

1.  **Check Existing Issues/Discussions:** Search GitHub Issues and Discussions (if enabled) to see if a similar idea has already been proposed or discussed.
2.  **Create a New Issue/Discussion:**
    *   Use the "Feature request" issue template if available, or create a standard issue/discussion post.
    *   Clearly describe the proposed feature or enhancement.
    *   Explain the problem it solves or the benefit it provides ("Why?").
    *   Outline potential implementation ideas or user workflows if applicable.
    *   Use appropriate labels (e.g., `enhancement`, `feature-request`, `new-mode`).

## 5. Contributing Documentation ✍️

Improvements to documentation (READMEs, KB guides, mode documentation, rule explanations) are always welcome!

1.  **Identify Area:** Find documentation that is unclear, incomplete, incorrect, or missing.
2.  **Fork & Branch:** Fork the main repository on GitHub and create a new branch for your changes (e.g., `docs/update-kb-linking-guide`).
3.  **Make Changes:** Edit the relevant `.md` files, adhering to the TOML+MD standard where applicable. Ensure clarity, accuracy, and consistency.
4.  **Commit & Push:** Commit your changes with clear messages and push the branch to your fork.
5.  **Create Pull Request:** Open a Pull Request (PR) from your branch to the `main` branch of the upstream repository.
    *   Provide a clear description of the changes made in the PR.
    *   Link to any relevant issues if applicable.

## 6. Contributing Rules 📜

If you have identified a need for a new workspace-wide rule or an improvement to an existing one:

1.  **Discuss (Optional but Recommended):** Consider discussing the proposed rule change in a GitHub Issue or on Discord first, especially for significant changes to core rules.
2.  **Fork & Branch:** Follow the standard Git workflow (Fork, Branch).
3.  **Create/Edit Rule:** Create or modify the rule file(s) in the appropriate `.roo/rules/` or `.roo/rules-[mode_slug]/` directory, following the TOML+MD format and best practices (see `06_Advanced_Usage_Customization/02_Custom_Instructions_Rules.md`).
4.  **Test:** Test the impact of the rule change locally by running relevant modes and scenarios.
5.  **Commit, Push, PR:** Follow the standard Pull Request process. Explain the rationale for the new/modified rule in the PR description.

## 7. Contributing New Modes 🤖

Creating and contributing new specialist modes is a great way to extend Roo Commander's capabilities.

1.  **Discuss (Recommended):** Propose the new mode idea in a GitHub Issue or on Discord to gather feedback and avoid duplicating effort.
2.  **Develop Locally:** Follow the steps outlined in the Custom Modes guide (`06_Advanced_Usage_Customization/01_Custom_Modes.md`) to create the mode directory, `.mode.md` file, mode-specific rules, and initial KB content within your local workspace fork.
3.  **Test Thoroughly:** Test your new mode extensively with various inputs and scenarios. Ensure it interacts correctly with coordinators and other modes.
4.  **Add Documentation:** Ensure the Markdown body of your `.mode.md` file is well-documented (Description, Capabilities, Workflow, etc.). Add a `README.md` to the mode's `kb/` directory.
5.  **Commit, Push, PR:**
    *   Add the new mode files (`.ruru/modes/[your-slug]/...`, `.roo/rules-[your-slug]/...`) to Git.
    *   **Do NOT commit the `.roomodes` file directly.** The build script handles this.
    *   Submit a Pull Request containing the new mode's files.
    *   Clearly explain the mode's purpose, capabilities, and how to use it in the PR description.

## 8. General Contribution Workflow (Pull Requests) 🔄

*   **Fork:** Create a fork of the `jezweb/roo-commander` repository on GitHub.
*   **Branch:** Create a new branch in your fork for your specific contribution (e.g., `fix/typo-in-readme`, `feat/add-rust-mode`).
*   **Code/Document:** Make your changes locally in your branch.
*   **Test:** Test your changes thoroughly.
*   **Commit:** Use clear and concise commit messages, potentially following conventional commit standards if adopted by the project.
*   **Push:** Push your branch to your fork on GitHub.
*   **Pull Request (PR):** Open a PR from your branch to the `main` branch of the `jezweb/roo-commander` repository.
    *   Fill out the PR template clearly, explaining the "what" and "why" of your changes.
    *   Link to any relevant GitHub Issues.
    *   Be prepared to discuss your changes and make adjustments based on reviewer feedback.

## 9. Conclusion ✅

Your contributions, big or small, help make Roo Commander a better tool for everyone. Whether it's reporting a bug, improving documentation, or developing a new mode, your efforts are appreciated. Please follow these guidelines to ensure a smooth contribution process. Don't hesitate to ask questions in the community channels if you're unsure about anything!