+++
# --- Basic Metadata ---
id = "KB-RC-SETUP-INSTALL"
title = "Getting Started: Installing Roo Commander"
status = "draft"
difficulty = "beginner"
estimated_time = "~5-10 minutes"
target_audience = ["users", "developers"]
prerequisites = ["Visual Studio Code (VS Code) installed", "Roo Code VS Code extension installed and enabled"]
learning_objectives = ["Install the Roo Commander framework into a VS Code workspace using a release build", "Understand the core files and directories created during installation", "Know how to activate the framework after installation"]
template_schema_doc = ".ruru/templates/toml-md/10_guide_tutorial.README.md"
tags = ["roo-commander", "getting-started", "installation", "setup", "guide", "tutorial", "release", "workspace", "configuration"]
related_docs = [
    "../README.md", # Link to the KB README
    "../../../README.md", # Link to main project README
    "../../../DOC-SETUP-GUIDE-001.md" # Link to IntelliManage setup guide
    ]
+++

# Getting Started: Installing Roo Commander

## 1. Introduction / Goal 🎯

This guide explains how to install the Roo Commander framework into your VS Code workspace using the pre-built release packages. Roo Commander provides a structured set of AI modes, rules, workflows, and templates designed to enhance your development process with AI assistance and integrated project management via IntelliManage.

The goal is to add the necessary configuration files and directories to your project so that the Roo Code extension can load and utilize the Roo Commander modes and workflows.

## 2. Prerequisites Checklist ✅

*   [ ] Visual Studio Code is installed.
*   [ ] The [Roo Code VS Code extension](https://marketplace.visualstudio.com/items?itemName=RooCode.roo-code) is installed and enabled.

## 3. Step 1: Download the Latest Release Build 💾

*   **Action:** Obtain the latest pre-packaged release build.
*   **Procedure:**
    1.  Navigate to the Roo Commander GitHub repository's `.builds/` directory: [Roo Commander Builds Folder](https://github.com/jezweb/roo-commander/tree/main/.builds)
    2.  Identify the latest versioned `.zip` file (e.g., `roo-commander-vX.Y.Z-Codename.zip`). Look for the highest version number.
    3.  Download this `.zip` file to your local machine.

## 4. Step 2: Extract to Workspace Root ↔️

*   **Action:** Extract the contents of the downloaded `.zip` file directly into the root directory of your VS Code project workspace.
*   **Procedure:**
    1.  Open your project folder in VS Code or your system's file explorer. The "workspace root" is the top-level folder containing your project's main files and potentially your `.git` directory.
    2.  Use your system's unzip tool to extract the **contents** of the downloaded zip file directly into this root folder.
    3.  **Important:** Ensure the extraction process merges the contents into your existing workspace root, creating the hidden `.roo/` and `.ruru/` directories at the top level. Do **not** extract it into a new subfolder named after the zip file.
    4.  **Overwrite Confirmation:** Your unzip tool might ask if you want to overwrite existing files (especially if you are upgrading Roo Commander). Choose **Yes/Merge/Overwrite** for all files to ensure the latest configuration is applied.

## 5. Step 3: Verify Files and Directories 📂

*   **Action:** Check that the core Roo Commander directories and files have been added to your workspace root.
*   **Procedure:**
    1.  Ensure your file explorer (in VS Code or your OS) is set to show hidden files/folders (those starting with a dot `.`).
    2.  Verify the presence of the following key items in your workspace root:
        *   `.roo/` (Directory containing core rules)
        *   `.ruru/` (Directory containing modes, templates, workflows, processes, docs, and the `projects` directory for IntelliManage)
        *   `.roomodes` (File listing custom modes for Roo Code to load)
        *   `(Other helper scripts like build_*.js may also be present)`
*   **Note:** The extraction adds many files within `.roo/` and `.ruru/`. The presence of these top-level items indicates a successful extraction.

## 6. Step 4: Reload VS Code Window 🔄

*   **Action:** Reload your VS Code window to allow the Roo Code extension to detect the newly added modes and configurations.
*   **Procedure:**
    1.  Open the Command Palette in VS Code (`Ctrl+Shift+P` or `Cmd+Shift+P`).
    2.  Type "Reload Window" and select the command `Developer: Reload Window`.
    3.  VS Code will restart, loading the new Roo Commander configuration.

## 7. Verification / Next Steps ✅

*   **Verification:** After reloading, open the Roo Code chat interface. Click on the mode selection dropdown (usually showing "code" initially). You should now see "👑 Roo Commander" and potentially other modes like "session-manager" listed. Selecting "👑 Roo Commander" should activate it.
*   **Next Steps:**
    *   You can now start interacting with Roo Commander! Try selecting the "👑 Roo Commander" mode and saying hello.
    *   To set up project management features, you will need to initialize at least one project using the IntelliManage setup commands. Refer to the **IntelliManage: Setup & Configuration Guide (`DOC-SETUP-GUIDE-001.md`)** for detailed instructions on using `!pm init project ...`.

## 8. Troubleshooting ❓

*   **Modes Not Appearing:** Ensure you extracted the zip contents directly into the workspace *root*. Double-check that the `.roo/`, `.ruru/`, and `.roomodes` items are at the top level, not inside another folder. Reload the VS Code window again.
*   **File Conflicts:** If you had a previous manual setup, overwriting files during extraction is generally expected and correct.

---

You have now successfully installed the Roo Commander framework into your workspace!