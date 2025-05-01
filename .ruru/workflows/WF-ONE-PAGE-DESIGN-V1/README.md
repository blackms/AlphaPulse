+++
# --- Workflow Metadata ---
id = "WF-ONE-PAGE-DESIGN-V1"
title = "Workflow: One Page Web Design"
description = """
A workflow for rapidly generating a single-page website design using the `design-one-shot` mode.
It takes a user request detailing the desired page and produces the corresponding HTML, CSS, and minimal JavaScript code.
"""
version = "1.0.0"
status = "Draft"
tags = ["workflow", "design", "one-page", "web", "frontend", "design-one-shot"]

# --- Execution Control ---
entry_point = "00_start.md"

# --- Interface ---
inputs = [
    "User request detailing the desired one-page website (e.g., purpose, style, content sections, target audience).",
    "output_directory: (Optional) Specify a directory path for saving generated files. Defaults to `.ruru/artifacts/one-page-design/[timestamp]/` if not provided.",
]
outputs = [
    "HTML, CSS, and minimal JS files for the one-page website.",
    "Path to the generated files.",
    "Summary of the design process."
]

# --- Housekeeping ---
last_updated = "{{DATE}}"
template_schema_doc = ".ruru/templates/toml-md/23_workflow_readme.md"
related_docs = [".ruru/modes/design-one-shot/"] # Reference the primary mode used
+++

# Workflow: One Page Web Design

## Overview

A workflow for rapidly generating a single-page website design using the `design-one-shot` mode.
It takes a user request detailing the desired page and produces the corresponding HTML, CSS, and minimal JavaScript code.

## Usage

Initiate this workflow when a user requests a quick, single-page website design. The workflow delegates the core design task to the `design-one-shot` specialist mode.

1.  The workflow starts by confirming the user's request.
2.  It delegates the design generation to the `design-one-shot` mode.
3.  It receives the generated files (HTML/CSS/JS).
4.  (Optional) It can include a review step.
5.  It presents the final file paths and a summary to the user.

## Inputs

*   **User Request:** A detailed description of the desired one-page website. This should include:
    *   Purpose of the page (e.g., portfolio, landing page, event announcement).
    *   Desired style or aesthetic (e.g., modern, minimalist, playful).
    *   Key content sections (e.g., hero, about, services, contact).
    *   Target audience.
    *   Any specific elements or features to include.
*   **(Optional) Output Directory:** A relative path where the generated files should be saved. If omitted, a default timestamped directory under `.ruru/artifacts/one-page-design/` will be used.

## Outputs

*   **Website Files:** Complete HTML, CSS, and potentially minimal JavaScript files constituting the single-page design.
*   **File Paths:** The relative paths within the workspace where the generated files have been saved.
*   **Summary:** A brief confirmation that the design has been generated.