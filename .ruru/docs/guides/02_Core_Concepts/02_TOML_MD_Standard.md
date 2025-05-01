# --- Basic Metadata ---
id = "KB-RC-CONCEPTS-TOMLMD"
title = "Core Concept: TOML+Markdown (TOML MD) Standard"
status = "draft"
doc_version = "1.0" # Version of the standard being described
content_version = 1.0
audience = ["users", "developers", "architects", "contributors", "ai_modes"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/09_documentation.README.md"
tags = ["roo-commander", "intellimanage", "core-concept", "toml", "markdown", "format", "standard", "metadata", "schema"]
related_docs = [
    "../README.md", # Link to the KB README
    ".roo/rules/01-standard-toml-md-format.md", # Link to the formal rule
    ".roo/templates/toml-md/README.md", # Link to the template index
    "01_File_System_Structure.md",
    ".ruru/planning/IntelliManage/03-DOC-SCHEMA-001.md" # Link to IntelliManage Schemas
    ]
+++

# Core Concept: TOML+Markdown (TOML MD) Standard

## 1. Introduction / Overview 🎯

A fundamental aspect of Roo Commander and the IntelliManage framework is the use of a specific file format called **TOML+Markdown (TOML MD)** for storing project artifacts like tasks, epics, decisions (ADRs), context sources, and even some configuration files (like mode definitions).

This standard combines the strengths of two formats:
*   **TOML (Tom's Obvious, Minimal Language):** For structured, machine-readable metadata.
*   **Markdown:** For rich, human-readable content.

This document explains the structure, rationale, and usage of this standard format.

## 2. Why TOML+Markdown? 🤔

Using this combined format provides several key advantages:

*   **Machine Readability:** The TOML frontmatter allows AI agents and automated tools to reliably parse essential metadata (like `id`, `status`, `type`, `tags`, `links`) without needing to understand the nuances of the free-form Markdown content.
*   **Human Readability:** The main content is written in standard Markdown, making it easy for team members to read, write, and collaborate on artifact details, descriptions, criteria, and notes.
*   **Structured & Flexible:** It offers the best of both worlds – structured data where needed (TOML) and flexible prose where needed (Markdown).
*   **Version Control Friendly:** Both TOML and Markdown are text-based, making them ideal for version control systems like Git. Changes are easily trackable and diffable.
*   **Consistency:** Enforces a consistent way to store metadata across different artifact types.

## 3. Standard Structure 🧱

Every file adhering to the TOML+MD standard **MUST** follow this structure:

```markdown
+++
# --- TOML Frontmatter Block ---
# Keys and values defined here MUST follow valid TOML syntax.
# The specific fields depend on the artifact type (see schemas).
id = "ARTIFACT-ID-123"
title = "Example Artifact Title"
status = "🟡 To Do"
tags = ["example", "standard"]
# ... other metadata fields ...
+++

# Markdown Body Starts Here

## Section Heading

This is the main content area where you use standard Markdown formatting.

*   Lists
*   **Bold Text**
*   `Inline Code`

```python
# Code Blocks
print("Hello, TOML+MD!")
```

## Acceptance Criteria ✅

- [ ] Checkbox item 1
- [ ] Checkbox item 2

```mermaid
graph TD
    A --> B;
``````

*   **`+++` Delimiters:** The TOML metadata block **MUST** begin and end with exactly three plus signs (`+++`) on their own lines.
*   **TOML Block:** Contains key-value pairs defining the artifact's metadata. Strict TOML syntax is required. Refer to specific schemas (like `DOC-SCHEMA-001` or template READMEs) for required/optional fields for each artifact type.
*   **Markdown Body:** Standard Markdown content follows the closing `+++` delimiter. Use headings, lists, code blocks, checklists, etc., as needed to structure the human-readable information.

## 4. Key Considerations ✅

*   **Syntax:** Be strict with TOML syntax within the `+++` block (e.g., strings in quotes, arrays in brackets). Invalid TOML will cause parsing errors.
*   **Schemas:** Adhere to the specific TOML schema defined for the artifact type you are creating or editing. Consult the relevant template README (`.ruru/templates/toml-md/[template_name].README.md`) or the main schema document (`DOC-SCHEMA-001`).
*   **Templates:** Always use the provided templates (`.ruru/templates/toml-md/`) as a starting point when creating new artifacts to ensure the correct structure and required fields are present.
*   **Rule:** The mandatory use of this format is defined in the workspace rule: `.roo/rules/01-standard-toml-md-format.md`.

## 5. Conclusion

The TOML+MD standard provides a robust and flexible format for managing project artifacts within Roo Commander and IntelliManage. It balances the need for structured, machine-readable data with clear, human-readable content, supporting both automated processes and effective team collaboration.