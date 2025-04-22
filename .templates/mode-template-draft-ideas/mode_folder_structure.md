# Roo Mode Folder Structure (v7)

This document defines the standard folder structure for organizing Roo Code modes within the `v7.0/modes/` directory. This structure is based on the v7 Mode Hierarchy and is designed for clarity, scalability, and the ability to store mode-specific resources.

## Structure Overview

The core idea is to use nested folders that reflect the mode hierarchy and departmental organization. Each mode resides within its own dedicated folder.

```
v7.0/
└── modes/
    ├── [LEVEL_PREFIX]-[level_name]/  # e.g., 00x-executive, 01x-director
    │   └── [department_name]/        # Optional: Only for lead/worker levels
    │       └── [mode_slug]/          # Folder for the specific mode
    │           ├── [level_prefix]-[department_shortcode]-[mode_slug].mode.md  # The main Markdown definition file (all lowercase, hyphenated)
    │           └── resources/        # Optional: Subdirectory for mode-specific files
    │               └── ...
    └── ...
```

## Detailed Structure Breakdown

Based on the v7 Hierarchy (`00x-executive`, `01x-director`, `02x-lead`, `03x-worker`, `04x-assistant`):

*   **Level 0: Executive (`00x-executive`)**
    *   Path: `v7.0/modes/00x-executive/[mode_slug]/[level_prefix]-[department_shortcode]-[mode_slug].mode.md`
    *   Example: `v7.0/modes/00x-executive/roo-commander/000-exec-roo-commander.mode.md`

*   **Level 1: Director (`01x-director`)**
    *   Path: `v7.0/modes/01x-director/[mode_slug]/[level_prefix]-[department_shortcode]-[mode_slug].mode.md`
    *   Example: `v7.0/modes/01x-director/project-manager/010-dir-project-manager.mode.md`

*   **Level 2: Lead (`02x-lead`)**
    *   Grouped by department.
    *   Path: `v7.0/modes/02x-lead/[department_name]/[mode_slug]/[level_prefix]-[department_shortcode]-[mode_slug].mode.md`
    *   Example: `v7.0/modes/02x-lead/frontend/frontend-lead/020-lead-fe-frontend-lead.mode.md`

*   **Level 3: Worker (`03x-worker`)**
    *   Grouped by department using numeric prefixes.
    *   Path: `v7.0/modes/03x-worker/[department_prefix]-[department_name]/[mode_slug]/[level_prefix]-[department_shortcode]-[mode_slug].mode.md`
    *   Example (Frontend): `v7.0/modes/03x-worker/031-frontend/react-specialist/031-work-fe-react-specialist.mode.md`
    *   Example (Cross-Functional): `v7.0/modes/03x-worker/039-cross-functional/git-manager/039-work-xf-git-manager.mode.md`

*   **Level 4: Assistant (`04x-assistant`)**
    *   Path: `v7.0/modes/04x-assistant/[mode_slug]/[level_prefix]-[department_shortcode]-[mode_slug].mode.md`
    *   Example: `v7.0/modes/04x-assistant/context-resolver/040-asst-context-resolver.mode.md`

## Key Principles

*   **Hierarchy Mapping:** The top-level folders directly map to the defined hierarchy levels using numeric prefixes.
*   **Departmentalization:** Leads and Workers are further organized into departmental subfolders for clarity.
*   **Mode Isolation:** Each mode has its own folder (named by slug), allowing for self-contained definitions and resources.
*   **Primary Definition:** The main definition file within each mode's folder MUST follow the convention: `[level_prefix]-[department_shortcode]-[mode_slug].mode.md` (all lowercase, hyphenated).
*   **Resources:** Optional mode-specific files (e.g., context documents, examples, templates) can be placed in a `resources/` subdirectory within the mode's folder.

This structure provides a clear, organized, and extensible way to manage the growing collection of Roo modes.