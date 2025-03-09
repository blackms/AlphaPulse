# RooCommander Component-Based System Prompt

This directory contains the component-based system prompt for RooCommander. The system prompt is assembled from individual component files in the `components/` directory.

## Directory Structure

- `components/` - Individual component files for the system prompt
  - `header.txt` - Introduction and role definition
  - `assessment-framework.txt` - "What-How-What" methodology framework
  - `technology-identification.txt` - Technology identification process
  - `reference-documentation.txt` - Reference documentation management
  - `team-structure.txt` - Team structure analysis
  - `configuration-persistence.txt` - Configuration persistence system
  - `mode-selection.txt` - Mode selection logic
  - `safety-rules.txt` - Safety protocols

- `system-prompt-commander` - Complete assembled system prompt

## Component Assembly

The components are assembled in the following order:

1. `header.txt`
2. (Tool essentials - from system)
3. `assessment-framework.txt`
4. `technology-identification.txt`
5. `team-structure.txt`
6. `reference-documentation.txt`
7. `configuration-persistence.txt`
8. `mode-selection.txt`
9. `safety-rules.txt`

## Using the System Prompt

The assembled system prompt is used in the following contexts:

1. Core RooCommander mode definition
2. Custom mode configuration for projects
3. Technology-specific guidance based on reference documentation

## Component Updates

When updating components:

1. Edit individual component files
2. Reassemble the system prompt
3. Test with representative user interactions

## Reference Documentation Integration

The system prompt is designed to work with the hierarchical reference documentation structure in `custom-modes-pool/reference-docs/`. It leverages `ask_perplexity` MCP tool when available to create missing reference documentation.

## Configuration Persistence

The system uses `.rooconfig.md` for configuration persistence, tracking project profiles, team structures, and reference documentation paths.