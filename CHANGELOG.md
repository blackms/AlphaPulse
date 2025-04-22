## v7.0.5 (Wallaby) - 2025-04-21

### Added
*   **Enhanced Command Compatibility:** Introduced a new system rule (`.roo/rules/05-os-aware-commands.md`) ensuring that CLI commands generated via the `execute_command` tool are automatically tailored for the user's specific operating system (Linux, macOS, Windows), improving cross-platform reliability.
*   **New Prime Meta-Development Suite:** Added three specialized 'Prime' modes (`prime`, `prime-dev`, `prime-txt`) and their corresponding rules (`.roo/rules-prime*`). This suite provides a streamlined and controlled workflow specifically designed for modifying Roo Commander's own configuration files (modes, rules, KBs), enhancing maintainability and safety during meta-development.