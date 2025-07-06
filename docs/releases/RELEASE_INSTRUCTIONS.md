# GitHub Release Instructions for v0.1.1

## Option 1: Automated Release (Recommended)

### Prerequisites
1. GitHub Personal Access Token with `repo` permissions
   - Go to: https://github.com/settings/tokens
   - Create a new token with `repo` scope

### Steps
1. Run the batch file:
   ```cmd
   create_release.bat
   ```
2. Enter your GitHub token when prompted
3. The script will create the release automatically

## Option 2: Manual Release

### Steps
1. Go to: https://github.com/blackms/AlphaPulse/releases
2. Click "Create a new release"
3. Select tag: `v0.1.1`
4. Set release title: `v0.1.1`
5. Copy and paste the following description:

```
## Release v0.1.1

### Changed
- Refactored backtester to use new `alpha_pulse/agents` module instead of deprecated `src/agents`.
- Removed the old `src/agents` directory and all legacy agent code.
- Confirmed all documentation and diagrams are up-to-date after agents module cleanup.

### Technical Details
This release includes a major code cleanup that:
- Migrates the backtesting system from the old `src/agents` implementation to the new `alpha_pulse/agents` module
- Removes legacy agent code that was no longer maintained
- Ensures all documentation and architecture diagrams remain accurate
- Improves code maintainability and reduces technical debt

### Breaking Changes
None - this is a refactoring release that maintains backward compatibility.
```

6. Click "Publish release"

## Option 3: Using GitHub CLI

If you have GitHub CLI installed:
```bash
gh release create v0.1.1 --title "v0.1.1" --notes-file RELEASE_NOTES.md
```

## Release Summary
- **Tag:** v0.1.1
- **Type:** Patch release (refactoring)
- **Breaking Changes:** None
- **Main Changes:** Code cleanup and agent module migration 