# Milestone System

## Purpose
Record milestones when:
- Major component completed
- Critical bug fixed
- Implementation changed
- Feature delivered

## Core Principles
- **Concise**: Every token counts
- **Factual**: Concrete details, not stories
- **Relevant**: Include essential only
- **Future-Focused**: What next devs need
- **Learning**: Document issues and solutions

## Handoff to Milestone
1. **Create Handoffs First**:
   - Write handoffs during development
   - Capture work completed in each session
   - Accumulate in handoffs/ directory
   - Use sequential numbering (1-setup.md, 2-entities.md)

2. **Distill to Milestones**:
   - Extract key info from handoffs
   - Identify patterns across handoffs
   - Consolidate repeated themes
   - Transform details into concise facts
   - Prioritize long-term value info

## Workflow
1. **Development with Handoffs**:
   - Document work in handoff files
   - Progress through sequential handoffs
   - Handoffs accumulate in handoffs/ directory

2. **At milestone**:
   - Create milestone folder with descriptive name
   - Use sequential numbers (1-feature, 2-api, etc.)
   - Distill handoffs into milestone docs
   - Move docs to milestone folder
   - Name reflects actual achievement

## Files
1. **0-milestone-summary.md**:
   - Date completed
   - Changes implemented
   - Decisions made and why
   - Discoveries found
   - Current system state

2. **0-lessons-learned.md**:
   - Problems encountered
   - Working solutions
   - Tools/libraries used
   - Edge cases identified
   - Reusable patterns

## Naming Convention
- System files: prefix with "0-" (0-milestone-summary.md)
- Handoffs: numbered without "0-" (1-setup.md)
- Milestone folders: numbered without "0-" (1-feature)

## Writing Style

### 0-milestone-summary.md
```
## Changes
- Data connector with batch processing
- Optimized query approach (30x faster)
- Cross-platform path handling

## Decisions
- Validation library update: 40% faster
- Nested env vars for configuration
- Default fallback for missing references

## Discoveries
- Duplicate entities: Item A has IDs 64, 125
- Relationship gaps: 246/252 entities connected
- Missing refs: IDs 53, 54 not in dataset
```

### 0-lessons-learned.md
```
## Config Library Migration

**Problem:** `Cannot import Settings from library`

**Solution:**
- Update dependency version
- Use new config pattern
- Update initialization with correct prefix

## Null Value Handling

**Problem:** `Invalid value in transformation`

**Solution:**
- Create sanitization function
- Replace invalid values with null
- Apply to all values before serialization
```

## Example
Before: 3 handoff files
```
handoffs/1-api-setup.md
handoffs/2-core-entities-implementation.md
handoffs/3-relationship-fixes.md
```

After:
```
handoffs/ (continues to accumulate new handoffs)

handoffs/1-core-entities/
  ├── 0-milestone-summary.md  # Decisions, discoveries
  └── 0-lessons-learned.md    # Patterns, fixes