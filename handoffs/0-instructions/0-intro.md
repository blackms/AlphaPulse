# Handoff and Milestone System

## Core Concept

This system enables efficient knowledge transfer between LLM sessions through:
1. Handoffs: Sequential session reports capturing completed work
2. Milestones: Consolidated knowledge from multiple handoffs

## Documents Structure

**Handoffs**:
- Numbered sequentially (1-setup.md, 2-implementation.md)
- Located in handoffs/ root directory
- Contain specific completed work details

**Milestones**:
- Stored in numbered folders (1-feature-complete/)
- Consolidate multiple handoff documents
- Summarize achievements and lessons learned

## Creation Triggers

**Create handoff documents when**:
- Completing a significant project segment
- Context becomes ~30% irrelevant to current task
- After 10+ conversation exchanges
- During debugging sessions exceeding 5 exchanges without resolution

**Create milestone documents when**:
- Major feature/component implementation complete
- Project phase completed
- 3-5 handoffs accumulated since last milestone
- Critical problem solved with valuable lessons
- Project reaches stable/deployable state

## Context Assessment Process

Before each major response:
1. Review context window contents:
   - Most relevant: current task, recent files, active discussions
   - Moderately relevant: background information, earlier work
   - Low relevance: initial setup, tangential discussions
2. Determine if handoff needed based on assessment

## Implementation

Add to project's `.clinerules` file:

```
Assess context relevance after each substantive exchange. Create handoff documents in the handoffs/ directory when context becomes diluted with irrelevant information, after completing discrete project segments, or during prolonged debugging sessions. Create milestone documents in handoffs/ subdirectories when completing major features or after 3-5 related handoffs accumulate. A fresh LLM session with focused context often solves problems that an overloaded session cannot.
```

## Compatibility

Optimized for Claude 3.7 Sonnet with thinking enabled (2k-8k reasoning capacity).

## Process Flow

1. Create sequential handoff documents during development
2. Generate milestone when threshold conditions met
3. Create milestone folder and move relevant handoff files
4. Create summary documentation in milestone folder

## Reference Documentation

- [1-handoff-instructions.md](./1-handoff-instructions.md): Handoff document format
- [2-milestone-instructions.md](./2-milestone-instructions.md): Milestone process
- [3-milestone-scripts.md](./3-milestone-scripts.md): Automation scripts