# Shift Change: Handoff Guidelines

## Purpose
Shift change report - tell next person what happened during your shift, not how to do their job. Include events and learnings not documented elsewhere.

## Information Sources
1. **Memory**: Review all previous prompts and responses
2. **Conversation**: Extract key information from entire thread
3. **Project Context**: Consider all information from the project

## Template
```
# [TOPIC] Handoff - [DATE]

## Summary
[2-3 sentence overview]

## Priority Development Requirements (PDR)
- **HIGH**: [Must address immediately]
- **MEDIUM**: [Address soon]
- **LOW**: [Be aware]

## Discoveries
- [Unexpected finding 1]
- [Unexpected finding 2]

## Problems & Solutions
- **Problem**: [Issue description]
  **Solution**: [Solution applied]
  ```code example if needed```

## Work in Progress
- [Task 1]: [Progress %]
- [Task 2]: [Progress %]

## Deviations
- [Changed X to Y because Z]

## References
- [doc/path1]
- [doc/path2]
```

## Include
1. **Date**: Current date at document top
2. **Summary**: Brief overview of accomplishments and status
3. **PDR**: Prioritized items needing attention (HIGH/MEDIUM/LOW)
4. **Discoveries**: Unexpected findings and insights
5. **Problems & Solutions**: Pair each problem with solution, include code when helpful
6. **Work in Progress**: Tasks still being worked on with completion estimates
7. **Deviations**: Changes from original plan/approach
8. **References**: Links to relevant docs, code, previous handoffs

## Technical Guidelines
1. **Concise Detail**: Include necessary technical details, avoid exposition
2. **Targeted Code**: Only include code snippets when they clarify a solution
3. **Actionable Info**: Focus on what next developer needs to continue
4. **Error Details**: Include exact error messages for bugs

## Exclude
1. **Documented Info**: Skip if in README, .env, docs
2. **How-to Info**: Don't explain standard procedures
3. **General Context**: Don't explain basics if documented elsewhere

## Structure
1. **Numbering**: Use sequential numbers per directory (1-setup.md)
2. **Brevity**: Be concise but thorough
3. **Organization**: Use chronological or priority structure
4. **Visuals**: Use mermaid charts for complex workflows
   ```mermaid
   graph TD
     A[Problem] --> B[Solution 1]
     A --> C[Solution 2]
   ```

## Example
❌ `The auth system uses JWT tokens with 24h expiry.`

✅ `[2025-02-25] Login failures caused by timezone in token validation. Fixed with UTC standardization.`

Remember: Pass the baton - share only what isn't obvious from existing documentation.