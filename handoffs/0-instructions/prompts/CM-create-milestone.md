# Creating a Milestone

Use this prompt when you need to create a new milestone to consolidate accumulated handoffs.

## Simple Prompt Template

```
I need to create a milestone for our completed [FEATURE/COMPONENT]. Please:
(The handoff directory may not be at the project root)
1. Check if there are handoff documents in the handoffs/ root directory:
   - If no handoffs exist, suggest creating a handoff first before proceeding
   - If handoffs exist but seem outdated (based on dates/content), suggest creating a final handoff

2. Read the handoffs/0-instructions/2-milestone-instructions.md
3. Determine the next sequential milestone number by examining existing milestone directories
4. Create the milestone directory with that number
5. Move all numbered handoff documents from the handoffs/ root into this milestone directory
6. Create the required 0-milestone-summary.md and 0-lessons-learned.md files
7. Using the language that seems most appropriate for the current environment, write a script like the examples in 3-milestone-scripts.md to reorganize the handoffs into a milestone folder.
```


## Important Workflow Note

A milestone should always consolidate recent handoffs. Creating a final handoff before making a milestone ensures that all recent work is captured in the milestone summary. This maintains the logical progression:

1. Work on feature/component → Create handoffs during development
2. Complete feature/component → Create final handoff with completion status
3. Consolidate work → Create milestone that includes this final handoff