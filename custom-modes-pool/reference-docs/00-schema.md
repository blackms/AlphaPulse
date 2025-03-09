# Technical Reference Documentation Schema

This document outlines the comprehensive schema for creating version-specific technology reference documentation. These reference files serve as knowledge sources for LLM-generated custom modes for specific technologies and versions.

## Filename Format

Use the following format for filenames: `{technology-name}-v{version}.md`

Examples:
- `react-v18.md`
- `django-v4.2.md`
- `tensorflow-v2.12.md`

## Document Structure

Each reference document MUST include the following sections in this order:

### 1. Title (Required)
Format: `# {Technology Name} v{Version} Developer Mode`

### 2. Version-Specific Features (Required)
List 8-10 key features or characteristics specific to this technology version. Focus on what makes this version distinct.

```
## Version-Specific Features
- Feature 1 with brief technical description
- Feature 2 with brief technical description
- ...
```

### 3. Key Skills and Expertise (Required)
List 7-10 technical skills developers need to work effectively with this technology version.

```
## Key Skills and Expertise
- Skill 1
- Skill 2
- ...
```

### 4. Best Practices (Required)
List 7-10 recommended practices specific to this technology version.

```
## Best Practices
- Practice 1
- Practice 2
- ...
```

### 5. File Types (Required)
List relevant file extensions and types used with this technology.

```
## File Types
- File type 1 (.extension)
- File type 2 (.extension)
- ...
```

### 6. Related Packages (Required)
List main dependencies, libraries, or tools with their compatible version ranges.

```
## Related Packages
- package1 ^x.y.z
- package2 ^x.y.z
- ...
```

### 7. Differences From Previous Version (Optional but Recommended)
Include when documenting version upgrades to highlight migration considerations.

```
## Differences From {Previous Version}
- **New APIs**: 
  - API 1 with brief description
  - API 2 with brief description
  
- **Removed Features**:
  - Feature 1 with migration guidance
  - Feature 2 with migration guidance
  
- **Enhanced Features**:
  - Enhancement 1
  - Enhancement 2
```

### 8. Custom Instructions (Required)
Provide detailed guidance for Roo to assist developers using this technology version. 
This should be a comprehensive paragraph covering key aspects of development with 
this technology version, including migration considerations if applicable.

```
## Custom Instructions
Technology-specific guidance with version-specific information...
```

## Level of Detail

The reference documentation should be highly detailed and technical:

1. **Specificity**: Include specific API names, method signatures, and version numbers
2. **Technicality**: Provide technical explanations, not just feature lists
3. **Code References**: Include naming conventions, package structures, and technical terms
4. **Version Precision**: Be explicit about version-specific behaviors
5. **Migration Path**: When applicable, include clear instructions for upgrading

## Example Code Inclusion (Optional)

When helpful, include brief code snippets that demonstrate version-specific features:

```
// Example for Technology vX.Y
import { newFeature } from 'technology';

function exampleUsage() {
  newFeature.doSomething({
    option1: true,
    option2: 'value'
  });
}
```

## Validation Checklist

Before submitting a reference document, verify:

- [ ] Filename follows the proper format
- [ ] All required sections are included
- [ ] Version-specific information is accurate
- [ ] Dependencies include compatible version ranges
- [ ] Custom instructions are comprehensive
- [ ] If applicable, differences from previous version are clearly articulated
- [ ] Technical details are precise and specific
- [ ] Document provides sufficient depth for LLM to generate expert guidance

## Important Notes

1. Be objective and factual - avoid subjective opinions about the technology
2. Include precise version numbers for all dependencies
3. Focus on technical information, not general descriptions
4. When in doubt, include more technical detail rather than less
5. Use consistent terminology throughout the document
6. For major version changes, emphasize breaking changes and migration paths
7. Cross-reference official documentation when possible

The more precise and detailed the reference documentation, the more effectively the LLM can generate specialized modes tailored to specific technology versions.