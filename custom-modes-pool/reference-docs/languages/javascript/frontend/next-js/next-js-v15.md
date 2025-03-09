# Next.js v15 Developer Mode

## Version-Specific Features
- React 19 Support with Server Components and Actions
- Modified Caching Strategy (less aggressive defaults)
- Partial Prerendering for mixed static/dynamic content
- `next/after` API for post-response execution
- Turbopack Stable Release for development
- Built-in Form Component with automatic validation
- TypeScript Configuration Support (next.config.ts)
- ESLint 9 Support with backward compatibility
- Redesigned Error UI with improved stack traces
- Streaming Metadata for improved page transitions

## Key Skills and Expertise
- Server Components architecture and patterns
- React 19 features and component model
- Advanced data fetching strategies
- Caching optimization techniques
- Server-side rendering and hydration
- TypeScript for type-safe development
- Form handling and validation
- Performance optimization with partial prerendering
- Middleware and edge functions
- SEO and metadata optimization

## Best Practices
- Explicitly define caching strategies with fetch options
- Use Partial Prerendering for mixed static/dynamic pages
- Leverage `next/after` for non-critical background tasks
- Adopt the built-in form component for enhanced UX
- Configure appropriate render strategies per route
- Use TypeScript for configuration files
- Implement proper error boundaries and fallbacks
- Structure API routes efficiently
- Apply proper metadata for SEO optimization
- Use React Server Components for data-fetching operations

## File Types
- JavaScript (.js, .jsx)
- TypeScript (.ts, .tsx)
- CSS/SCSS (.css, .scss)
- Configuration files (next.config.js/ts)
- Environment files (.env*)
- Markdown (.md)
- JSON configuration files (.json)

## Related Packages
- react ^19.0.0-rc.0
- react-dom ^19.0.0-rc.0
- next ^15.0.0
- @types/react ^18.2.0
- @types/node ^20.0.0
- typescript ^5.0.0
- eslint ^8.0.0 or ^9.0.0
- eslint-config-next ^15.0.0

## Differences From Next.js 13
- **New APIs**: 
  - `next/after` for background task execution
  - Built-in Form Component with validation
  - TypeScript configuration support
  - Partial Prerendering API
  
- **Modified Features**:
  - Default caching behavior is less aggressive
  - Turbopack is now stable for development
  - ESLint 9 support with backward compatibility
  
- **Enhanced Features**:
  - React 19 integration (vs React 18 in v13)
  - Improved error UI and debugging experience
  - Faster development server startup (up to 53%)
  - Better streaming for dynamic content
  - Enhanced metadata handling

## Custom Instructions
Develop Next.js v15 applications with a focus on performance optimization through intelligent caching strategies and partial prerendering. Use React 19 features appropriately, taking advantage of Server Components for data-fetching and static content, and Client Components for interactivity. Implement explicit caching directives with fetch options rather than relying on defaults. Utilize the `next/after` API for post-response tasks like logging and analytics. Take advantage of the built-in form component for better user experiences with automatic validation and progressive enhancement. Structure your application with appropriate file-based routing conventions, and use TypeScript for enhanced type safety, including in your configuration files. For complex applications, combine partial prerendering with streaming to deliver the best balance of performance and dynamic content. When migrating from earlier versions, carefully review and update your caching strategies to align with the new default behaviors in Next.js 15.