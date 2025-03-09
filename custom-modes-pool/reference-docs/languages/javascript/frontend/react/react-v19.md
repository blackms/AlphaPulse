# React v19 Developer Mode

## Version-Specific Features
- React Compiler for automatic optimizations
- Actions for asynchronous operations management
- Enhanced Server Components support
- Improved Concurrent Rendering
- Asset Loading Optimization (preload/preinit APIs)
- Custom Cache Implementation
- Document Metadata management
- WebAssembly Integration (`useWasm` hook)
- Full Custom Elements (Web Components) support
- Enhanced TypeScript integration

## Key Skills and Expertise
- Server Component architecture and patterns
- Client/Server component boundaries
- React compiler optimization principles
- Actions for data operations
- Asset loading strategies
- WebAssembly integration
- React custom caching strategies
- Server-side rendering optimization
- TypeScript with React 19 features

## Best Practices
- Let React Compiler handle optimizations (avoid manual memoization)
- Use Server Components for data-fetching and static content
- Implement Actions for asynchronous operations
- Leverage custom caching strategies for data-heavy applications
- Use preload/preinit for critical resources
- Properly separate client and server components
- Implement proper error boundaries with Suspense
- Replace forwardRef with direct ref props where possible
- Use `<Context>` directly as a provider

## File Types
- JavaScript (.js, .jsx)
- TypeScript (.ts, .tsx)
- CSS/SCSS (.css, .scss)
- Server Component files (.server.js/.server.tsx)
- Client Component files (.client.js/.client.tsx)

## Related Packages
- react ^19.0.0
- react-dom ^19.0.0
- @types/react ^19.0.0
- react-server-dom-webpack ^19.0.0
- @react/compiler (experimental)

## Differences From React 18
- **New APIs**: 
  - `use` hook for reading resources in render
  - Direct `ref` prop in function components
  - `<Context>` as direct provider
  - `useEvent` and `useResource` hooks
  - `preload` and `preinit` resource loading APIs

- **Removed Features**:
  - PropTypes and defaultProps for function components
  - String refs
  - React.createFactory
  - Several react-dom/test-utils
  - ReactDOM.render (fully removed)

- **Enhanced Features**:
  - Further optimized concurrent rendering
  - Improved Server Components support
  - Enhanced Suspense with better data fetching
  - Smarter hydration process
  - Automatic compiler optimizations

## Custom Instructions
Develop React v19 applications leveraging the new architectural patterns and performance optimizations. Take advantage of Server Components for data-fetching and static content rendering to reduce client bundle size. Let the React Compiler handle optimizations instead of manual memoization with useCallback/useMemo. Use the new Actions API for managing complex asynchronous operations with automatic state transitions. Implement the preload/preinit APIs for critical resources to improve performance. Leverage the enhanced Suspense capabilities for better loading experiences and error handling. Use the new `use` hook for resource reading and the direct ref props to simplify component code. When migrating from React 18, pay special attention to breaking changes like the removal of propTypes and defaultProps for function components, replacing them with TypeScript types and ES6 default parameters. Properly separate client and server components, and ensure your testing approach accounts for changes in the react-dom/test-utils APIs.