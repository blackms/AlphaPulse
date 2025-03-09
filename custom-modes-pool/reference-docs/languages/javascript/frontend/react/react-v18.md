# React v18 Developer Mode

## Version-Specific Features
- Concurrent Rendering
- Automatic Batching
- Transitions API
- Suspense for Data Fetching
- Server Components (experimental)

## Key Skills and Expertise
- Component architecture and design patterns
- React hooks and functional components
- State management with Redux, Context API, and Zustand
- Performance optimization and memoization
- React Router and navigation patterns
- React Testing Library and Jest
- TypeScript integration with React

## Best Practices
- Component composition over inheritance
- Hooks for state and side effects
- Immutable state updates
- Proper code splitting
- Accessibility compliance
- Use createRoot instead of ReactDOM.render
- Leverage automatic batching for performance
- Implement useTransition for improved UX during updates

## File Types
- JavaScript (.js, .jsx)
- TypeScript (.ts, .tsx)
- CSS/SCSS (.css, .scss)
- HTML (.html)
- JSON configuration files (.json)

## Related Packages
- react-dom ^18.0.0
- react-router-dom ^6.0.0
- @reduxjs/toolkit ^1.8.0
- @tanstack/react-query ^4.0.0
- styled-components ^5.3.0 / @emotion/react ^11.0.0

## Differences From React 17
- **New APIs**:
  - createRoot replaces ReactDOM.render
  - startTransition and useTransition for non-blocking updates
  - useDeferredValue for deferring expensive computations
  - useId for generating stable unique IDs
  - useSyncExternalStore for external stores
  - useInsertionEffect for CSS-in-JS libraries

- **Enhanced Features**:
  - Automatic batching extended to all updates
  - Improved Suspense with SSR support
  - Improved hydration with selective hydration
  - Streaming SSR for faster Time To First Byte
  - Stricter Strict Mode with double-mount behavior

## Custom Instructions
When implementing React v18 applications, prioritize functional components with hooks over class components. Take advantage of React 18's concurrent features like useTransition and Suspense for improved user experience. Use proper React patterns like compound components, render props, or custom hooks to maximize reusability. Always consider performance optimizations like memoization, virtualization for long lists, and efficient re-rendering strategies. Implement proper error boundaries and suspense for better user experience. Migrate from ReactDOM.render to createRoot API for all applications.