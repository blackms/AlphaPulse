# React v19.0 Developer Mode

## Version-Specific Features
- **React Compiler** - New compiler that automates memoization, reducing the need for manual performance optimizations
- **Actions** - Built-in solution for handling form submissions and mutations without client-server communication libraries
- **Document Metadata** - Direct manipulation of title and metadata from components without external libraries
- **Server Components** - Full integration of Server Components for rendering components on the server
- **Automatic Batching** - Enhanced state update batching for better performance
- **Suspense with Streaming SSR** - Improved server-side rendering with selective hydration
- **Transition API** - Prioritization of UI updates with useTransition and startTransition
- **Asset Loading** - Built-in resource preloading with the use directive
- **useFormStatus and useFormState** - New hooks for form handling with better UX
- **useOptimistic** - Hook for immediate UI updates with optimistic state changes

## Key Skills and Expertise
- **JavaScript/TypeScript** programming with modern language features
- **Component Design** principles and patterns
- **React Hooks** usage and custom hook creation
- **State Management** approaches with context, signals, or external stores
- **Performance Optimization** techniques specific to React
- **Server Component Architecture** for hybrid applications
- **CSS-in-JS or Styling Solutions** for component styling
- **Testing React Applications** with testing libraries
- **Next.js Integration** for full-stack React applications
- **TypeScript Integration** for type-safe component development

## Best Practices
- Leverage the React Compiler's optimizations instead of manual memoization when possible
- Use Server Components for data-fetching and non-interactive UI components
- Implement proper error boundaries for resilient applications
- Take advantage of the Actions API for form handling and server mutations
- Use TypeScript for type-safe props and state
- Implement proper component composition with custom hooks
- Utilize useTransition for improved user experience during state transitions
- Leverage automatic batching for better performance
- Implement code splitting with lazy loading
- Use Context API appropriately with consideration for performance

## File Types
- JavaScript/TypeScript source files (.js, .jsx, .ts, .tsx)
- CSS and CSS modules (.css, .module.css)
- Styled-components or Emotion files (.tsx with styles)
- Jest test files (.test.js, .test.tsx)
- React component files (.jsx, .tsx)
- Static assets (.svg, .png, .jpg)
- Configuration files (package.json, tsconfig.json)
- Build configuration (vite.config.js, webpack.config.js)
- Environment files (.env.*)
- React Server Component files (.server.js, .server.tsx)

## Related Packages
- react ^19.0.0
- react-dom ^19.0.0
- typescript ^5.3.0
- @types/react ^19.0.0
- @types/react-dom ^19.0.0
- vite ^5.0.0
- @vitejs/plugin-react ^5.0.0
- jest ^29.7.0
- @testing-library/react ^15.0.0
- eslint-plugin-react ^8.0.0

## Differences From Previous Version
- **New APIs**:
  - React Compiler for automatic optimizations
  - Actions for form submissions and server mutations
  - Document Metadata API for title and meta tags
  - useFormStatus and useFormState hooks
  - useOptimistic hook for optimistic UI updates
  
- **Enhanced Features**:
  - Improved Server Components integration
  - Better automatic batching system
  - Enhanced Suspense with streaming SSR
  - Streamlined asset loading with use directive
  - More efficient reconciliation algorithm

## Custom Instructions
When working with React 19.0, focus on leveraging its new compiler and progressive rendering capabilities to build high-performance applications. This major version introduces significant improvements to React's core architecture, particularly with the React Compiler which automatically optimizes components to reduce unnecessary re-renders without manual memoization. Take advantage of Server Components for data-fetching and non-interactive UI parts, which can significantly reduce bundle sizes and improve initial load performance. For form handling and server mutations, utilize the new Actions API which simplifies client-server communication without requiring external libraries. When managing state transitions, implement useTransition to prioritize UI updates for better user experience during data loading or intensive computations. For form implementations, leverage the new useFormStatus and useFormState hooks to provide better feedback during form submissions. When developing with TypeScript, use discriminated unions for props to create more type-safe components, and leverage TypeScript's utility types for prop transformations. For styling, choose an approach that aligns with your team's preferences, whether that's CSS modules, styled-components, or Tailwind CSS. When optimizing application performance, use the built-in React DevTools Profiler to identify rendering bottlenecks, and implement code splitting with React.lazy and Suspense to reduce initial bundle sizes. For complex applications, consider state management solutions like Zustand or Jotai which integrate well with React's latest features, or leverage React's built-in Context API for simpler cases.