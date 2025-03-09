# Vue v3.5 Developer Mode

## Version-Specific Features
- Optimized Reactivity System with 56% reduced memory usage
- Enhanced Custom Elements support with new configuration options
- New `useTemplateRef()` API for dynamic template references
- Improved TypeScript integration and type inference
- Significantly faster reactivity tracking for large arrays (up to 10x)
- Support for custom elements without Shadow DOM
- New APIs: `useHost()`, `useShadowRoot()`, and `this.$host`
- Reactivity stability improvements for SSR
- Performance optimizations for deeply reactive objects

## Key Skills and Expertise
- Vue component architecture and lifecycle
- Vue 3 Composition API and Reactivity System
- Web Components and Custom Elements integration
- Pinia/Vuex for state management
- Vue Router for SPA navigation
- TypeScript with Vue type system
- Single File Components with `<script setup>` syntax
- Unit testing with Vue Test Utils or Vitest
- Build tools (Vite, Webpack)
- Advanced CSS and component styling

## Best Practices
- Use Composition API with `<script setup>` syntax
- Leverage the optimized reactivity system for performance
- Utilize TypeScript for type safety and better DX
- Extract reusable composables for shared logic
- Implement proper state management with Pinia
- Use dynamic template refs with `useTemplateRef()`
- Apply custom elements when building reusable components
- Structure applications with feature-based organization
- Use Suspense for async component loading
- Prefer shallow refs for large data structures

## File Types
- Vue Single File Components (.vue)
- JavaScript (.js)
- TypeScript (.ts)
- CSS/SCSS/LESS (.css, .scss, .less)
- HTML (.html)
- JSON configuration files (.json)

## Related Packages
- vue ^3.5.0
- vue-router ^4.2.0
- pinia ^2.1.0
- vite ^4.4.0
- @vue/test-utils ^2.4.0
- vitest ^0.34.0
- typescript ^5.0.0
- @vitejs/plugin-vue ^4.3.0

## Differences From Vue 3.0
- **New APIs**: 
  - `useTemplateRef()` for dynamic template references
  - `useHost()` and `useShadowRoot()` for custom elements
  - Custom element configuration via `configureApp`
  
- **Enhanced Features**:
  - 56% reduced memory usage in reactivity system
  - Up to 10x faster reactivity tracking for large arrays
  - Improved TypeScript type inference
  - Better SSR stability with resolved memory leaks
  - Custom elements without Shadow DOM support
  
- **Performance Improvements**:
  - Optimized rendering pipeline
  - More efficient reactivity change tracking
  - Better memory management
  - Faster updates for deeply nested objects

## Custom Instructions
Develop Vue v3.5 applications with a focus on performance and modern architecture patterns. Leverage the significantly optimized reactivity system for better memory usage and faster updates, especially with large data structures. Use the Composition API with `<script setup>` syntax for cleaner, more maintainable components. Extract shared logic into composables and implement proper state management with Pinia. Take advantage of the new APIs like `useTemplateRef()` for dynamic template references, and explore custom elements integration for more reusable component patterns. Use TypeScript consistently for better developer experience and type safety. When building components, follow the principle of single responsibility and properly handle component lifecycles. For larger applications, implement a feature-based folder structure and lazy loading with Suspense. Test components thoroughly using Vue Test Utils or Vitest with component testing best practices.