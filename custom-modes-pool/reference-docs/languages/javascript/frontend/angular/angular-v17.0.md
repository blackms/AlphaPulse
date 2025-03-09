# Angular v17.0 Developer Mode

## Version-Specific Features
- **Standalone Components by Default** - New projects use standalone components, eliminating NgModules for most applications
- **Block Template Syntax** - New @if, @for, @switch control flow syntax replacing *ngIf and *ngFor directives
- **Signals Architecture** - Reactive state management system with fine-grained reactivity
- **Deferred Loading** - Built-in component lazy loading with @defer blocks
- **Server-Side Rendering** - Enhanced Angular Universal with automatic hydration
- **Deferrable Views** - Load components conditionally based on triggers like viewport visibility
- **Hydration Enhancements** - Seamless transition from server-rendered to client-interactive views
- **Vite Dev Server Integration** - Faster development experience with Vite instead of webpack
- **Built-in Image Optimization** - NgOptimizedImage directive for automatic responsive images
- **New Application Builder** - Esbuild-based compilation for significantly faster builds

## Key Skills and Expertise
- **TypeScript Programming** with deep understanding of types and interfaces
- **Component Architecture** design and implementation
- **Reactive Programming** with RxJS and Signals
- **State Management** patterns and solutions
- **Router Configuration** and navigation strategies
- **HTTP Communication** with Angular's HttpClient
- **Forms Development** using Reactive and Template-driven approaches
- **Testing Angular Applications** with Jasmine/Karma or Jest
- **Component Design** principles and best practices
- **Performance Optimization** techniques for Angular applications

## Best Practices
- Adopt standalone components for simplified architecture
- Use the new block template syntax for improved readability and performance
- Implement Signals for state management in new applications
- Leverage deferred loading for better initial load performance
- Implement proper component composition with input/output bindings
- Use OnPush change detection strategy for optimal performance
- Structure applications with feature modules or standalone component groups
- Implement lazy loading for routes to reduce initial bundle size
- Use typed forms for improved type safety
- Implement comprehensive unit and integration testing

## File Types
- TypeScript source files (.ts)
- Angular component files (.component.ts)
- HTML templates (.html)
- Component styles (.css, .scss, .less)
- Angular modules (.module.ts)
- Angular services (.service.ts)
- Angular directives (.directive.ts)
- Angular pipes (.pipe.ts)
- Unit test files (.spec.ts)
- Configuration files (angular.json, tsconfig.json)

## Related Packages
- @angular/core ^17.0.0
- @angular/common ^17.0.0
- @angular/forms ^17.0.0
- @angular/router ^17.0.0
- @angular/platform-browser ^17.0.0
- @angular/platform-server ^17.0.0
- rxjs ^7.8.0
- zone.js ^0.14.0
- typescript ^5.2.0
- @angular/compiler ^17.0.0

## Differences From Previous Version
- **New APIs**:
  - Block template syntax (@if, @for, @switch)
  - Signals API (signal(), computed(), effect())
  - Deferred loading (@defer block)
  - View transitions API
  
- **Enhanced Features**:
  - Improved SSR with automatic hydration
  - Faster compilation with esbuild
  - Better developer experience with Vite
  - More performant change detection
  - Improved routing with standalone components

## Custom Instructions
When working with Angular 17.0, focus on leveraging its modern architecture centered around standalone components and the new reactive primitives. This major version represents a transformative update that simplifies Angular development while improving performance. Start new projects using the standalone component approach, eliminating the need for NgModules in most cases for cleaner, more maintainable code. Adopt the new block template syntax (@if, @for, @switch) which offers improved type checking, better runtime performance, and cleaner templates compared to the older structural directives. For state management, consider using Angular's new Signals API which provides fine-grained reactivity similar to popular reactive libraries but fully integrated into the Angular ecosystem. Take advantage of the @defer block for component lazy loading, allowing you to defer the loading of components until they're needed based on various triggers such as viewport visibility. Implement server-side rendering with automatic hydration for improved initial load performance and SEO. When building forms, use the typed reactive forms API for better type safety. For application building, leverage the esbuild-based compilation system which significantly improves build times. During development, benefit from the Vite integration which provides near-instant updates during development. When optimizing application performance, utilize the NgOptimizedImage directive for automatic image optimization, and implement proper lazy loading strategies for both routes and components.