# Angular v19 Developer Mode

## Version-Specific Features
- Incremental Hydration with @defer syntax for selective component loading
- Route-level Render Mode configuration (SSR, CSR, Pre-rendering)
- Linked Signals for reactive state management and HTTP requests
- Event Replay for enhanced SSR and smoother interactions
- Standalone Components as default configuration
- Hot Module Replacement for improved development experience
- Enhanced TypeScript integration with stricter typing
- Improved build performance and smaller bundle sizes
- Angular Signals ecosystem maturation

## Key Skills and Expertise
- TypeScript and ES6+ JavaScript fundamentals
- Angular framework architecture and component model
- Angular CLI for project scaffolding and management
- Reactive programming with RxJS
- State management (NgRx, Signals)
- Angular routing and lazy loading
- Server-side rendering and hydration techniques
- Component design patterns and best practices
- Unit testing with Jasmine/Karma
- End-to-end testing with Cypress or Playwright

## Best Practices
- Use standalone components by default
- Leverage Angular Signals for state management
- Implement proper change detection strategies
- Utilize @defer for performance-critical applications
- Configure route-level rendering modes appropriately
- Follow Angular style guide for consistent code
- Implement lazy loading for feature modules
- Use correct input/output binding patterns
- Properly handle component lifecycles
- Implement comprehensive error handling

## File Types
- TypeScript (.ts)
- HTML templates (.html)
- CSS/SCSS/LESS (.css, .scss, .less)
- TypeScript configuration (.tsconfig.json)
- Angular configuration (angular.json)
- Package configuration (package.json)
- Module definition files (.d.ts)

## Related Packages
- @angular/core ^19.0.0
- @angular/common ^19.0.0
- @angular/forms ^19.0.0
- @angular/router ^19.0.0
- @angular/compiler ^19.0.0
- @angular/platform-browser ^19.0.0
- @angular/platform-browser-dynamic ^19.0.0
- rxjs ^7.0.0
- zone.js ^0.13.0
- typescript ~5.2.0

## Differences From Angular 16
- **New APIs**:
  - Incremental Hydration API with @defer syntax
  - Route-level Render Mode configuration
  - Linked Signals for reactive state management
  - Event Replay for SSR improvements
  
- **Removed Features**:
  - NgModules are no longer the default (standalone components are)
  - Some deprecated APIs from Angular 16 have been removed
  - Legacy HTTP client completely removed in favor of modern implementation
  
- **Enhanced Features**:
  - Signals system has matured significantly since v16
  - Improved build performance and smaller bundles
  - Better TypeScript integration with stricter typing
  - More efficient change detection system
  - Enhanced developer tooling and debugging experience

## Custom Instructions
Develop Angular v19 applications with a focus on performance optimization and modern architecture patterns. Leverage standalone components as the default approach, and take full advantage of Angular Signals for reactive state management. Implement incremental hydration with @defer for improved loading performance, and configure route-level rendering modes to optimize for different sections of your application. Properly structure your application with feature-based organization and lazy loading. Use TypeScript effectively with strict typing for better code quality. When migrating from older versions, pay special attention to the removal of NgModules as the default approach, and update your state management strategy to leverage Signals where appropriate. Follow the Angular style guide rigorously for maintainable, consistent code. Implement comprehensive testing with both unit and end-to-end tests to ensure application quality and stability.