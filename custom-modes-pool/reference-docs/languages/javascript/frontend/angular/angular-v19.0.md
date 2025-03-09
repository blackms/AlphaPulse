# Angular v19.0 Developer Mode

## Version-Specific Features
- **Standalone Components by Default** - No need to specify standalone: true anymore, creating simpler component architecture
- **Incremental Hydration** - Developer preview feature allowing parts of the application to hydrate independently
- **Route Level Render Mode** - Ability to specify render modes (client-side, server-side, prerendered) at the route level
- **Signal-Based Resource API** - Enhanced state management and HTTP request handling with reactive primitives
- **View Transitions** - Smoother transitions between routes with withViewTransitions function
- **Zoneless Applications** - Support for running Angular without Zone.js for improved performance
- **LinkedSignal** - New reactive primitive to enhance state management capabilities
- **Time Manipulation APIs** - Tools for manipulating time in Angular applications for testing and debugging
- **Signals Architecture** - Continued refinement of the reactive state management system introduced in v17
- **Server-Side Rendering** - Further enhanced with more granular control over rendering strategies

## Key Skills and Expertise
- **TypeScript Programming** with strong understanding of TypeScript 5.2+
- **Component Architecture** design and implementation with standalone approach
- **Reactive Programming** with RxJS and enhanced Signals API
- **State Management** using the Signal-Based Resource API
- **Router Configuration** with route-level render mode specifications
- **Server-Side Rendering** concepts and implementation strategies
- **HTTP Communication** with Angular's HttpClient and Resource API
- **Forms Development** using Reactive and Template-driven approaches
- **Testing Angular Applications** including zoneless application testing
- **Performance Optimization** techniques for modern Angular applications

## Best Practices
- Embrace standalone components as the default architectural approach
- Utilize the `strictStandalone` compiler flag to enforce consistency
- Leverage Signals and Signal-Based Resource API for reactive programming
- Prepare for zoneless applications with signals and OnPush change detection
- Implement proper view transitions for smoother user experience
- Use route-level render mode configuration for optimized application performance
- Adopt incremental hydration where appropriate for improved initial load times
- Continue using OnPush change detection strategy for optimal performance
- Implement lazy loading for routes and deferred components to reduce initial bundle size
- Keep dependencies updated regularly for security and compatibility

## File Types
- TypeScript source files (.ts)
- Angular component files (.component.ts)
- HTML templates (.html)
- Component styles (.css, .scss, .less)
- Angular services (.service.ts)
- Angular directives (.directive.ts)
- Angular pipes (.pipe.ts)
- Unit test files (.spec.ts)
- Configuration files (angular.json, tsconfig.json)
- Server configuration files for SSR

## Related Packages
- @angular/core ^19.0.0
- @angular/common ^19.0.0
- @angular/forms ^19.0.0
- @angular/router ^19.0.0
- @angular/platform-browser ^19.0.0
- @angular/platform-server ^19.0.0
- rxjs ^7.8.0
- zone.js ~0.14.0 (optional with zoneless support)
- typescript ~5.2.0
- @angular/compiler ^19.0.0
- @angular/cli ^19.0.0
- @angular-devkit/build-angular ^19.0.0

## Differences From Previous Version
- **New APIs**:
  - Signal-Based Resource API for state and HTTP management
  - Route-level render mode configuration options
  - LinkedSignal for enhanced state management
  - Time manipulation APIs for testing
  - View transitions with withViewTransitions
  
- **Enhanced Features**:
  - Standalone components now default (no need for standalone: true)
  - Support for zoneless applications without Zone.js
  - Improved server-side rendering with more granular control
  - Incremental hydration for better performance
  - More mature Signals implementation with additional capabilities

## Custom Instructions
When working with Angular 19.0, focus on embracing its simplified architecture and enhanced performance capabilities. This version builds on the transformative changes introduced in v17, making standalone components the default approach (no need to specify `standalone: true` anymore) for cleaner, more maintainable code. Take advantage of the enhanced server-side rendering capabilities, particularly the ability to specify render modes at the route level to optimize your application's performance and SEO. Consider using the new Signal-Based Resource API for managing application state and HTTP requests in a more reactive way. For smoother user experiences, implement view transitions using the withViewTransitions function to create visually appealing navigation between routes. If you're building high-performance applications, explore the zoneless application support which allows Angular to run without Zone.js, reducing unnecessary change detection cycles. Prepare your applications for this approach by using signals and OnPush change detection. Leverage the new LinkedSignal reactive primitive to enhance your state management capabilities. For testing time-dependent features, utilize the new time manipulation APIs. When upgrading from previous versions, use the Angular CLI's `ng update` command to check for breaking changes and modify your code as necessary. Keep your dependencies updated regularly to ensure compatibility and security, and continue implementing best practices like lazy loading and efficient change detection strategies for optimal performance.