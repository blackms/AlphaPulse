# Laravel v12.0 Developer Mode

## Version-Specific Features
- **New Starter Kits** - Modern starter kits for React, Vue, and Livewire with shadcn components and TypeScript
- **WorkOS AuthKit Integration** - Built-in support for social authentication, passkeys, and Single Sign-On (SSO)
- **Minimal Breaking Changes** - Focus on stability and backwards compatibility for easier upgrades
- **PHP 8.2-8.4 Support** - Compatibility with the latest PHP versions
- **Enhanced MVC Architecture** - Continued refinement of Laravel's Model-View-Controller implementation
- **API Integrations** - Further improvements to built-in abstractions for working with external APIs
- **Continued Pest Integration** - Enhanced support for Pest testing framework
- **Performance Optimizations** - Improved caching mechanisms and faster query building
- **nestedWhere() Method** - New Eloquent query builder method for complex conditions
- **GraphQL Support** - Improved capabilities for building GraphQL APIs

## Key Skills and Expertise
- **PHP 8.2+ Programming** with modern language features
- **MVC Architecture** implementation using Laravel patterns
- **Eloquent ORM** for database interactions
- **Blade Templating** for view rendering
- **Request Lifecycle** understanding and middleware implementation
- **Authentication and Authorization** using Laravel's systems
- **API Development** with Laravel resources and controllers
- **Database Schema Management** with migrations
- **Job Queues and Background Processing**
- **Testing with Pest** framework

## Best Practices
- Follow PSR-2 for coding style and PSR-4 for autoloading classes
- Keep business logic in models (Fat Models, Skinny Controllers)
- Utilize PHP 8.2+ features like readonly properties and native enums
- Implement proper validation using Laravel's validation system
- Use Laravel Sanctum for API authentication
- Leverage caching for frequently accessed data
- Optimize database queries with eager loading to avoid N+1 problems
- Structure applications with domains or modules when appropriate
- Implement proper error handling with custom exception handlers
- Implement comprehensive testing with Pest

## File Types
- PHP source files (.php)
- Blade template files (.blade.php)
- Environment configuration (.env)
- JSON configuration files (.json)
- YAML configuration files (.yaml)
- Migration files
- Seeder files
- Factory files
- Test files
- Language files
- TypeScript files (.ts) for modern starter kits

## Related Packages
- laravel/framework ^12.0
- laravel/sanctum ^4.0
- laravel/workos-authkit ^1.0
- pestphp/pest ^2.0
- spatie/laravel-permission ^6.0
- barryvdh/laravel-debugbar ^3.9
- inertiajs/inertia-laravel ^1.0
- nunomaduro/collision ^8.0
- spatie/laravel-query-builder ^5.0
- livewire/livewire ^3.0

## Differences From Previous Version
- **New APIs**:
  - Modern starter kits for React, Vue, and Livewire
  - WorkOS AuthKit integration for authentication
  - nestedWhere() method for complex queries
  - GraphQL support enhancements
  
- **Removed Features**:
  - Breeze and Jetstream starter kits replaced by new starter kits
  - Some legacy compatibility layers
  
- **Enhanced Features**:
  - Focus on stability with minimal breaking changes
  - Performance optimizations for caching and queries
  - Improved error reporting and debugging tools
  - Enhanced security features

## Custom Instructions
When working with Laravel 12, focus on leveraging its stability and modern features to build robust web applications. This version emphasizes backwards compatibility while introducing quality-of-life improvements. Take advantage of the new starter kits which provide modern front-end integration with React, Vue, or Livewire, complete with shadcn components and TypeScript support. For authentication, explore the integrated WorkOS AuthKit which provides social logins, passkeys, and SSO capabilities out of the box. Continue using Laravel's convention-over-configuration approach, with its expressive syntax for database operations, routing, and middleware. Utilize Eloquent ORM for database interactions, taking advantage of relationships, eager loading, and the new nestedWhere() method for complex queries. For testing, continue with Pest as the preferred testing framework, benefiting from its expressive syntax. When building APIs, use Laravel's API resources for transforming models into JSON responses, and consider exploring the enhanced GraphQL support for more complex API needs. For larger applications, maintain the domain-oriented or modular organization approach to keep your codebase maintainable. Follow the PSR standards for coding style and autoloading, and leverage PHP 8.2+ features for more expressive and type-safe code. Remember that Laravel 12 is designed for a smooth upgrade path from Laravel 11, so existing applications can generally upgrade with minimal code changes.