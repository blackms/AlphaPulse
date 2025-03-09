# Laravel v11.0 Developer Mode

## Version-Specific Features
- **Stripped Back Initialization** - Significantly reduced bootstrap process for improved performance
- **Invokable Routes** - Support for invokable classes as route handlers for cleaner routing
- **Pest Integration** - First-class support for Pest testing framework instead of PHPUnit
- **Simplified Application Structure** - Streamlined directory structure with fewer default files
- **Prompt Commands** - Interactive CLI prompts for Artisan commands
- **API Integrations** - Built-in abstractions for working with external APIs
- **Native Enum Support** - Comprehensive support for PHP 8.1+ enums throughout the framework
- **Lazy Collections** - Enhanced lazy collection capabilities for memory-efficient data processing
- **Improved Validation** - More powerful validation rules and better error reporting
- **Streamlined Middleware** - Simplified middleware registration and handling

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
- Leverage the new invokable routes for single-action controllers
- Utilize PHP 8.2+ features like readonly properties and native enums
- Implement proper validation using Laravel's validation system
- Use Laravel Sanctum for API authentication
- Structure applications with domains or modules when appropriate
- Implement proper error handling with custom exception handlers
- Use Laravel's queuing system for long-running tasks
- Leverage Eloquent relationships for efficient data loading
- Implement comprehensive testing with Pest
- Use typed properties and return types for better code quality

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

## Related Packages
- laravel/framework ^11.0
- laravel/sanctum ^4.0
- laravel/breeze ^2.0
- laravel/jetstream ^5.0
- pestphp/pest ^2.0
- spatie/laravel-permission ^6.0
- barryvdh/laravel-debugbar ^3.9
- inertiajs/inertia-laravel ^1.0
- nunomaduro/collision ^8.0
- spatie/laravel-query-builder ^5.0

## Differences From Previous Version
- **New APIs**:
  - Invokable route handlers
  - Enhanced API integration tools
  - Simplified middleware registration
  - New Artisan prompt commands
  
- **Removed Features**:
  - PHPUnit is no longer the default testing framework
  - Some legacy support for older PHP versions
  - Several bootstrapping files have been removed
  
- **Enhanced Features**:
  - Significantly faster application bootstrap
  - More powerful validation capabilities
  - Improved Eloquent performance
  - Better error reporting and debugging tools

## Custom Instructions
When working with Laravel 11.0, focus on leveraging its streamlined architecture and modern PHP features to build efficient web applications. This major version represents a significant evolution with a focus on simplicity and performance. Take advantage of the stripped-back initialization process which significantly improves application bootstrap time. For routing, embrace the new invokable route handlers which provide a cleaner approach to single-action controllers. Structure your application following Laravel's convention-over-configuration philosophy, but be aware that Laravel 11 introduces a more simplified directory structure with fewer default files. For database operations, utilize Eloquent ORM with its expressive syntax, leveraging relationships, eager loading, and query scopes for efficient data access. Implement validations using Laravel's powerful validation system, taking advantage of the enhanced validation rules in this version. For testing, adopt Pest as the preferred testing framework, which is now the default in Laravel 11, offering a more expressive syntax for your test suites. When building APIs, use Laravel's API resources for transforming models into JSON responses, and consider Laravel Sanctum for lightweight API authentication. For larger applications, consider organizing your code into domains or modules rather than the traditional MVC structure, which can help maintain separation of concerns as your application grows. Take advantage of Laravel's queue system for handling long-running tasks asynchronously, improving user experience by offloading time-consuming operations to background processes.