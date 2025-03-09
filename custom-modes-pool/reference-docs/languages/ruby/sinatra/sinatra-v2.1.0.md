# Sinatra v2.1.0 Developer Mode

## Version-Specific Features
- **Lightweight Micro-framework** - Minimalist approach to web development allowing quick API and application creation with little boilerplate
- **Enhanced Routing System** - Powerful pattern matching for URLs using Mustermann for more flexible route definitions
- **Nested View Subdirectories** - Improved template rendering with support for organizing views in nested subdirectories
- **Streaming Response Support** - Migrated from Thin to Rainbows in development environment for improved streaming capabilities
- **Modular Application Architecture** - Support for both classic and modular application styles for better code organization
- **Ruby 2.3+ Compatibility** - Dropped support for Ruby 2.2, now requiring Ruby 2.3 or later for improved language features
- **Rack Middleware Integration** - Seamless integration with Rack ecosystem allowing use of various middleware components
- **Improved Security** - Enhanced protection against common web vulnerabilities through updated rack-protection
- **Template Engine Flexibility** - Support for multiple templating engines including ERB, HAML, Slim, and others
- **Built-in Development Server** - Integrated WEBrick server for quick development without additional configuration

## Key Skills and Expertise
- **Ruby Programming** with focus on Ruby 2.3+ features and idioms
- **HTTP and Web Protocol** understanding including request/response cycle
- **RESTful API Design** principles and implementation patterns
- **Rack Middleware** concepts and custom middleware development
- **Template Engine** usage across multiple formats (ERB, HAML, etc.)
- **Database Integration** techniques particularly with lightweight ORMs
- **MVC Architecture** principles as applied to lightweight frameworks
- **Front-end Integration** with JavaScript frameworks and vanilla JS
- **Testing Methodologies** for API and web applications
- **Deployment Strategies** for Sinatra applications in various environments

## Best Practices
- Use modular application style for larger projects to improve code organization and maintainability
- Leverage Sinatra's lightweight nature for microservices and APIs rather than complex monolithic applications
- Implement proper error handling with custom error pages and status codes
- Utilize helper methods to keep route handlers clean and focused
- Structure the application using MVC principles even though Sinatra doesn't enforce them
- Take advantage of Sinatra extensions from sinatra-contrib for common functionality
- Use environment-specific configuration for development, testing, and production
- Implement proper authentication and authorization mechanisms
- Write comprehensive tests using RSpec, Minitest, or Rack::Test
- Keep route definitions concise and organized by HTTP method

## File Types
- Ruby source files (.rb)
- ERB/HAML/Slim templates (.erb, .haml, .slim)
- Configuration files (config.ru, Gemfile)
- Static assets (.css, .js, .jpg, .png)
- YAML configuration files (.yml, .yaml)
- JSON data files (.json)
- Markdown documentation (.md)
- Test files (_spec.rb, _test.rb)
- Database migration files
- Environment configuration files (.env)

## Related Packages
- sinatra-contrib ^2.1.0
- rack-protection ^2.1.0
- rack ^2.2.0
- tilt ^2.0.10
- mustermann ^1.1.1
- activesupport ^6.0.0
- sinatra-activerecord ^2.0.23
- puma ^5.6.0
- thin ^1.8.0
- sequel ^5.49.0
- sqlite3 ^1.4.2
- pg ^1.2.3
- rspec ^3.10.0

## Differences From Previous Version
- **Dropped Support**:
  - Discontinued support for Ruby 2.2, now requiring Ruby 2.3+
  - Removed legacy features and deprecated methods
  
- **New Features**:
  - Improved view template handling with nested subdirectory support
  - Migrated from Thin to Rainbows for development streaming
  - Enhanced route pattern matching via updated Mustermann
  
- **Enhanced Features**:
  - Better error handling and reporting
  - Improved performance for common operations
  - Updated security protections through rack-protection 2.1.0

## Custom Instructions
When working with Sinatra 2.1.0, focus on leveraging its lightweight and flexible nature to build efficient web applications and APIs. Unlike more opinionated frameworks like Rails, Sinatra gives you precise control over your application's architecture and dependencies. Take advantage of its minimal approach for smaller projects, microservices, or APIs where Rails would be excessive. Structure your application using the modular style for anything beyond trivial applications to maintain organization as your codebase grows. Use sinatra-contrib extensions to add common functionality without reinventing the wheel - particularly sinatra/reloader during development for automatic code reloading. When designing APIs, utilize Sinatra's intuitive routing DSL to create clean, RESTful endpoints that follow best practices. For database access, pair Sinatra with a lightweight ORM like Sequel or use ActiveRecord through sinatra-activerecord if you prefer that ecosystem. Remember that Sinatra provides the basics but delegates many decisions to you as the developer, so carefully select complementary libraries for features like authentication, form validation, and more complex requirements. For deployment, leverage Sinatra's compatibility with standard Rack servers like Puma or Unicorn in production environments, using config.ru for configuration.