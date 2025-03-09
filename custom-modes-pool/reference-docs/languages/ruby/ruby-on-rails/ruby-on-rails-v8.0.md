# Ruby on Rails v8.0 Developer Mode

## Version-Specific Features
- **Built-in Authentication** - Native authentication system for handling user sessions and password resets
- **Solid Trifecta** - Database-backed adapters for WebSockets, caching, and job queues
- **Native Rate Limiting** - Built-in request throttling capabilities for improved security
- **Propshaft Asset Pipeline** - New default asset pipeline replacing Sprockets
- **Kamal Deployment Integration** - Simplified deployment with Kamal 2
- **Active Record Enhancements** - Distinction between float types, advanced table options, and bulk inserts
- **MVC Architecture** - Structured application layout with models, views, and controllers
- **Convention over Configuration** - Standardized naming patterns to reduce boilerplate
- **RESTful Routing** - HTTP verb-based resource management
- **Strong Parameter Security** - Protection against mass assignment vulnerabilities

## Key Skills and Expertise
- **Ruby Programming Language** proficiency
- **MVC Architecture** implementation
- **Database Design** and SQL knowledge
- **HTML, CSS, and JavaScript** fundamentals
- **RESTful API** design and implementation
- **Testing Methodologies** (RSpec, Minitest)
- **Git Version Control** for code management
- **Web Security Concepts** for application protection
- **Object-Oriented Design** principles
- **Debugging and Troubleshooting** techniques

## Best Practices
- Follow Rails naming conventions for files, routes, and database tables
- Write clean, modular code with single-responsibility methods
- Implement appropriate caching strategies to improve performance
- Use eager loading to avoid N+1 query problems
- Write comprehensive tests covering critical application paths
- Keep dependencies updated for security and performance improvements
- Implement proper authorization alongside the new authentication system
- Use concerns and service objects for cross-cutting functionality
- Leverage the asset pipeline for efficient asset management
- Follow RESTful routing conventions for predictable API design

## File Types
- Ruby source files (.rb)
- ERB templates (.erb) for views
- YAML configuration files (.yml, .yaml)
- JavaScript files (.js)
- CSS and SCSS files (.css, .scss)
- JSON data files (.json)
- Markdown documentation (.md)
- Database migration files (.rb)
- Test files (_test.rb, _spec.rb)
- Initializer configuration (.rb in config/initializers)

## Related Packages
- Ruby ^2.7.0
- Bundler ^2.0
- SQLite3 ^1.4 (for development)
- Puma ^5.0 (web server)
- Propshaft ^1.0 (asset pipeline)
- Kamal ^2.0 (deployment)
- Minitest or RSpec (testing)
- Turbo (for frontend interactions)
- Stimulus (for JavaScript behaviors)
- Solid Cable, Solid Cache, Solid Queue (for the trifecta)

## Differences From Previous Version
- **New Features**:
  - Built-in authentication system replacing the need for Devise or other gems
  - Solid Trifecta components (Cable, Cache, Queue) reducing Redis dependency
  - Native rate limiting for request throttling
  - Propshaft as the default asset pipeline
  
- **Enhanced Capabilities**:
  - Improved SQLite support for production environments
  - Better Active Record with advanced PostgreSQL features
  - Tighter Kamal integration for simplified deployments
  - More efficient bulk operations in Active Record
  - Enhanced security measures and vulnerability fixes

## Custom Instructions
When working with Ruby on Rails 8.0, focus on leveraging its new built-in features to reduce external dependencies and streamline your application architecture. Take advantage of the native authentication system for user management, which gives you more control and visibility into the authentication process compared to third-party gems. For real-time features, explore the Solid Trifecta components: Solid Cable for WebSocket connections, Solid Cache for caching needs, and Solid Queue for background job processing, all of which provide database-backed alternatives to Redis-dependent solutions. When managing assets, utilize the new Propshaft asset pipeline which offers faster compilation and simpler configuration than Sprockets. For deployment, integrate with Kamal 2 for simplified containerized deployments to production. Structure your application following the MVC pattern and Rails conventions, utilizing models for data logic, controllers for request handling, and views for presentation. Optimize database performance by leveraging Active Record's enhanced features, including proper indexing, eager loading associations, and bulk operations where appropriate. Implement comprehensive testing using either Minitest (Rails' default) or RSpec, focusing on both unit and integration tests. For frontend interactivity, use the included Turbo for page updates and Stimulus for JavaScript behaviors. Secure your application using the built-in rate limiting capabilities to prevent brute force attacks and implement proper authorization alongside the new authentication system. Finally, follow Rails' "convention over configuration" philosophy to maximize productivity and maintain a clean, maintainable codebase.