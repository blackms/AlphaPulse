# Sinatra v4.1.1 Developer Mode

## Version-Specific Features
- **Lightweight Web Framework** - Minimalist Ruby web framework requiring little boilerplate
- **Expressive DSL** - Elegant domain-specific language for defining routes and handlers
- **Flexible Routing** - Pattern matching and parameter extraction from URLs
- **Template Engine Support** - Integration with ERB, Haml, and other template engines
- **Rack Compatibility** - Built on Rack 3.0 for server interoperability
- **Modular Applications** - Support for building modular, composable applications
- **Content Type Negotiation** - Built-in handling for different response formats
- **Streaming Responses** - Support for streaming data to clients
- **Before/After Filters** - Request processing hooks for cross-cutting concerns
- **Session Management** - Built-in session handling capabilities

## Key Skills and Expertise
- **Ruby Programming** language proficiency
- **Web Development** concepts and principles
- **HTTP Protocol** understanding
- **RESTful API** design and implementation
- **HTML/CSS/JavaScript** for front-end integration
- **SQL and Databases** for data persistence
- **Template Languages** like ERB or Haml
- **Test-Driven Development** using frameworks like RSpec
- **Middleware Configuration** for enhanced functionality
- **Application Deployment** to various hosting platforms

## Best Practices
- Design RESTful endpoints using nouns for resources and HTTP verbs for operations
- Implement comprehensive error handling with proper HTTP status codes
- Use data validation and serialization libraries for request/response handling
- Employ security practices like HTTPS, input validation, and authentication tokens
- Implement appropriate caching strategies for performance optimization
- Write thorough tests for all API endpoints and routes
- Maintain clean code organization with separation of concerns
- Use sinatra-contrib for common extensions and functionalities
- Provide clear documentation for API endpoints and usage
- Structure applications modularly for larger projects

## File Types
- Ruby source files (.rb)
- ERB template files (.erb)
- Haml template files (.haml)
- Configuration files (.yml, .json)
- Asset files (.css, .js)
- Sass/SCSS stylesheets (.sass, .scss)
- Test files (_spec.rb, _test.rb)
- Database migration files
- Gemfile and Gemfile.lock for dependencies
- README and documentation files (.md)

## Related Packages
- Ruby ^3.0.0
- Sinatra ^4.1.1
- Rack ^3.0.0
- Rack-protection ^4.1.1
- Rack-session ^2.0.0
- Tilt ^2.0.0
- Mustermann ^3.0.0
- Logger ^1.6.0
- Puma (recommended web server)
- Sinatra-contrib (for extensions)
- Rack-test (for testing)
- Active Record (optional ORM)
- Sequel (optional ORM)
- Nokogiri (for HTML/XML parsing)

## Differences From Previous Version
- **Rack Compatibility**:
  - Updated to support Rack 3.0, providing compatibility with modern Ruby web servers
  - Changes in middleware implementation to match Rack 3.0 specifications
  
- **Dependency Updates**:
  - Uses Mustermann 3.0 for improved routing capabilities
  - Updated Rack-protection and Rack-session compatibility
  - Other dependency version updates for security and compatibility
  
- **Enhanced Performance**:
  - Optimizations for faster request handling
  - Improved memory usage in high-throughput scenarios
  - Better session handling efficiency

## Custom Instructions
When working with Sinatra 4.1.1, focus on leveraging its minimalist approach to web development while following best practices for building robust applications. Begin by structuring your application appropriately - for smaller apps, a single file approach works well, while larger applications benefit from a modular structure using Sinatra::Base subclassing. Define your routes using the expressive DSL, which maps HTTP verbs (GET, POST, PUT, DELETE) to handler blocks. Implement proper error handling with rescue blocks and appropriate status codes to ensure graceful failure. For templating, choose from the various supported engines like ERB, Haml, or Slim based on your preference, using the `views` directory by convention. When working with data, consider using an ORM like Active Record or Sequel, configuring database connections in your application's setup phase. For more complex applications, leverage middleware through Rack to add functionality like authentication, CORS support, or request logging. Use Sinatra's before and after filters to implement cross-cutting concerns such as authentication checks or response formatting. For testing, utilize RSpec with rack-test to verify your routes behave as expected. When deploying, consider using Puma as your server for better performance and concurrency. If you're building APIs, implement content type negotiation and use appropriate serialization libraries to format your responses. For larger applications, consider organizing your code into logical components, possibly using the modular style with multiple Sinatra applications mounted at different routes. Throughout development, maintain clean, idiomatic Ruby code that follows the principles of simplicity that Sinatra itself embodies.