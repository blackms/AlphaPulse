# Echo v4.11.3 Developer Mode

## Version-Specific Features
- **High-Performance HTTP Router** - Optimized router with smart prioritized route matching
- **Middleware System** - Extensible middleware architecture with both global and route-specific middleware
- **Data Binding and Validation** - Automatic request data binding to Go structs with validation support
- **Context-Based Request Handling** - Context objects for efficient request and response management
- **Template Rendering** - Flexible HTML template rendering with multiple engine support
- **HTTP/2 Support** - Full support for HTTP/2 protocol with TLS
- **Auto TLS with Let's Encrypt** - Built-in Let's Encrypt integration for automatic TLS certificate management
- **Static File Serving** - Efficient static file serving with customizable options
- **WebSocket Support** - Native WebSocket capabilities for real-time applications
- **Highly Customizable Error Handling** - Centralized HTTP error handling with custom error pages

## Key Skills and Expertise
- **Go Programming** with focus on concurrency patterns
- **HTTP Protocol** fundamentals and best practices
- **RESTful API Design** principles and implementation
- **Middleware Development** for cross-cutting concerns
- **JSON and API Serialization** techniques
- **Authentication Mechanisms** including JWT and OAuth
- **Template Processing** for server-side rendering
- **Database Integration** with various datastores
- **Testing Web Applications** using Go's testing tools
- **Performance Optimization** for high-traffic applications

## Best Practices
- Organize routes using groups for logical API structuring
- Implement proper middleware chains for request processing
- Use context methods for accessing request data and setting responses
- Leverage the Echo binder for automatic request payload binding
- Implement custom validators for complex validation requirements
- Handle errors consistently using Echo's error handling system
- Use route naming for reverse route generation
- Implement proper logging for production monitoring
- Structure applications using clean architecture principles
- Create comprehensive tests using Echo's testing utilities

## File Types
- Go source files (.go)
- Go module files (go.mod, go.sum)
- HTML templates (.html, .tmpl)
- Static assets (.css, .js, .jpg, .png)
- JavaScript files (.js)
- Configuration files (.json, .yaml, .toml)
- SQL migration files (.sql)
- Markdown documentation (.md)
- Test files (_test.go)
- Environment configuration files (.env)

## Related Packages
- labstack/echo ^4.11.3
- go-playground/validator ^10.15.0
- golang-jwt/jwt ^5.0.0
- go-gorm/gorm ^1.25.0
- go-redis/redis ^8.11.5
- jackc/pgx ^5.4.3
- go-sql-driver/mysql ^1.7.1
- spf13/viper ^1.16.0
- rs/zerolog ^1.30.0
- stretchr/testify ^1.8.4

## Differences From Previous Version
- **New APIs**:
  - Enhanced binding mechanism for improved type handling
  - Additional middleware components for common use cases
  - Extended context methods for better request handling
  
- **Enhanced Features**:
  - Improved router performance and route matching
  - Better error handling with more detailed error reporting
  - Enhanced WebSocket support with additional configuration options
  - Expanded middleware ecosystem with security-focused additions

## Custom Instructions
When working with Echo v4.11.3, focus on leveraging its minimalistic yet powerful design for building high-performance web applications and APIs. Echo provides a clean, idiomatic API while maintaining excellent performance characteristics. Structure your application using Echo's router and group functionality to create well-organized API endpoints. Take advantage of the Context object for efficient request and response handling, using its methods to access request data and set responses appropriately. For data processing, use Echo's binding system to automatically map request payloads to Go structs, and implement validation using validator tags. Implement middleware for cross-cutting concerns like authentication, logging, and request timing, taking advantage of Echo's ability to apply middleware globally or to specific routes. For error handling, use Echo's centralized error handler to maintain consistent error responses across your application. When developing JSON APIs, use Echo's built-in JSON handling for efficient serialization and deserialization. For more complex applications, consider structuring your code using clean architecture principles, separating your business logic from the Echo-specific code to maintain testability and flexibility. When deploying, take advantage of Echo's HTTP/2 and TLS support for secure, high-performance communications.