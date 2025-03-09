# Gin v1.9.1 Developer Mode

## Version-Specific Features
- **High-Performance Routing** - Custom version of HttpRouter offering exceptional routing speed
- **Middleware Support** - Flexible middleware chaining for request processing
- **Panic Recovery** - Automatic recovery from panics to ensure server availability
- **JSON Validation** - Built-in JSON parsing and validation capabilities
- **Route Grouping** - Support for organizing and nesting route groups
- **Error Management** - Convenient error collection during request processing
- **Flexible Rendering** - Easy-to-use API for JSON, XML, and HTML responses
- **Bytedance/Sonic Integration** - High-performance JSON serializer/deserializer
- **HTTP Status Constants** - Predefined HTTP status codes for clearer code
- **Custom Middleware Creation** - Simple API for extending functionality

## Key Skills and Expertise
- **Go Programming Language** proficiency
- **HTTP Protocol** understanding
- **RESTful API Design** principles
- **Middleware Concepts** for request/response processing
- **JSON and Data Serialization** formats
- **HTML Templating** for web applications
- **Concurrent Programming** in Go
- **Database Integration** (SQL or NoSQL)
- **Testing Methodologies** for Go applications
- **API Documentation** tools and practices

## Best Practices
- Organize project with clear structure for maintainability
- Use RESTful conventions for endpoint naming and HTTP methods
- Utilize middleware for cross-cutting concerns like logging and authentication
- Implement consistent error handling across the application
- Write comprehensive unit and integration tests
- Use gin.New() instead of gin.Default() for production to control middleware
- Leverage Gin's binding capabilities for request validation
- Group related routes for better organization
- Use context for request-scoped values and cancellation
- Implement proper logging for debugging and monitoring

## File Types
- Go source files (.go)
- HTML template files (.html, .tmpl)
- JSON configuration files (.json)
- YAML configuration files (.yaml, .yml)
- Go module files (go.mod, go.sum)
- Static asset files (.css, .js, images)
- Markdown documentation (.md)
- Swagger specification files (.json, .yaml)
- Test files (_test.go)
- Environment configuration files (.env)

## Related Packages
- github.com/gin-gonic/gin ^1.9.1
- github.com/go-playground/validator/v10 (for validation)
- github.com/json-iterator/go (for JSON handling)
- github.com/bytedance/sonic (JSON serializer)
- github.com/swaggo/gin-swagger (API documentation)
- github.com/gin-contrib/* (official extensions)
- golang.org/x/crypto (for secure operations)
- github.com/golang-jwt/jwt (for JWT authentication)
- github.com/jinzhu/gorm or github.com/go-gorm/gorm (for ORM)
- github.com/stretchr/testify (for testing)

## Differences From Previous Version
- **Enhanced JSON Handling**:
  - Integration of bytedance/sonic as JSON serializer/deserializer for better performance
  - Improved JSON binding capabilities
  
- **Route Changes**:
  - Potential modifications in route matching behavior
  - Updated internal HTTP router implementation
  
- **New Utility Functions**:
  - Addition of ShouldBindBodyWith shortcut for binding
  - Support for custom BindUnmarshaler for binding
  - New OptionFunc and With for flexible configuration
  
- **Compatibility Updates**:
  - Support for Go 1.20's http.rwUnwrapper in responseWriter
  - Updated dependencies for better security and performance

## Custom Instructions
When working with Gin 1.9.1, focus on leveraging its high-performance routing and middleware capabilities to build efficient web services. Start by structuring your application with a clear organization - separating routes, handlers, middleware, and models into distinct packages. Use Gin's router to define your API endpoints following RESTful conventions, utilizing appropriate HTTP methods for different operations (GET for retrieval, POST for creation, etc.). Take advantage of route groups to organize related endpoints and apply common middleware to specific groups. Implement custom middleware for cross-cutting concerns like authentication, logging, and error handling, using Gin's context to pass data between middleware and handlers. For request validation, use Gin's binding features with struct tags to automatically validate incoming JSON, form data, or query parameters. When responding to clients, utilize the appropriate rendering functions (JSON, XML, HTML) based on your application's needs. For error handling, establish a consistent approach using Gin's error collection capabilities and HTTP status codes. To optimize performance, consider using the bytedance/sonic JSON serializer which was introduced in this version for faster JSON processing. When deploying to production, remember to use gin.New() instead of gin.Default() to have explicit control over which middleware is included. For larger applications, implement a dependency injection pattern to manage service dependencies and facilitate testing. Finally, thoroughly test your application, both with unit tests for individual components and integration tests for API endpoints, using Go's testing package alongside Gin's testing utilities.