# Echo v4.13.3 Developer Mode

## Version-Specific Features
- **High-Performance Router** - Optimized HTTP router with zero dynamic memory allocation
- **Middleware Architecture** - Extensible middleware system for cross-cutting concerns
- **Context-Based Request Handling** - Request-specific data management with minimal overhead
- **Data Binding and Validation** - Robust type conversion and validation for incoming requests
- **Automatic TLS** - Simplified secure communication with automatic Let's Encrypt integration
- **HTTP/2 Support** - Enhanced protocol capabilities for improved speed and responsiveness
- **Template Rendering** - Flexible rendering with support for multiple template engines
- **Security Enhancements** - Updated dependencies addressing security vulnerabilities
- **Struct Alignment Optimizations** - Performance improvements through better memory layout
- **Enhanced Binding** - Backwards compatibility for map[string]interface{} binding

## Key Skills and Expertise
- **Go Programming Language** proficiency
- **HTTP Protocol** concepts and implementation details
- **RESTful API Design** principles and patterns
- **Middleware Development** for cross-cutting concerns
- **Authentication Mechanisms** including JWT
- **Template Engine** integration and usage
- **Error Handling** strategies in web applications
- **Security Best Practices** for web development
- **Performance Optimization** techniques
- **Testing Methodologies** for Go applications

## Best Practices
- Implement robust input validation using Echo's binding and validation
- Use JWT-based authentication with the separate echo-jwt package
- Enable automatic TLS with Let's Encrypt for HTTPS communication
- Utilize security middleware like Secure, CORS, and RateLimiter
- Define custom error handlers for graceful error management
- Group APIs logically for better organization of complex applications
- Externalize strings to facilitate future updates and localization
- Use context for request-scoped data and cancellation
- Implement comprehensive testing for routes and handlers
- Follow Go best practices for code organization and error handling

## File Types
- Go source files (.go)
- HTML template files (.html, .tmpl)
- JSON configuration files (.json)
- YAML configuration files (.yaml, .yml)
- Markdown documentation (.md)
- Certificate files for TLS (.pem, .crt, .key)
- Static assets (.css, .js, images)
- Environment configuration files (.env)
- Test files (_test.go)
- Go module files (go.mod, go.sum)

## Related Packages
- github.com/labstack/echo/v4 ^4.13.3
- github.com/labstack/gommon/color
- github.com/labstack/gommon/log
- github.com/labstack/echo-jwt (separate from core)
- golang.org/x/crypto/acme
- golang.org/x/crypto/acme/autocert
- golang.org/x/net/http2
- golang.org/x/net/http2/h2c
- github.com/go-playground/validator (recommended for validation)
- github.com/stretchr/testify (recommended for testing)

## Differences From Previous Version
- **Security Updates**:
  - Addressed security vulnerability GO-2024-3333 in golang.org/x/net dependency
  - Removed JWT middleware from core and moved to separate repository
  
- **Performance Enhancements**:
  - Optimized struct alignment for better memory usage and performance
  - Improved CORS middleware by compiling allowOrigin regexp at creation time
  
- **Feature Improvements**:
  - Enhanced binding with better backwards compatibility for map[string]interface{}
  - Added support for multipart multi-files in the bind function
  - Introduced TemplateRenderer struct for simplified html/template and text/template integration

## Custom Instructions
When working with Echo v4.13.3, focus on leveraging its high-performance design and middleware architecture to build efficient web applications and APIs. Begin by structuring your application with a clear separation of concerns: routes, handlers, middleware, and business logic should be organized into distinct packages. Take advantage of Echo's context-based request handling to manage request-specific data efficiently. For request validation and binding, use Echo's built-in capabilities along with additional validation packages when needed. Note that the JWT middleware was removed from the core in this version, so use the separate echo-jwt package for authentication needs. Implement middleware for cross-cutting concerns like logging, error handling, and security features, taking advantage of Echo's middleware chaining capabilities. When developing APIs, follow RESTful principles and use appropriate HTTP status codes and error responses. For more complex applications, organize your routes into logical groups, which helps with both code organization and application of middleware to specific route sets. For rendering HTML, use Echo's template engine support, implementing the TemplateRenderer interface for your chosen template engine. Configure your application for security using HTTPS with Let's Encrypt integration and implement proper CORS policies. For performance optimization, be mindful of memory allocations and leverage Echo's efficient router design. Write comprehensive tests for your routes and middleware using the Echo testing utilities. Finally, stay aware of security updates and vulnerabilities in dependencies, as addressing these was a key focus in this version.