# Fiber v2.52.0 Developer Mode

## Version-Specific Features
- **Express-Inspired API** - Familiar syntax for Node.js developers transitioning to Go
- **Fasthttp Foundation** - Built on top of Fasthttp for exceptional performance
- **Zero Memory Allocation** - Designed to minimize garbage collection overhead
- **Robust Routing System** - Support for parameters, grouping, and nested routes
- **Middleware Architecture** - Flexible middleware system for request processing
- **Static File Serving** - Optimized handling of static assets
- **Template Engine Support** - Compatible with various template engines
- **HTTP/2 Support** - Modern protocol features for improved performance
- **WebSocket Support** - Real-time bidirectional communication capabilities
- **Rate Limiting** - Built-in request throttling for API protection

## Key Skills and Expertise
- **Go Programming Language** proficiency
- **HTTP Protocol** understanding
- **RESTful API Design** principles and implementation
- **Middleware Development** for cross-cutting concerns
- **Performance Optimization** techniques
- **Concurrent Programming** in Go
- **Database Integration** (SQL and NoSQL)
- **Template Processing** for dynamic content
- **Error Handling** strategies
- **Testing Methodologies** for Go applications

## Best Practices
- Use Fiber's built-in methods for common HTTP operations
- Implement comprehensive error handling
- Utilize middleware for cross-cutting concerns like logging and authentication
- Group related routes for better organization
- Leverage Fiber's Config struct for framework customization
- Use context methods for request and response handling
- Implement proper validation for incoming requests
- Set appropriate file size limits for uploads
- Follow Go idioms and conventions
- Write comprehensive tests for routes and handlers

## File Types
- Go source files (.go)
- Configuration files (.env, .yaml, .json)
- Template files (.html, .tmpl)
- Static assets (.css, .js, .png, .jpg)
- Go module files (go.mod, go.sum)
- Environment configuration (.env)
- Documentation files (.md)
- Unit test files (_test.go)
- Database migration files
- API specification files (OpenAPI/Swagger)

## Related Packages
- github.com/gofiber/fiber/v2 ^2.52.0
- github.com/valyala/fasthttp ^1.51.0
- github.com/gofiber/template ^1.8.2
- github.com/gofiber/utils ^1.1.0
- github.com/gofiber/middleware/* (various middleware modules)
- github.com/gofiber/jwt/v2 (for JWT authentication)
- github.com/gofiber/storage/* (database integrations)
- github.com/gofiber/websocket/v2 (WebSocket support)
- github.com/gofiber/helmet/v2 (security headers)
- github.com/gofiber/swagger (API documentation)

## Differences From Previous Version
- **Performance Improvements**:
  - Enhanced routing efficiency
  - Optimized memory usage patterns
  - Improved request handling performance
  
- **Bug Fixes**:
  - Resolution of issues from previous versions
  - Enhanced stability for production environments
  
- **Updates**:
  - Dependency upgrades to latest compatible versions
  - Minor API enhancements for developer convenience
  - Improved error messages and debugging information

## Custom Instructions
When working with Fiber v2.52.0, focus on leveraging its high-performance capabilities built on the Fasthttp foundation. Organize your project with a clear structure, typically following the standard Go project layout with dedicated packages for handlers, middleware, models, and utilities. Define your application routes using Fiber's intuitive API, which should feel familiar if you have experience with Express.js. For handling common web development tasks, use Fiber's built-in methods for routing, middleware application, and response generation. Implement middleware for cross-cutting concerns like logging, authentication, and error handling, taking advantage of Fiber's middleware chaining capabilities. When processing requests, utilize Fiber's context methods for accessing request data, validating input, and sending responses in various formats including JSON, XML, or HTML. For file operations, leverage Fiber's optimized static file serving and file upload handling with proper size limits and validation. When developing APIs, follow RESTful principles and use appropriate HTTP status codes. For template rendering, configure one of the supported template engines through the template middleware. Enhance your application's security with Fiber's security-focused middleware like helmet, CORS, and CSRF protection. Take advantage of Fiber's websocket support for real-time applications. For database operations, use one of the many Go database packages alongside Fiber, structuring your data access layer for maintainability. Throughout development, focus on writing idiomatic Go code with proper error handling and leverage Go's concurrency features when appropriate, while being mindful of Fiber's design goal of minimizing memory allocations for optimal performance.