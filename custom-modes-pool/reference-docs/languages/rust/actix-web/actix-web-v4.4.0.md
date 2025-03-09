# Actix Web v4.4.0 Developer Mode

## Version-Specific Features
- **Actor System Foundation** - Built on the Actix actor framework for concurrent request processing
- **Async/Await Support** - Full support for Rust's async/await syntax for efficient asynchronous handling
- **Resource Routing** - Flexible routing system with pattern matching and parameter extraction
- **Middleware System** - Powerful middleware framework for request/response processing
- **Extractor Pattern** - Type-safe request data extraction using Rust's type system
- **WebSocket Support** - First-class support for WebSocket connections
- **HTTP/2 Support** - Full HTTP/2 protocol implementation
- **Static File Serving** - Efficient static asset handling with caching support
- **Responder Trait** - Flexible response generation through the Responder trait
- **TLS/SSL Support** - Secure communication with rustls and native-tls options

## Key Skills and Expertise
- **Rust Programming** with strong understanding of ownership and borrowing
- **Asynchronous Programming** using async/await and Futures
- **HTTP Protocol** knowledge for web application development
- **RESTful API Design** principles and implementation
- **Middleware Development** for cross-cutting concerns
- **JSON Processing** with serde for serialization/deserialization
- **Database Integration** particularly with SQLx, Diesel, or r2d2
- **Error Handling** using Rust's Result type and error patterns
- **Testing Web Applications** with Actix's testing utilities
- **Security Implementation** including authentication and authorization

## Best Practices
- Leverage the extractor pattern for type-safe request handling
- Implement proper error handling with custom error types
- Use app data for sharing state between handlers
- Structure larger applications with scoped services
- Implement middleware for cross-cutting concerns
- Use async handlers for efficient request processing
- Take advantage of the Responder trait for flexible responses
- Organize routes using scopes for logical API structure
- Implement comprehensive testing using Actix's test utilities
- Use proper logging for production monitoring

## File Types
- Rust source files (.rs)
- Cargo configuration (Cargo.toml)
- Rust module files (mod.rs)
- HTML templates (.html, .hbs)
- Static assets (.css, .js, .jpg, .png)
- Configuration files (.toml, .yaml, .json)
- Environment files (.env)
- SQL migration files (.sql)
- Markdown documentation (.md)
- Test files (with #[cfg(test)] modules)

## Related Packages
- actix-web ^4.4.0
- actix-rt ^2.9.0
- tokio ^1.32.0
- serde ^1.0.188
- serde_json ^1.0.107
- sqlx ^0.7.2
- diesel ^2.1.0
- r2d2 ^0.8.10
- jsonwebtoken ^9.1.0
- env_logger ^0.10.0
- actix-files ^0.6.2
- actix-cors ^0.6.4
- uuid ^1.5.0

## Differences From Previous Version
- **New APIs**:
  - Enhanced extractors for more efficient request data handling
  - Improved middleware API for better composability
  - New test utilities for more comprehensive testing
  
- **Enhanced Features**:
  - Better performance for routing and request handling
  - Improved error messages and diagnostics
  - More efficient WebSocket handling
  - Enhanced static file serving with better caching options

## Custom Instructions
When working with Actix Web 4.4.0, focus on leveraging its high-performance asynchronous architecture based on Rust's async/await capabilities. Actix Web excels at building fast, reliable web services and APIs with strong type safety. Structure your application using the HttpServer and App builders, organizing routes logically with scopes and resource definitions. Take advantage of Actix's extractor pattern to safely parse and validate request data using Rust's type system - this is one of the framework's most powerful features. For handling application state, use app_data to share data between handlers, ensuring thread safety with appropriate synchronization primitives like Arc and Mutex. Implement custom error handling by defining error types that implement the ResponseError trait, allowing for consistent error responses across your API. When building middleware, consider both transform and service factory approaches depending on your needs. For database integration, pair Actix with async database libraries like SQLx or Diesel with r2d2 connection pools. Use the Responder trait effectively to create flexible response handlers that can return different content types based on application logic. When testing, leverage Actix's test utilities to create integration tests that verify your API's behavior without needing to run a full server. For production deployments, ensure proper configuration of workers and thread counts to maximize performance on modern hardware.