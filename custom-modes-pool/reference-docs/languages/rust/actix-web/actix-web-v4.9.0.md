# Actix Web v4.9.0 Developer Mode

## Version-Specific Features
- **High-Performance** - Consistently ranked among the fastest web frameworks
- **HTTP/1.x and HTTP/2** - Full support for modern HTTP protocols
- **Async Runtime** - Built on Tokio for asynchronous request handling
- **Middleware System** - Extensible middleware architecture
- **Type-Safe Extractors** - Strong typing from request to response
- **Built-in Compression** - Transparent content compression (br, gzip, deflate, zstd)
- **WebSocket Support** - Client/server WebSocket capabilities
- **Multipart Handling** - Processing of multipart form data and file uploads
- **Static File Serving** - Efficient delivery of static assets
- **TLS Support** - SSL integration using OpenSSL or Rustls

## Key Skills and Expertise
- **Rust Programming** with strong typing and ownership model
- **Asynchronous Programming** using async/await syntax
- **HTTP Protocol** understanding and implementation
- **RESTful API Design** principles and best practices
- **Tokio Runtime** for asynchronous task execution
- **Concurrent Programming** patterns in Rust
- **Error Handling** with Rust's Result type
- **Web Security** concepts and implementation
- **JSON Serialization/Deserialization** with serde
- **Testing Asynchronous Code** with Rust's testing framework

## Best Practices
- Use async/await syntax for request handlers
- Leverage macros for routing when appropriate
- Utilize the App instance for registering request handlers
- Take advantage of HttpServer for serving incoming requests
- Implement proper error handling with custom error types
- Use middleware for cross-cutting concerns
- Leverage type-safe extractors for request data
- Apply proper resource management for connections
- Implement comprehensive testing
- Follow Rust's ownership and borrowing rules

## File Types
- Rust source files (.rs)
- Cargo configuration (.toml)
- JSON data files (.json)
- YAML configuration files (.yaml, .yml)
- Web assets (.html, .css, .js)
- Environment files (.env)
- Database migration files
- Static asset files (images, fonts)
- Template files for rendering
- Documentation files (.md)

## Related Packages
- actix-web ^4.9.0
- actix-rt (Actix runtime)
- actix-http (Low-level HTTP implementation)
- actix-service (Service traits)
- actix-router (URL routing)
- actix-codec (Codec utilities)
- actix-utils (Various utilities)
- tokio (Asynchronous runtime)
- serde, serde_json (Serialization/deserialization)
- futures (Asynchronous primitives)
- rustls or openssl (TLS support)

## Differences From Previous Version
- **New APIs**:
  - Added middleware::from_fn() helper function
  - Added web::ThinData extractor for lightweight data access
  
- **Performance Enhancements**:
  - Reduced memory usage in response streaming
  - Various optimizations throughout the codebase
  
- **Compatibility Updates**:
  - Minimum supported Rust version (MSRV) now 1.72
  - Updated dependencies to latest versions
  - Bug fixes and stability improvements

## Custom Instructions
When working with Actix Web 4.9.0, focus on leveraging its high-performance capabilities and type-safe design to build robust web applications and APIs. Begin by structuring your application with clear separation of concerns: routes, handlers, middleware, and business logic should be organized in distinct modules. For routing, use either the macro-based approach with #[get], #[post], etc., or the builder pattern with App::service() for more complex scenarios. Take advantage of Actix Web's powerful extractors to safely access and validate request data - Path for URL parameters, Query for query strings, Json for request bodies, and the new ThinData for lightweight data access. Implement error handling using Rust's Result type, paired with impl ResponseError for your custom error types to provide meaningful responses. For asynchronous operations, use Rust's async/await syntax, ensuring proper handling of futures and avoiding blocking the Tokio runtime. Leverage middleware for cross-cutting concerns like authentication, logging, and CORS, using the new middleware::from_fn() helper for simple custom middleware. For improved performance, utilize the reduced memory usage in response streaming when handling large responses or file downloads. When deploying, consider using HttpServer's workers() method to take advantage of multiple cores. Remember to follow Rust's ownership and borrowing rules carefully, especially when sharing state between request handlers. For testing, use actix-web's testing utilities to write comprehensive integration tests that verify your API's behavior. Finally, ensure your application is compatible with Rust 1.72 or newer, as this is the minimum supported version for Actix Web 4.9.0.