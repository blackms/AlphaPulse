# Ktor v3.1.1 Developer Mode

## Version-Specific Features
- **Server-Sent Events (SSE)** - Enhanced SSE plugin for both client and server with multi-engine support
- **Improved WebSockets** - Added WebSocket support in Curl engine and timeout configuration
- **Enhanced Multipart Handling** - Support for receiving multipart data with Ktor client
- **Expanded Platform Support** - CIO server support for WasmJS and JS targets
- **Unix Domain Socket** - Support for Native targets
- **Improved Logging** - Enhanced format similar to OkHttp with per-line messages
- **Compression Enhancements** - Support for compression of request body with ContentEncoding client plugin
- **Asynchronous Server** - Non-blocking architecture based on Kotlin coroutines
- **Modular Plugin System** - Extensible framework through application features
- **Multiplatform Client** - HTTP client with cross-platform support including iOS, Android, JVM, and JavaScript

## Key Skills and Expertise
- **Kotlin Programming** with coroutines and flow
- **Asynchronous Programming** patterns and concepts
- **HTTP Protocol** including headers, status codes, and content negotiation
- **RESTful API Design** principles and implementation
- **JSON Serialization** with kotlinx.serialization
- **WebSocket Communication** for real-time applications
- **Server-Sent Events** for push notifications
- **Multiplatform Development** across various targets
- **Authentication and Security** implementation
- **Testing Asynchronous APIs** with specific tools

## Best Practices
- Use the Ktor Gradle plugin to ensure dependency version consistency
- Implement proper logging using SLF4J API with a suitable logging framework
- Structure your applications using feature-based architecture
- Utilize extension functions to enhance maintainability
- Take advantage of the expanded platform support for cross-platform development
- Use the kotlinx-io library for efficient IO operations
- Implement proper error handling with status pages
- Leverage content negotiation for flexible API responses
- Use appropriate coroutine scopes for resource management
- Configure proper timeout settings for WebSockets and SSE

## File Types
- Kotlin source files (.kt)
- Kotlin script files (.kts) for build configuration
- Configuration files (.conf, application.conf)
- Resource files (.properties, .json, .xml)
- Template files (if using server-side rendering)
- Static content (.html, .css, .js)
- Test files (_test.kt)
- Certificate files (.pem, .jks) for HTTPS
- Build output files (.jar, .klib)
- Docker configuration files (Dockerfile, docker-compose.yml)

## Related Packages
- ktor-client-core ^3.1.1
- ktor-client-cio ^3.1.1
- ktor-server-core-jvm ^3.1.1
- ktor-server-netty-jvm ^3.1.1
- kotlinx-coroutines-core ^1.7.3
- kotlinx-serialization-json ^1.6.0
- kotlinx-io (version compatible with Ktor 3.1.1)
- logback-classic ^1.4.11
- Kotlin ^2.0.0
- ktor-server-core-jvm ^3.1.1
- ktor-server-netty-jvm ^3.1.1
- ktor-serialization ^3.1.1

## Differences From Previous Version
- **Kotlin 2.0 Support**:
  - Utilizes features from the latest Kotlin 2.0 release
  - Enhanced type system and compiler capabilities
  
- **Performance Improvements**:
  - Integration with kotlinx-io library for more efficient IO operations
  - Reduced unnecessary copying of bytes between channels
  - Significant performance enhancements in some benchmarks
  
- **API Changes**:
  - Deprecated some low-level IO APIs in favor of kotlinx-io alternatives
  - New APIs for Server-Sent Events and other features
  - Enhanced WebSocket and SSE now respect connection timeout settings
  
- **Platform Expansion**:
  - Added WasmJS target support
  - Improved ARM target support in Ktor client with Kotlin/Native and Curl
  - NodeJs target support in ktor-network

## Custom Instructions
When working with Ktor 3.1.1, focus on leveraging its asynchronous, non-blocking architecture built on Kotlin coroutines to create high-performance web applications and HTTP clients. This version introduces significant improvements in Server-Sent Events (SSE) and WebSocket support, making it particularly well-suited for real-time applications. Take advantage of the expanded platform support, which now includes WasmJS, JS targets with the CIO server, and ARM targets with the Curl client engine - enabling truly cross-platform development from a single codebase. Structure your application using Ktor's plugin-based architecture, which allows for modular and maintainable code. For efficient data serialization, use kotlinx.serialization which integrates seamlessly with Ktor's content negotiation feature. When implementing clients, leverage the enhanced multipart handling and compression capabilities for request bodies. Pay special attention to proper resource management using structured concurrency patterns with coroutines. For logging, use the improved logging format which now prints messages per line in a format similar to OkHttp. If you're upgrading from Ktor 2.x, be aware of the shift to kotlinx-io for IO operations, which improves performance but may require some API adjustments in your codebase. For testing, use Ktor's built-in test utilities which support testing both server and client code without requiring a running server. When deploying, consider using Ktor's application configuration capabilities to externalize environment-specific settings.