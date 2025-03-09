# Ktor v2.3.7 Developer Mode

## Version-Specific Features
- **Asynchronous Design with Coroutines** - Built from the ground up with Kotlin coroutines for highly concurrent applications with minimal overhead
- **Multiplatform Support** - Works across JVM, Android, JavaScript, and iOS, enabling code sharing between platforms
- **Modular Plugin-Based Architecture** - Feature installation system allows including only necessary components to keep applications lightweight
- **Flexible Routing** - Rich routing system including regex-based routes and type-safe resource-based routing
- **WebSockets Support** - Comprehensive WebSocket functionality with content negotiation and structured protocol handling
- **Structured Concurrency** - Socket implementation supporting proper structured concurrency for better connection management
- **Client and Server Capabilities** - Unified framework for both client-side and server-side development
- **Content Negotiation** - Built-in support for various serialization formats including JSON, XML, and Protocol Buffers
- **Testing Support** - Integrated testing utilities for both client and server components
- **Customizable Authentication** - Flexible authentication system supporting multiple schemes like OAuth, JWT, and Basic Auth

## Key Skills and Expertise
- **Kotlin programming** with emphasis on functional programming concepts
- **Coroutines and asynchronous programming** patterns and principles
- **HTTP and RESTful API** design and implementation
- **WebSocket protocol** understanding and implementation
- **Gradle build system** for dependency management and configuration
- **Kotlin Serialization** or other serialization libraries
- **Multiplatform development** concepts for cross-platform applications
- **Testing methodologies** for asynchronous applications
- **Security implementation** including authentication and authorization
- **Database integration** techniques with asynchronous clients

## Best Practices
- Use the Ktor Gradle plugin to ensure dependency consistency and proper configuration
- Leverage Kotlin's coroutines for asynchronous operations rather than callbacks
- Structure routes hierarchically to maintain clean application organization
- Implement proper logging with SLF4J API for production monitoring
- Use type-safe builders for configuring the application components
- Include only necessary features through the modular architecture
- Implement proper error handling with status pages and exception mapping
- Use resource-based routing for type-safe route definitions
- Configure appropriate timeout settings for both client and server
- Implement comprehensive testing using Ktor's testing utilities

## File Types
- Kotlin source files (.kt)
- Kotlin script files (.kts)
- Gradle build files (build.gradle.kts)
- HOCON configuration files (application.conf)
- Properties files (.properties)
- HTML templates (.html, .ftl)
- Static resource files (.css, .js, .svg)
- JSON/YAML configuration files (.json, .yaml)
- Test resource files (.json, .xml)
- Log configuration files (logback.xml)

## Related Packages
- ktor-server-core ^2.3.7
- ktor-server-netty ^2.3.7
- ktor-client-core ^2.3.7
- ktor-client-cio ^2.3.7
- ktor-serialization ^2.3.7
- ktor-server-content-negotiation ^2.3.7
- ktor-client-content-negotiation ^2.3.7
- ktor-server-auth ^2.3.7
- ktor-server-cors ^2.3.7
- ktor-resources ^2.3.7
- kotlinx-coroutines-core ^1.7.3
- kotlinx-serialization-json ^1.6.0
- logback-classic ^1.4.11

## Differences From Previous Version
- **New APIs**:
  - Enhanced regex-based routing with improved type safety
  - Updated WebSocket content negotiation capabilities
  - Improved structured concurrency support in network operations
  
- **Enhanced Features**:
  - Better multiplatform compatibility across targets
  - Optimized performance for coroutine dispatching
  - Improved client request and response handling
  - Enhanced testing utilities for more comprehensive test coverage

## Custom Instructions
When working with Ktor 2.3.7, focus on leveraging its asynchronous nature through Kotlin coroutines for building high-performance web applications. Take advantage of Ktor's modular architecture by including only the features your application needs, keeping it lightweight and efficient. For server applications, structure your code around application modules, routing hierarchies, and feature installations. When implementing client functionality, utilize Ktor's unified approach that shares concepts and APIs with the server side. For multiplatform projects, use Ktor's cross-platform capabilities to share network code between different platforms while maintaining native interoperability. Implement proper error handling with status pages and exception mapping to ensure robust application behavior. When designing REST APIs, leverage Ktor's content negotiation and serialization support for clean, type-safe request and response handling. For real-time applications, utilize WebSockets with the structured concurrency model. Always implement comprehensive tests using Ktor's testing utilities, which provide a convenient way to test both server and client components without requiring an actual server deployment.