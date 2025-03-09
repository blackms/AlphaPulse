# Micronaut v4.7.6 Developer Mode

## Version-Specific Features
- **Java 17 Baseline** - Requires minimum Java 17 for building and running applications
- **GraalVM 23 Support** - Improved integration with GraalVM and native image compilation
- **Compile-time DI** - Dependency injection without reflection for fast startup
- **Expression Language** - Compilation-time, type-safe, and reflection-free EL implementation
- **Enhanced Configuration** - Arbitrary nesting of @ConfigurationProperties and @EachProperty annotations
- **Virtual Threads Support** - Preview support for Project Loom virtual threads
- **HTTP/3 Support** - Experimental implementation via Netty incubator project
- **LangChain4j Integration** - Enhanced capabilities for working with LLMs and AI technologies
- **Jakarta Migration** - Complete transition from javax to jakarta specifications
- **Improved Modularity** - Refactored core to reduce application footprint
- **Context Propagation API** - Better reactor instrumentation and Kotlin Coroutines integration

## Key Skills and Expertise
- **Java Programming** with Java 17+ features
- **Reactive Programming** concepts and implementation
- **Dependency Injection** principles and patterns
- **Microservices Architecture** design and implementation
- **Cloud-Native Development** with container-based deployments
- **GraalVM Native Image** compilation and optimization
- **Kotlin and/or Groovy** for polyglot development
- **Build Tools** (Gradle or Maven) configuration
- **Testing Microservices** with appropriate frameworks
- **Serverless Computing** concepts for cloud functions

## Best Practices
- Use the Micronaut Platform Catalog plugin for dependency management
- Leverage compile-time validation to catch errors early
- Design APIs with appropriate granularity for microservices
- Implement proper error handling with HTTP status codes
- Configure appropriate thread pools for reactive processing
- Utilize annotation-based HTTP filters for request/response handling
- Make use of configuration sharing for distributed applications
- Implement proper health checks and metrics for observability
- Design services to be stateless for better scaling
- Leverage type-safe client-server communication

## File Types
- Java source files (.java)
- Kotlin source files (.kt)
- Groovy source files (.groovy)
- YAML configuration files (application.yml)
- Properties files (application.properties)
- Build configuration files (build.gradle, pom.xml)
- JSON data files for API testing and documentation
- OpenAPI/Swagger specification files
- Dockerfile and container configuration files
- JUnit/Spock test files

## Related Packages
- Micronaut Core ^4.7.6
- Micronaut Data (latest compatible)
- Micronaut Security (latest compatible)
- Micronaut Logging ^1.5.1
- Micronaut Flyway ^7.6.1
- Micronaut Liquibase ^6.6.1
- Micronaut Oracle Cloud ^4.3.6
- Micronaut Pulsar ^2.5.1
- Kotlin ^1.8.0
- GraalVM ^23.0.0
- Apache Groovy ^4.0.0
- Jakarta EE API (compatible version)

## Differences From Previous Version
- **New APIs**:
  - Type-safe and reflection-free Expression Language
  - Context Propagation API for better async programming
  - Annotation-based HTTP filters for request/response handling
  
- **Enhanced Features**:
  - Migration from javax to jakarta packages
  - Improved modularity for reduced application footprint
  - Better error messages for missing configuration and beans
  - Enhanced configuration properties with arbitrary nesting
  - Kotlin Symbol Processing (KSP) support in addition to KAPT
  - Virtual Threads and HTTP/3 experimental support
  - Integration with LangChain4j for AI capabilities

## Custom Instructions
When working with Micronaut 4.7.6, focus on leveraging its key strengths: ultra-fast startup times, low memory footprint, and compile-time processing. This version requires Java 17 as a baseline and offers improved GraalVM 23 support, making it excellent for cloud-native and serverless applications. Take advantage of the new type-safe Expression Language that works at compilation-time without reflection, enabling more efficient evaluation of expressions. The framework's design emphasizes ahead-of-time (AOT) compilation to avoid traditional Java framework issues like slow startup and high memory consumption. Structure your application using dependency injection, but remember that Micronaut performs DI at compile-time rather than runtime, which improves performance but requires a different approach than Spring or other reflection-based frameworks. For configuration, utilize the enhanced @ConfigurationProperties and @EachProperty annotations which now support arbitrary nesting. When developing reactive applications, leverage the new Context Propagation API for better reactor instrumentation and Kotlin Coroutines integration. For HTTP services, consider experimenting with the new annotation-based HTTP filters for more granular control and the experimental HTTP/3 support. If working with AI functionality, explore the LangChain4j integration. Always use the Micronaut Platform Catalog plugin for dependency management to ensure compatibility across components. When preparing for production, take advantage of Micronaut's GraalVM support to compile your application to a native executable for even faster startup and lower memory usage, especially valuable in containerized and serverless environments.