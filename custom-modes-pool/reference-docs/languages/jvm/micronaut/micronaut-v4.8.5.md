# Micronaut v4.8.5 Developer Mode

## Version-Specific Features
- **Compile-time dependency injection** - Eliminates runtime reflection for significantly faster startup and reduced memory footprint
- **GraalVM native image support** - First-class support for AOT compilation to native executables with optimized performance
- **Reactive HTTP server and client** - Built on Netty for efficient non-blocking I/O and high concurrency
- **Cloud-native capabilities** - Built-in service discovery, distributed configuration, and tracing
- **Serverless function support** - Optimized for AWS Lambda, Azure Functions, and Google Cloud Functions
- **Minimal memory footprint** - Designed for microservices with efficient resource utilization
- **Micronaut Data** - Compile-time repository APIs for SQL and NoSQL databases
- **Micronaut Security** - Comprehensive authentication and authorization framework
- **OpenAPI documentation** - Automated generation of API documentation at compile time
- **LangChain4j integration** - Support for AI/LLM integration via the GraalVM development kit

## Key Skills and Expertise
- **Java, Kotlin, or Groovy programming** with focus on modern language features
- **Reactive programming** concepts and patterns
- **Microservice architecture** principles and implementation
- **Cloud-native application development** with containerization
- **Serverless computing** approaches and patterns
- **Database integration** using Micronaut Data
- **API design** following RESTful principles
- **Message-driven applications** using Kafka, RabbitMQ, or similar
- **GraalVM native image compilation** techniques
- **Testing methodologies** for microservices and reactive systems

## Best Practices
- Leverage compile-time dependency injection instead of runtime reflection
- Use reactive programming patterns for scalable, non-blocking applications
- Implement cloud-native patterns for resilience and scalability
- Take advantage of AOT compilation for improved startup time and memory usage
- Design APIs with OpenAPI documentation generated at compile time
- Utilize Micronaut Data for type-safe database access
- Implement comprehensive security using Micronaut Security
- Create modular applications with clear separation of concerns
- Leverage serverless capabilities for event-driven architectures
- Use message-driven approaches for asynchronous communication between services

## File Types
- Java source files (.java)
- Kotlin source files (.kt)
- Groovy source files (.groovy)
- Configuration files (.yml, .yaml, .properties)
- Build files (build.gradle, pom.xml)
- OpenAPI specification files (.yml, .yaml)
- JSON data files (.json)
- SQL migration scripts (.sql)
- Static resources (.html, .css, .js)
- Binary file formats supported by Micronaut (images, documents, archives)

## Related Packages
- GraalVM Development Kit for Micronaut ^0.6.0
- Micronaut Data ^4.7.0
- Micronaut Security ^4.6.0
- Micronaut Views ^4.5.0
- Micronaut Microstream ^2.2.0
- Micronaut Kafka ^5.3.0
- Micronaut RabbitMQ ^4.3.0
- Micronaut JPA ^4.6.0
- Micronaut Oracle Cloud ^3.4.0
- Micronaut AWS ^4.1.0

## Differences From Previous Version
- **New APIs**:
  - Enhanced LangChain4j integration for AI/LLM capabilities
  - Improved GraalVM native image support
  - Additional observability features
  
- **Enhanced Features**:
  - Better cloud provider integrations
  - Optimized reactive processing
  - Improved database connectivity with Micronaut Data
  - Enhanced security features

## Custom Instructions
When working with Micronaut 4.8.5, focus on leveraging its core strengths of compile-time processing, minimal memory footprint, and rapid startup time. This version builds on Micronaut's foundation as a modern JVM-based framework optimized for microservices and serverless applications. Take advantage of the compile-time dependency injection system that eliminates the need for reflection, resulting in faster startup and reduced memory usage. Utilize the reactive programming model built on Netty for efficient non-blocking I/O operations. For database access, implement Micronaut Data repositories that generate optimized database access code at compile time rather than runtime. When deploying to cloud environments, leverage Micronaut's native cloud integration features for service discovery, distributed configuration, and observability. For maximum performance, compile your applications to native executables using GraalVM support, which is particularly valuable for serverless deployments. The framework's modular design allows you to include only the dependencies you need, further optimizing resource usage. This version also includes enhanced integration with LangChain4j via the GraalVM development kit, enabling AI and LLM capabilities in your applications.