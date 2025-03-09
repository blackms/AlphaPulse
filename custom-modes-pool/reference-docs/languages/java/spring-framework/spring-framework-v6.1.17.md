# Spring Framework v6.1.17 Developer Mode

## Version-Specific Features
- **Jakarta EE 9-10 Support** - Compatible with Jakarta EE 9-10 standards and jakarta namespace
- **Method Validation Support** - Built-in validation for controller method parameters
- **Enhanced Observability** - Improved integration with Micrometer and Micrometer Tracing
- **RFC 7807 Problem Details** - Enhanced support for standardized error responses
- **Dependency Injection** - Core support for IoC with various injection methods
- **Aspect-Oriented Programming** - Comprehensive AOP capabilities for cross-cutting concerns
- **Transaction Management** - Declarative and programmatic transaction support
- **Spring MVC** - Web framework with comprehensive request handling
- **Spring WebFlux** - Reactive web framework for non-blocking applications
- **Extensive Testing Support** - Robust testing framework for Spring applications

## Key Skills and Expertise
- **Java Programming** with Java 17+ features
- **Dependency Injection** concepts and implementation
- **Spring Configuration** using Java, annotations, and XML
- **Bean Lifecycle Management** and scopes
- **Aspect-Oriented Programming** with Spring AOP
- **Transaction Management** with @Transactional
- **REST API Development** with Spring MVC and WebFlux
- **Testing Spring Applications** with the Spring TestContext Framework
- **Spring Expression Language (SpEL)** usage
- **Spring Integration Patterns** with various technologies

## Best Practices
- Leverage constructor injection for mandatory dependencies
- Utilize Spring Boot auto-configuration when available
- Follow proper exception handling with Spring's exception translation
- Use appropriate transaction boundaries and isolation levels
- Implement proper validation using Bean Validation (Jakarta Validation)
- Structure applications following Spring's component model
- Configure appropriate bean scopes based on requirements
- Use Spring profiles for environment-specific configurations
- Leverage Spring's caching abstraction for performance
- Implement proper security measures with Spring Security

## File Types
- Java source files (.java)
- XML configuration files (.xml)
- Properties files (.properties)
- YAML configuration files (.yml, .yaml)
- Kotlin source files (.kt)
- Groovy source files (.groovy)
- SQL scripts (.sql)
- Thymeleaf templates (.html)
- Spring JavaConfig files (@Configuration classes)
- JUnit test files (@SpringBootTest classes)

## Related Packages
- Spring Framework Core ^6.1.17
- Spring Web ^6.1.17
- Spring WebFlux ^6.1.17
- Spring WebMVC ^6.1.17
- Spring Data ^3.1.x
- Spring Security ^6.1.x
- Spring Boot ^3.3.9
- Jakarta EE ^9.0-10.0
- Hibernate ORM ^6.3.x
- Jackson ^2.15.0

## Differences From Previous Version
- **Bug Fixes**:
  - Contains 17 fixes and documentation improvements
  - Enhanced stability and reliability for production environments
  
- **Compatibility**:
  - Support for Java 17 through 23
  - Improved integration with Spring Boot 3.3.9
  - Ongoing refinements to Jakarta EE compatibility
  - Continued enhancements to observability features

## Custom Instructions
When working with Spring Framework 6.1.17, focus on leveraging its mature, stable feature set for building enterprise Java applications. This maintenance release builds upon the significant improvements introduced in the 6.x line while providing important bug fixes and enhancements. As part of the main production line with long-term support, 6.1.17 is ideal for stable, production applications. Design your Spring applications following the standard component model, using Spring's stereotypes (@Component, @Service, @Repository, @Controller) to structure your codebase logically. For dependency injection, prefer constructor injection for required dependencies to ensure proper initialization. Take full advantage of Spring's observability features with Micrometer and Micrometer Tracing to gain insights into your application's performance and health. When building REST APIs, utilize the enhanced RFC 7807 problem details support for standardized error responses that are easier for clients to consume programmatically. For validation, leverage the built-in method validation support for controller method parameters to ensure data integrity with minimal boilerplate code. When configuring your application, choose the approach that best suits your needs: Java-based @Configuration for compile-time safety, property files for environment-specific settings, and profiles for environment isolation. For transaction management, use Spring's @Transactional annotation with appropriate propagation settings to ensure data consistency. When testing, use Spring's comprehensive testing support to create integration tests that accurately reflect how your components will behave in production.