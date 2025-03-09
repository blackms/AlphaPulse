# Spring Framework v6.2.3 Developer Mode

## Version-Specific Features
- **Placeholder Escaping** - Support for escaping property placeholders using a configurable escape character
- **Fallback Beans** - New @Fallback annotation for default implementations when no other bean of that type is provided
- **Dynamic Property Registration** - Ability to register dynamic properties in tests via DynamicPropertyRegistrar beans
- **Enhanced URL Parsing** - New implementations based on Living URL standard and RFC 3986 syntax
- **Improved Generic Type Matching** - Deeper generic type matching for injection points
- **Revised Autowiring Algorithm** - Parameter name matches and @Qualifier matches now overrule @Priority ranking
- **Full Null-Safety Migration** - Complete framework migration to JSpecify and NullAway for improved type safety
- **Dependency Injection** - Core support for IoC (Inversion of Control) with various injection styles
- **Aspect-Oriented Programming** - Comprehensive AOP capabilities for cross-cutting concerns
- **Transaction Management** - Declarative and programmatic transaction support for various data sources

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
- Use @Fallback for default implementations instead of @Primary when appropriate
- Utilize the new URL parser implementations for enhanced security
- Leverage Spring's constructor injection for mandatory dependencies
- Follow consistent naming conventions for Spring components
- Add @Override on methods overriding or implementing super type methods
- Use @since for new classes and public/protected methods
- Reference class fields using 'this' for clarity
- Take advantage of Spring Boot auto-configuration where appropriate
- Use Spring profiles for environment-specific configurations
- Implement proper exception handling with Spring's exception translation

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
- Spring Framework Core ^6.2.3
- Spring Web ^6.2.3
- Spring WebFlux ^6.2.3
- Spring WebMVC ^6.2.3
- Spring Data ^3.2.3
- Spring Security ^6.2.0
- Spring Boot ^3.4.3, ^3.5.0-M2
- Jakarta EE ^9.0
- Hibernate ORM ^6.4.0
- Jackson ^2.15.0

## Differences From Previous Version
- **New APIs**:
  - @Fallback annotation for default implementations
  - DynamicPropertyRegistrar for test property registration
  - Enhanced URL parser implementations
  
- **Enhanced Features**:
  - Revised autowiring algorithm with improved priority handling
  - Deeper generic type matching for injection points
  - Early failure for @ComponentScan with REGISTER_BEAN conditions
  - Clearer indication that bean definition overriding is discouraged
  - More robust null-safety with JSpecify and NullAway
  - Additional bug fixes and performance improvements

## Custom Instructions
When working with Spring Framework 6.2.3, focus on leveraging its comprehensive programming and configuration model for building modern Java-based enterprise applications. This version introduces several useful features including the @Fallback annotation, which provides a more elegant way to define default implementations compared to @Primary annotations. Take advantage of the placeholder escaping support when you need literal placeholder syntax in your properties, and explore the new URL parser implementations for enhanced security in web applications. For dependency injection, Spring 6.2.3 offers improved generic type matching and a revised autowiring algorithm that prioritizes parameter name matches and @Qualifier annotations over @Priority rankings. When designing Spring applications, follow a component-based architecture using Spring's stereotypes (@Component, @Service, @Repository, @Controller) and favor constructor injection for required dependencies to ensure proper initialization. For testing, leverage the new dynamic property registration capability to simplify test setup. Take advantage of Spring's transaction management for data consistency, using @Transactional with appropriate propagation settings. When working with REST APIs, use Spring MVC or WebFlux with proper content negotiation and error handling. Stay aware of nullability through Spring's null-safety annotations (@NonNull, @Nullable) which are now backed by a complete JSpecify and NullAway migration. For configuration, choose the appropriate approach based on your needs: Java-based @Configuration for compile-time safety, property files for environment-specific settings, and profiles for environment isolation.