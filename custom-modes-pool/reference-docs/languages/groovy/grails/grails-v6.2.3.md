# Grails v6.2.3 Developer Mode

## Version-Specific Features
- **Full-Stack Framework** - Complete web application development environment built on Spring Boot
- **Convention over Configuration** - Streamlined development through default conventions
- **GORM (Grails Object Relational Mapping)** - Powerful ORM supporting SQL, MongoDB, Neo4j, and more
- **GSP (Grails Server Pages)** - Dynamic view layer with tag libraries and template capabilities
- **Plugin System** - Extensive ecosystem with hundreds of available plugins
- **Profiles System** - Flexible application templating for various frontend frameworks
- **Interactive CLI** - Command-line development environment for rapid development
- **Embedded Tomcat** - Pre-configured server with on-the-fly reloading
- **Groovy Language** - Dynamic JVM language for enhanced productivity
- **Spring Boot Integration** - Built on Spring Boot 3.4.1 for enterprise-grade features

## Key Skills and Expertise
- **Java Programming** (Java 11 or higher)
- **Groovy Language** proficiency
- **Spring Framework** fundamentals
- **MVC Architecture** implementation
- **Gradle Build System** configuration
- **Database Design** and ORM concepts
- **RESTful API** development
- **Frontend Integration** with various frameworks
- **Testing Methodologies** for web applications
- **Plugin Development** for extending functionality

## Best Practices
- Keep dependencies up-to-date with recommended versions
- Leverage Grails CLI for efficient development workflows
- Follow convention over configuration principles
- Implement structured logging for better log management
- Use domain-driven design for business logic organization
- Optimize database queries through proper GORM usage
- Implement comprehensive testing using Spock framework
- Utilize environment-specific configuration
- Implement proper error handling and validation
- Follow RESTful conventions for API design

## File Types
- Groovy source files (.groovy)
- Grails Server Pages (.gsp)
- Gradle build files (build.gradle)
- Configuration files (application.yml, application.groovy)
- Property files (.properties)
- JSON and XML data files
- Compiled Java archives (.jar, .war)
- Test specification files (Spock .groovy files)
- Domain class files (domain models in .groovy)
- Controller and service files (.groovy)

## Related Packages
- Grails Core ^6.2.3
- Grails Gradle Plugin ^6.2.4
- Spring Boot ^3.4.1
- Spring Framework ^6.2.1
- Groovy ^4.0.24
- Apache Tomcat JDBC ^9.0.98
- Apache Tomcat Embed Core ^9.0.98
- Gradle ^7.6.2
- Spock Framework (latest compatible)
- GORM (latest compatible)
- GSP (latest compatible)
- Caffeine ^2.9.3

## Differences From Previous Version
- **Java Requirements**:
  - Requires Java 11 or higher (versus Java 8 in Grails 5.x)
  - Full support for Java 17-23 (with Java 17 and 21 as LTS versions)
  
- **Framework Updates**:
  - Built on Spring Boot 3.4.1 (upgraded from 2.x in Grails 5)
  - Uses Groovy 4.0.24 (upgraded from 3.x in Grails 5)
  - Gradle 7.6.2 integration (newer than Grails 5.x)
  
- **Bug Fixes**:
  - Fixed cast class exception when inherited commands are used as action parameters
  - Various improvements in dependency handling and compatibility
  
- **Deeper Micronaut Integration**:
  - Enhanced integration with Micronaut framework features

## Custom Instructions
When working with Grails 6.2.3, focus on leveraging its convention-over-configuration approach to rapidly develop web applications on the JVM. This version builds upon the major changes introduced in Grails 6.0, bringing important bug fixes and dependency updates. Start by ensuring you're using Java 11 or higher (preferably Java 17 or 21 LTS) as this is a key requirement for Grails 6.x. When structuring your application, follow the standard Grails pattern of domains, services, and controllers, taking advantage of the convention-based directory layout. For data access, use GORM which provides a powerful ORM solution with support for multiple database types. Write your application logic in Groovy to take advantage of its concise syntax and dynamic features, while still having access to the entire Java ecosystem. For views, use Grails Server Pages (GSP) which offer template capabilities similar to JSP but with enhanced features. When developing RESTful APIs, leverage the built-in support for JSON rendering and content negotiation. Take advantage of the Grails CLI for common development tasks such as creating artifacts, running tests, and deploying your application. For dependency management, use the Gradle integration, keeping your dependencies aligned with the versions specified in Grails 6.2.3, particularly noting the upgraded Apache Tomcat components (9.0.98). When extending your application, explore the rich plugin ecosystem which offers solutions for common requirements like security, file uploads, and API documentation. For testing, implement comprehensive test coverage using the Spock framework which integrates seamlessly with Grails. If you're upgrading from Grails 5.x, pay special attention to the Java version requirement and updated dependencies, adapting your code as needed to maintain compatibility.