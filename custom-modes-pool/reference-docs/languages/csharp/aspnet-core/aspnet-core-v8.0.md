# ASP.NET Core v8.0 Developer Mode

## Version-Specific Features
- **Blazor United** - New unified model combining Server and WebAssembly rendering approaches for Blazor applications
- **.NET Aspire** - Cloud-native application stack for building distributed applications with improved observability
- **Identity API Endpoints** - Pre-built endpoints for authentication and user management
- **Route Groups** - Simplified route organization using the RouteGroup pattern
- **Minimal API Enhancements** - Improved parameter binding, file uploads, and OpenAPI support
- **Output Caching** - Advanced caching system for improved response times
- **Rate Limiting** - Built-in, highly configurable rate limiting middleware
- **AOT Compilation** - Ahead-of-time compilation support for improved startup performance
- **Native AOT in Minimal APIs** - Support for fully native compilation in Minimal API projects
- **Improved Dependency Injection** - Enhanced DI with keyed services and transient disposables optimization

## Key Skills and Expertise
- **C# Programming** with modern language features (C# 12)
- **.NET Runtime** understanding for performance optimization
- **HTTP Protocol** knowledge for web application development
- **Web API Design** principles and implementation
- **Authentication and Authorization** implementation
- **Entity Framework Core** for data access
- **Dependency Injection** patterns and container usage
- **Asynchronous Programming** with async/await
- **Middleware Development** for request processing
- **JavaScript Integration** for client-side functionality

## Best Practices
- Leverage minimal APIs for simpler, performance-focused endpoints
- Implement proper authentication with ASP.NET Core Identity
- Use dependency injection for better testability and loose coupling
- Implement proper exception handling middleware
- Utilize Output Caching for improved performance
- Follow RESTful design principles for APIs
- Implement proper logging and monitoring with built-in providers
- Use the options pattern for configuration management
- Implement proper model validation with data annotations or FluentValidation
- Leverage the new route grouping for organized API structure

## File Types
- C# source files (.cs)
- Razor view files (.cshtml)
- Razor component files (.razor)
- JSON configuration files (appsettings.json)
- Project files (.csproj)
- Solution files (.sln)
- Static files (.js, .css, .html)
- Docker configuration (Dockerfile, docker-compose.yml)
- Entity Framework migration files
- Certificate files (.pfx, .cer)

## Related Packages
- Microsoft.AspNetCore.App ^8.0.0
- Microsoft.EntityFrameworkCore ^8.0.0
- Microsoft.Extensions.DependencyInjection ^8.0.0
- Microsoft.Extensions.Configuration ^8.0.0
- Microsoft.AspNetCore.Identity.EntityFrameworkCore ^8.0.0
- Microsoft.AspNetCore.Authentication.JwtBearer ^8.0.0
- Microsoft.AspNetCore.Diagnostics.EntityFrameworkCore ^8.0.0
- Microsoft.AspNetCore.SignalR ^8.0.0
- Microsoft.NET.Build.Containers ^8.0.0
- System.IdentityModel.Tokens.Jwt ^7.0.0

## Differences From Previous Version
- **New APIs**:
  - Blazor United model for unified server/client rendering
  - Identity API endpoints for simplified authentication
  - .NET Aspire integration for cloud-native applications
  - Route group APIs for better route organization
  
- **Enhanced Features**:
  - Improved minimal API capabilities with parameter binding
  - More powerful output caching system
  - Native AOT support for improved performance
  - Better dependency injection system
  - Enhanced SignalR with improved scaling options

## Custom Instructions
When working with ASP.NET Core 8.0, focus on leveraging its modern, high-performance features for building web applications and APIs. This major version represents a significant evolution with the introduction of Blazor United, which combines the best aspects of server and WebAssembly rendering. For API development, embrace the minimal API approach which provides a streamlined, performant alternative to controller-based APIs, now enhanced with better parameter binding and file upload support. Take advantage of the new route grouping feature to organize endpoints logically and maintain clean code structure. For authentication, utilize the new Identity API Endpoints which eliminate boilerplate code for common user management scenarios. When building cloud-native applications, explore .NET Aspire which provides infrastructure components for distributed applications with excellent observability. Implement the enhanced output caching system for improved performance, with more granular control over cache policies. For high-traffic applications, utilize the built-in rate limiting middleware to protect your services. When appropriate, leverage AOT compilation, particularly in minimal APIs, to significantly improve startup time and reduce memory usage. For database access, pair ASP.NET Core with Entity Framework Core 8.0, taking advantage of its enhanced JSON column support and improved performance. For real-time features, utilize SignalR with its improved scaling capabilities for WebSocket-based communication.