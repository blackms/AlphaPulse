# ASP.NET Core v9.0 Developer Mode

## Version-Specific Features
- **Performance Optimizations** - Significant memory allocation reduction (up to 93%) and 50% faster exception handling
- **Static Asset Delivery** - New MapStaticAssets function with automatic compression and content-based ETags
- **Enhanced Minimal APIs** - Further refinements with better memory usage and improved OpenAPI support
- **Improved Error Handling** - New TypedResults methods like InternalServerError for better API responses
- **Keyed Services in Middleware** - Extended dependency injection capabilities for middleware components
- **Advanced HybridCache** - Enhanced caching system for improved application performance
- **SIMD Optimizations** - Better performance through improved vectorization using SIMD operations
- **Native AOT Enhancements** - Expanded support for native compilation including WinUI 3 applications
- **System.Text.Json Improvements** - Optimized internals for faster serialization and deserialization
- **.NET MAUI Integration** - Enhanced cross-platform capabilities with improved performance across controls

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
- Leverage asynchronous programming for I/O-bound operations to improve scalability
- Utilize the new minimal API enhancements for lightweight endpoints and microservices
- Implement proper dependency injection with attention to service lifetimes
- Optimize data access with AsNoTracking() for read-only queries and efficient pagination
- Use the new MapStaticAssets for optimized static file serving
- Take advantage of the reduced memory allocation and faster exception handling
- Always use HTTPS and implement HSTS for security
- Implement robust authentication and authorization
- Use dependency injection for better testability and loose coupling
- Leverage OpenAPI enhancements for better API documentation

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
- Microsoft.AspNetCore.App ^9.0.2
- Microsoft.EntityFrameworkCore ^9.0.2
- Microsoft.Extensions.DependencyInjection ^9.0.2
- Microsoft.Extensions.Configuration ^9.0.2
- Microsoft.AspNetCore.Identity.EntityFrameworkCore ^9.0.2
- Microsoft.AspNetCore.Authentication.JwtBearer ^9.0.2
- Microsoft.AspNetCore.Diagnostics.EntityFrameworkCore ^9.0.2
- Microsoft.AspNetCore.SignalR ^9.0.2
- Microsoft.AspNetCore.OpenApi ^9.0.2
- Microsoft.Net.Http.Headers ^9.0.2

## Differences From Previous Version
- **New APIs**:
  - MapStaticAssets for optimized static file delivery
  - Enhanced TypedResults with additional methods like InternalServerError
  - Native OpenAPI support without Swashbuckle dependency
  - Advanced SIMD vectorization capabilities
  
- **Enhanced Features**:
  - Significant performance improvements with reduced memory allocation
  - Faster exception handling (50% improvement)
  - Further optimized minimal API capabilities
  - Enhanced static file compression and caching
  - Improved .NET MAUI integration with better cross-platform support

## Custom Instructions
When working with ASP.NET Core 9.0, focus on leveraging its significant performance improvements and developer productivity enhancements. This version brings substantial memory allocation reductions (up to 93% in some scenarios) and 50% faster exception handling, making it an ideal choice for high-performance applications. Take advantage of the new MapStaticAssets function for optimized static file delivery with automatic compression (gzip in development, gzip + Brotli in production) and content-based ETags for efficient caching. For API development, utilize the enhanced minimal API capabilities which offer better memory usage and improved performance, along with native OpenAPI support that reduces dependency on external packages like Swashbuckle. Implement the new TypedResults methods for better error handling in your APIs. When working with dependency injection, explore the new Keyed Services in Middleware feature for more flexible service resolution. For data access, continue pairing ASP.NET Core with Entity Framework Core 9.0, implementing best practices like using AsNoTracking() for read-only queries and efficient pagination. For cross-platform development, leverage the improved .NET MAUI integration with enhanced performance across various controls. When appropriate, take advantage of Native AOT support for smaller, faster applications, including WinUI 3 integration. Always follow security best practices by implementing HTTPS, proper authentication, and authorization, utilizing the framework's built-in capabilities to ensure your applications are secure by default.