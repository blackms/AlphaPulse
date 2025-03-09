# Ruby on Rails v8.0.1 Developer Mode

## Version-Specific Features
- **Native Authentication System** - Built-in authentication generator that eliminates dependency on third-party gems like Devise while providing transparent user authentication
- **Solid Trifecta** - Integrated database-backed adapters for WebSockets (Solid Cable), caching (Solid Cache), and job queues (Solid Queue) that reduce Redis dependency
- **Propshaft Asset Pipeline** - New default asset pipeline replacing Sprockets, offering lighter and more efficient static asset management
- **Kamal Deployment** - Built-in support for Kamal 2 as the default deployment tool, automating Docker-based deployment configuration
- **Native Rate Limiting** - First-party request throttling capabilities integrated directly into the framework for improved security
- **SQLite in Production** - Enhanced support for using SQLite in production environments for caching, queuing, and as primary database
- **Optimized ActiveRecord** - Performance improvements including the new `select_count` method for faster record counting
- **Simplified Dependencies** - Reduced reliance on external services like Redis and complex database setups
- **Enhanced Error Protection** - Improved error handling and reporting for better debugging experience
- **Hotwire 2.0 Integration** - Updated integration with Hotwire for real-time page updates without writing JavaScript

## Key Skills and Expertise
- **Ruby Programming** with deep understanding of language features and idioms
- **Rails Architecture** including MVC pattern and RESTful design principles
- **Database Design and Management** particularly with SQLite, PostgreSQL, and MySQL
- **Front-end Technologies** including HTML, CSS, and JavaScript
- **Testing Methodologies** using RSpec, Minitest, and system testing
- **API Development** for both creation and consumption
- **Authentication and Authorization** implementation and security practices
- **Performance Optimization** techniques for Ruby and Rails applications
- **Deployment and DevOps** with Docker and Kamal
- **Version Control** with Git and GitHub/GitLab workflows

## Best Practices
- Utilize the native authentication system instead of third-party gems for better control and transparency
- Leverage the Solid Trifecta components to reduce external dependencies and simplify infrastructure
- Use Propshaft for static assets and supplement with specialized tools like esbuild for complex needs
- Implement native rate limiting to prevent abuse and ensure application stability
- Consider SQLite for production use cases where appropriate to simplify deployment and maintenance
- Take advantage of performance optimizations like the new `select_count` method in ActiveRecord
- Use Kamal for streamlined deployments, especially for Docker-based environments
- Follow Rails conventions and use generators to ensure consistent application structure
- Implement comprehensive testing at unit, integration, and system levels
- Consider modular application design using Rails Engines for complex applications

## File Types
- Ruby source files (.rb)
- ERB/HAML/Slim templates (.erb, .haml, .slim)
- JavaScript files (.js, .jsx)
- TypeScript files (.ts, .tsx)
- CSS/SCSS stylesheets (.css, .scss)
- YAML configuration files (.yml, .yaml)
- JSON data files (.json)
- SQL database migrations (.sql)
- Markdown documentation (.md)
- Asset files (.png, .jpg, .svg)

## Related Packages
- Ruby ^3.2.0
- SQLite ^3.42.0
- Propshaft ^1.0.0
- Kamal ^2.0.0
- Hotwire ^2.0.0
- Turbo ^8.0.0
- Stimulus ^2.0.0
- Jbuilder ^3.0.0
- Minitest ^5.20.0
- RSpec ^3.12.0

## Differences From Previous Version
- **New APIs**:
  - Native authentication system replacing the need for Devise
  - Built-in rate limiting for request throttling
  - `select_count` method for optimized record counting
  - Solid Cable, Solid Cache, and Solid Queue for database-backed services
  
- **Removed Features**:
  - Sprockets is no longer the default asset pipeline
  - Capistrano is replaced by Kamal as the default deployment tool
  
- **Enhanced Features**:
  - Improved SQLite support for production environments
  - Better error reporting and debugging tools
  - Streamlined Docker-based deployment
  - Optimized performance for common ActiveRecord operations

## Custom Instructions
When working with Ruby on Rails 8.0.1, focus on leveraging its simplified architecture and reduced dependencies. This version marks a significant shift in Rails philosophy, emphasizing SQLite as a production-ready database and reducing reliance on external services like Redis through the Solid Trifecta components. Take advantage of the native authentication system, which provides greater transparency and control compared to third-party gems like Devise. For asset management, use Propshaft for static assets, which is more lightweight than Sprockets, but be prepared to integrate specialized tools for complex JavaScript bundling. Implement the native rate limiting features to protect your application from abuse without additional dependencies. When deploying, utilize Kamal 2 for Docker-based deployments, which simplifies infrastructure management through automatically generated configuration files. For real-time features, leverage the updated Hotwire 2.0 integration, which enables dynamic updates without writing custom JavaScript. If migrating from Rails 7.x, pay particular attention to the asset pipeline changes and the new database-backed adapters for WebSockets, caching, and job processing, as these represent significant architectural shifts that might require application adjustments.