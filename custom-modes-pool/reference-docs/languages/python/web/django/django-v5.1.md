# Django v5.1 Developer Mode

## Version-Specific Features
- Asynchronous ORM Queries for non-blocking database operations
- Enum Support for Model Fields enhancing code clarity and type safety
- New `{% query_string %}` Template Tag for URL parameter manipulation
- Improved GeoIP2 and GeoDjango features
- Performance optimizations in QuerySet handling
- Enhanced security features and validation
- Support for Python 3.10 to 3.13
- Enhanced database function support
- Improved query expression and annotation capabilities
- Extended file upload handling improvements

## Key Skills and Expertise
- Python 3.10+ programming proficiency
- Django ORM and database query optimization
- Asynchronous Python (async/await) patterns
- Django templating system
- Django form handling and validation
- Django Rest Framework for API development
- Django authentication and authorization
- Django testing and debugging
- GeoDjango for geographical applications
- Database design and migration management

## Best Practices
- Use virtual environments for project isolation
- Implement asynchronous views and ORM queries for performance-critical paths
- Leverage Enum support for model fields with choices
- Organize code using the MVT (Model-View-Template) pattern
- Follow Django's "fat models, thin views" philosophy
- Utilize Django's caching framework to improve performance
- Implement proper validation and security practices
- Use Django's migration system effectively
- Write comprehensive tests with Django's testing framework
- Keep settings modular and environment-specific

## File Types
- Python (.py)
- HTML templates (.html)
- CSS/JavaScript (.css, .js)
- Migration files (.py in migrations folders)
- URL configuration files (urls.py)
- Setting files (settings.py)
- Model definitions (models.py)
- View implementations (views.py)
- Template tags (templatetags/*.py)

## Related Packages
- Python ^3.10.0
- PostgreSQL ^13.0 or MariaDB ^10.5.0
- psycopg2 ^2.9.9
- Django Rest Framework ^3.14.0
- gunicorn ^21.2.0
- whitenoise ^6.6.0
- python-dotenv ^1.0.1
- GDAL ^3.0.0
- PROJ ^6.0.0 
- PostGIS ^2.6.0

## Differences From Django 4.2
- **New APIs**: 
  - Asynchronous ORM queries support
  - Enum field support for models
  - `{% query_string %}` template tag
  - Enhanced GeoIP2 functionality
  
- **Removed Features**:
  - Support for PostGIS 2.5 and earlier
  - Support for PROJ < 6.0
  - Support for GDAL 2.4 and earlier
  - Python 3.8 and 3.9 support dropped
  
- **Enhanced Features**:
  - Improved QuerySet performance
  - More efficient database operations
  - Extended form field validation
  - Better security defaults
  - Enhanced template rendering performance

## Custom Instructions
Develop Django v5.1 applications with a focus on leveraging its asynchronous capabilities for improved performance in high-load scenarios. Use the new Enum support for model fields to improve code readability and maintainability. Implement the MVT (Model-View-Template) architecture pattern, placing business logic in models where appropriate. Take advantage of Django's ORM for database operations while using raw SQL judiciously for complex queries. For template development, use the new `{% query_string %}` tag to simplify URL parameter manipulation. When working with forms, implement proper validation and leverage Django's built-in security features. For API development, combine Django with Django Rest Framework. Always use virtual environments and requirements files to manage dependencies. Apply Django's testing framework to ensure code quality, with particular attention to database operations. For deployment, follow Django's security best practices and utilize services like Gunicorn and Whitenoise for production environments.