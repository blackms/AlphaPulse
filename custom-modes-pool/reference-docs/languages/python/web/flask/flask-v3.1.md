# Flask v3.1 Developer Mode

## Version-Specific Features
- Python 3.9+ Support
- Enhanced Asynchronous Programming Capabilities
- Improved Context Management using Python Context Vars
- Efficient WSGI Implementation via Werkzeug
- Jinja2 Template Engine Integration
- Advanced Security Features with MarkupSafe and itsdangerous
- Command-line Interface Support through Click
- Improved File Upload Handling
- Application Factory Pattern Support
- Blueprint System for Application Modularization

## Key Skills and Expertise
- Python 3.9+ Programming
- HTTP Protocol and RESTful API Design
- Routing and URL Building
- Request and Response Handling
- Template Rendering with Jinja2
- Flask Extension Integration
- Database Management with SQLAlchemy
- User Authentication and Session Management
- Error Handling and Debugging
- Testing Flask Applications
- Asynchronous Programming with Python
- Web Security Best Practices

## Best Practices
- Use Application Factory Pattern for Larger Applications
- Implement Blueprint Architecture for Modularization
- Utilize Virtual Environments for Dependency Isolation
- Leverage Flask-SQLAlchemy for ORM Database Access
- Implement Proper Error Handling and Logging
- Follow RESTful Principles for API Design
- Store Configuration in Environment Variables
- Apply Proper Input Validation and Sanitization
- Use Flask Extensions for Common Functionalities
- Implement Comprehensive Testing Strategy

## File Types
- Python Files (.py)
- HTML Templates (.html)
- CSS Stylesheets (.css, .scss)
- JavaScript Files (.js)
- Configuration Files (.cfg, .ini, .py)
- Environment Files (.env)
- Requirement Specifications (requirements.txt)
- Database Migration Scripts

## Related Packages
- Werkzeug >= 2.3.0
- Jinja2 >= 3.1.2
- itsdangerous >= 2.1.2
- Click >= 8.1.3
- Flask-SQLAlchemy ~= 3.0
- Flask-WTF ~= 1.2
- Flask-Login ~= 0.6
- Flask-RESTful ~= 0.3
- Flask-Migrate ~= 4.0
- Flask-CORS ~= 4.0
- Flask-JWT-Extended ~= 4.4
- pytest-flask ~= 1.2

## Custom Instructions
Develop Flask v3.1 applications focusing on clean, maintainable code using modern Python features. Implement the application factory pattern for larger projects to enable better testing and configuration management. Use blueprints to organize your application into logical components. Leverage Flask's extension ecosystem for common functionality like database ORM (Flask-SQLAlchemy), form handling (Flask-WTF), and authentication (Flask-Login). Take advantage of Flask's asynchronous support where appropriate, but be aware of the limitations in a WSGI environment. Follow RESTful principles when designing APIs, with proper status codes, response formats, and error handling. Implement comprehensive testing using pytest-flask. For database operations, use Flask-Migrate to manage schema changes over time. Secure your application by following web security best practices, including proper input validation, output escaping, CSRF protection, and secure session management. Use environment variables for configuration to maintain security and flexibility across different deployment environments.