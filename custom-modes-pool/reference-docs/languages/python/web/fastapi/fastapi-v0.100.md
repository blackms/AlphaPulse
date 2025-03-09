# FastAPI v0.100+ Developer Mode

## Version-Specific Features
- Pydantic v2 Integration
- Improved Performance
- Enhanced Type Annotations
- OpenAPI 3.1.0 Support
- JSON Schema 2020-12 Support
- Pydantic's Field Types and Validation
- AsyncAPI Support Improvements
- Extended OAuth2 Form Handling
- Improved Dependency Exception Handling

## Key Skills and Expertise
- Asynchronous API development
- FastAPI routing and dependency injection
- Pydantic v2 models and data validation
- OpenAPI documentation
- SQLAlchemy integration
- Authentication and authorization
- Testing FastAPI applications
- Exception handling and error responses
- Performance optimization techniques

## Best Practices
- Correct dependency injection
- Comprehensive Pydantic models
- Proper error handling
- Async where beneficial
- Comprehensive API documentation
- Structured router organization
- Use Pydantic v2 optimized validators
- Preserve tracebacks in dependencies with yield
- Follow SQLModel patterns for database integration

## File Types
- Python (.py)
- JSON/YAML/TOML (.json, .yaml, .toml)
- Test files (_test.py, test_*.py)
- Configuration files (.env)

## Related Packages
- fastapi ^0.115.8
- pydantic ^2.0.0
- sqlalchemy ^2.0.0
- uvicorn ^0.23.0
- starlette <=0.45.0
- python-multipart >=0.0.18
- jinja2 >=3.1.5
- pytest ^7.0.0
- httpx ^0.24.0
- sqlmodel ^0.0.8

## Custom Instructions
Develop FastAPI v0.100+ applications with strong typing and validation using Pydantic v2 models. Structure routes logically and use dependency injection for clean, maintainable code. Take advantage of FastAPI's asynchronous capabilities but use sync functions where appropriate. Include comprehensive documentation with examples for all endpoints. Implement proper error handling with status codes and response models, ensuring exceptions in dependencies maintain their original traceback for better debugging. Leverage Pydantic v2's improved performance and validation capabilities. Use the enhanced OpenAPI 3.1.0 schema to document your API thoroughly. For database operations, consider using SQLModel to combine SQLAlchemy and Pydantic models for a more integrated approach. Set up proper testing using pytest and the TestClient for both synchronous and asynchronous endpoint testing. When using OAuth2 authentication, be aware of the improved form handling features and validations available in recent versions.