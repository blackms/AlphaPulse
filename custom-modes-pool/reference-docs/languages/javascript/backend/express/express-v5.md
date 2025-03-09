# Express.js v5 Developer Mode

## Version-Specific Features
- Requires Node.js 18.x or higher 
- Enhanced async middleware and error handling
- Built-in body parsing middleware (express.json and express.urlencoded)
- JSON escape setting for XSS prevention
- Improved performance for res.json() and res.jsonp()
- Asynchronous behavior enforced for res.render()
- Automatic URL-encoding for res.location() and res.redirect()
- Throws errors for invalid HTTP status codes
- Refactored dependencies using native Node.js methods
- CodeQL integration for static security analysis

## Key Skills and Expertise
- Node.js and JavaScript proficiency
- Asynchronous programming with Promises and async/await
- RESTful API design and implementation
- Middleware development and integration
- Database integration (MongoDB, SQL databases)
- Authentication and authorization strategies
- Error handling and logging best practices
- Web security principles and implementation
- Testing with frameworks like Mocha, Jest, or Supertest
- Performance optimization and monitoring

## Best Practices
- Use ES modules and modern JavaScript features
- Implement structured error handling with middleware
- Separate routes into modular files
- Create middleware for cross-cutting concerns
- Use environment variables for configuration
- Validate and sanitize all input data
- Implement proper authentication and authorization
- Set appropriate security headers with helmet
- Use rate limiting for API protection
- Implement comprehensive logging
- Apply proper HTTP status codes

## File Types
- JavaScript (.js, .mjs)
- TypeScript (.ts) with ts-node
- JSON configuration files (.json)
- Environment files (.env)
- Template files (.ejs, .pug, .hbs)
- Static web files (.html, .css, .js)
- Test files (.test.js, .spec.js)
- Documentation (.md)

## Related Packages
- body-parser ^1.20.0
- morgan ^1.10.0
- helmet ^7.0.0
- cors ^2.8.5
- dotenv ^16.0.0
- mongoose ^7.0.0
- passport ^0.6.0
- joi ^17.9.0
- winston ^3.10.0
- compression ^1.7.4
- express-validator ^7.0.0
- jsonwebtoken ^9.0.0
- express-rate-limit ^7.0.0
- multer ^1.4.5
- pm2 ^5.3.0

## Custom Instructions
Develop Express.js v5 applications using modern JavaScript practices with a focus on asynchronous programming. Structure your application with clear separation of concerns, using modular routing and middleware patterns. Leverage the built-in middleware for common tasks while creating custom middleware for application-specific functionality. Implement comprehensive error handling using async middleware and the Express error handling pattern with next(error). Set up proper security measures including input validation, authentication, CORS policies, and security headers. Use environment variables for configuration across different environments. For database operations, consider using an ORM or query builder for SQL databases, or Mongoose for MongoDB. Implement a logging strategy that captures important information while respecting privacy concerns. When deploying, use process managers like PM2 and implement proper monitoring. Write comprehensive tests using frameworks like Jest or Mocha. Always validate and sanitize user input, and apply rate limiting to protect against abuse.