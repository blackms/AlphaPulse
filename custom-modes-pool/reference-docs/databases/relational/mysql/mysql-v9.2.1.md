# MySQL v9.2.1 Developer Mode

## Version-Specific Features
- **JavaScript Integration** - Support for JavaScript libraries in stored routines with SQL APIs and transaction handling
- **HeatWave Enhancements** - RAG search extensions with custom vector embeddings and automated zone map construction
- **SQL Window Functions** - Advanced analytical queries with OVER(), PARTITION BY, and frame specifications
- **Common Table Expressions (CTEs)** - Support for WITH clause and recursive queries
- **JSON Document Store** - Native JSON data type with comprehensive function support
- **Spatial Reference System Management** - New privileges and capabilities for spatial data handling
- **Invisible Indexes** - Ability to make indexes invisible for testing index removal impact
- **Atomic DDL Statements** - Crash-safe data definition language operations
- **InnoDB Enhancements** - Improved performance, fast recovery of tables loaded into HeatWave after restart
- **Role-Based Access Control** - Simplified user privilege management with roles

## Key Skills and Expertise
- **SQL Query Design** with focus on performance optimization
- **JavaScript Knowledge** for working with MySQL's JavaScript integration
- **Database Schema Design** principles and normalization
- **Transaction Management** with ACID properties
- **Indexing Strategies** for query optimization
- **Performance Tuning** of queries and server configuration
- **HeatWave Integration** for analytics and machine learning
- **Spatial Data Handling** including reference systems
- **Security Implementation** including authentication and authorization
- **JSON Data Modeling** and querying

## Best Practices
- Use utf8mb4 as the default character set and utf8mb4_bin as the collation
- Set InnoDB as the default storage engine
- Configure appropriate values for max_allowed_packet (at least 256M) and innodb_log_file_size (at least 2GB)
- Use READ-COMMITTED as the global transaction isolation level
- Configure binary logging to use row-based format
- Implement proper indexing strategy based on query patterns
- Use prepared statements to prevent SQL injection
- Normalize database design to appropriate normal form
- Utilize transactions for data integrity
- Leverage JavaScript capabilities for complex stored procedures when appropriate

## File Types
- SQL script files (.sql)
- JavaScript files for MySQL routines (.js)
- Database dump files (.sql, .dump)
- Configuration files (my.cnf, my.ini)
- Log files (error log, binary log, slow query log)
- InnoDB tablespace files (.ibd)
- Backup files (.bak, .backup)
- CSV/TSV export files (.csv, .tsv)
- SSL certificate files (.pem, .crt, .key)
- Shell script files for automation (.sh, .bat)

## Related Packages
- MySQL Server ^9.2.1
- MySQL Connector/J ^9.2.0
- MySQL Connector/ODBC ^9.2.0
- MySQL Connector/Python ^9.2.0
- MySQL Connector/NET ^9.2.0
- MySQL Workbench ^9.0.0
- MySQL Shell ^9.0.0
- MySQL Router ^9.0.0
- MySQL Enterprise Backup ^9.0.0
- MySQL Enterprise Monitor ^9.0.0

## Differences From Previous Version
- **New APIs**:
  - JavaScript libraries and APIs for stored routines
  - JavaScript MySQL transaction API
  - CREATE_SPATIAL_REFERENCE_SYSTEM privilege
  - Enhanced HeatWave integration capabilities
  
- **Enhanced Features**:
  - Significantly improved JavaScript integration
  - Better HeatWave capabilities with RAG search extensions
  - Fast recovery of InnoDB tables loaded into HeatWave
  - Automated zone map construction for query optimization
  - Innovation release track focusing on cutting-edge features

## Custom Instructions
When working with MySQL 9.2.1, focus on leveraging its Innovation release features, particularly the new JavaScript integration and enhanced HeatWave capabilities. This version represents a significant leap forward with the introduction of JavaScript support for stored routines, allowing for more complex logic within the database. Take advantage of the JavaScript MySQL transaction API for more flexible transaction handling when needed. For analytics and machine learning applications, explore the HeatWave enhancements, particularly the RAG search extensions with custom vector embeddings and automated zone map construction for faster query processing. When configuring your database, follow the recommended best practices including setting utf8mb4 as the default character set and using InnoDB as the storage engine. Configure appropriate values for critical parameters like max_allowed_packet (at least 256M) and innodb_log_file_size (at least 2GB). For transaction management, use READ-COMMITTED as the global isolation level and configure binary logging to use row-based format. If working with spatial data, utilize the new CREATE_SPATIAL_REFERENCE_SYSTEM privilege for better management of spatial reference systems. Be aware that as an Innovation release, version 9.2.1 focuses on cutting-edge features and may receive more frequent updates than Long Term Support (LTS) releases, so ensure your deployment strategy accounts for this more dynamic update cycle. When deploying to production, carefully assess whether the Innovation track aligns with your stability requirements or if an LTS release would be more appropriate for your use case.