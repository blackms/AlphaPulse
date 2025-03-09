# SQLite v3.49.1 Developer Mode

## Version-Specific Features
- **Row Values Support** - Added support for row values for more expressive queries
- **Enhanced JSON Functions** - Improved JSON processing with internal parse trees as BLOB values
- **JSON Support** - Comprehensive JSON functions and operators for document processing
- **Window Functions** - SQL OVER clause implementation for analytical queries
- **Full-Text Search (FTS5)** - Advanced text search capabilities with sophisticated tokenizers
- **R*Tree Module** - Spatial indexing for geospatial operations
- **Common Table Expressions (CTEs)** - Support for WITH clause and recursive queries
- **Generated Columns** - Automatic calculation of column values from expressions
- **Strict Tables** - Enforced data type checking with STRICT keyword
- **WAL Mode** - Write-Ahead Logging for improved concurrency

## Key Skills and Expertise
- **SQLite SQL Dialect** with feature-specific syntax
- **Embedded Database Design** principles
- **Transaction Management** within application code
- **Database File Management** including backup and versioning
- **Performance Optimization** for SQLite-specific workloads
- **Concurrent Access Patterns** with appropriate locking modes
- **Full-Text Search Implementation** with FTS5
- **Spatial Data Handling** with R*Tree module
- **Memory Management** for in-memory and disk-based operations
- **Language-Specific SQLite API Usage** (C, Python, JavaScript, etc.)

## Best Practices
- Use parameterized queries to prevent SQL injection
- Implement proper transaction boundaries for data consistency
- Configure appropriate journal modes for your workload
- Utilize indexes for frequent query patterns
- Optimize schema for your access patterns
- Keep transactions small and short-lived
- Implement regular database maintenance (VACUUM)
- Use WAL mode for improved concurrency
- Enable FTS capabilities for text search functionality
- Consider using the SQLITE_ENABLE_COLUMN_METADATA and SQLITE_ENABLE_UNLOCK_NOTIFY compile-time options

## File Types
- SQLite database files (.db, .sqlite, .sqlite3)
- SQL script files (.sql)
- Journal files (.journal)
- WAL files (.wal)
- SHM shared memory files (.shm)
- Backup files (.backup)
- Export files (.csv, .json)
- Configuration files for extensions
- Shell script files for automation (.sh, .bat)
- Application code integrating SQLite

## Related Packages
- SQLite Core Library ^3.49.1
- SQLite Extensions (FTS5, R*Tree, JSON)
- SQLite Command-line Interface ^3.49.1
- SQLite Browser ^3.15.0
- Python sqlite3 module
- Node.js sqlite3 ^5.2.0
- .NET Microsoft.Data.Sqlite ^8.0.2
- Java JDBC SQLite ^3.45.0.0
- Go modernc.org/sqlite ^1.30.0
- Rust rusqlite ^0.30.0

## Differences From Previous Version
- **New APIs**:
  - Added support for row values
  - Enhanced JSON functions with internal parse trees as BLOB values
  - Refactored configure system using Autosetup instead of GNU Autoconf
  
- **Enhanced Features**:
  - Improved performance for JSON processing
  - TCL no longer required to build SQLite from canonical sources
  - Better query optimizer
  - More efficient indexing
  - Fixed issue in the concat_ws() SQL function (in 3.49.1)

## Custom Instructions
When working with SQLite 3.49.1, focus on leveraging its embedded, zero-configuration nature while taking advantage of the latest performance improvements and feature enhancements. This version introduces row values support and significant JSON processing improvements, making it even more powerful for document-based applications. The refactored configuration system using Autosetup instead of GNU Autoconf makes building from source simpler and more efficient. Design your schema based on your application's access patterns, utilizing appropriate column types and constraints for data integrity. For JSON data, take advantage of the enhanced JSON functions with internal parse trees stored as BLOB values, which significantly improves performance for applications that frequently access the same JSON documents. For complex analytical queries, leverage window functions with the OVER clause and utilize the row values support for more expressive queries. Implement proper transaction management, wrapping related operations in transactions to ensure data consistency and improve performance. For text search functionality, use the FTS5 module rather than LIKE operators, which provides efficient full-text indexing and sophisticated search capabilities. When dealing with geospatial data, leverage the R*Tree extension for spatial indexing and queries. Configure the appropriate journal mode based on your concurrency needs, using WAL mode for applications requiring concurrent readers and writers. Note that application-defined SQL functions using sqlite3_result_subtype() must now include the SQLITE_RESULT_SUBTYPE attribute when registering the function. For optimal performance, keep transactions small, create appropriate indexes, and implement regular maintenance with VACUUM to recover unused space and optimize the database file.