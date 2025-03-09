# PostgreSQL v16.0 Developer Mode

## Version-Specific Features
- **Logical Replication Improvements** - Support for two-phase commit and row filtering in logical replication
- **Parallel Query Enhancements** - More efficient parallel query execution for complex operations
- **MERGE Command** - SQL standard MERGE implementation for conditional inserts/updates/deletes
- **SQL/JSON Enhancements** - Expanded JSON path expressions and processing capabilities
- **Improved Query Planning** - Better statistics and cost estimation for complex queries
- **Incremental Sorting** - Performance optimization for partially sorted data
- **ICU Collation Support** - International Components for Unicode collation integration
- **Pluggable Table Storage Interface** - Foundation for custom table access methods
- **WAL Compression** - Write-ahead log compression for reduced storage requirements
- **Performance Monitoring** - Enhanced pg_stat views and monitoring capabilities

## Key Skills and Expertise
- **Advanced SQL Query Design** with focus on PostgreSQL-specific features
- **Database Schema Optimization** principles and techniques
- **Transaction Management** with isolation levels
- **PostgreSQL Indexing** strategies including GIN, GiST, and BRIN indexes
- **Performance Tuning** of queries and server configuration
- **Extension Development** and usage
- **Stored Procedures and Functions** in PL/pgSQL and other languages
- **JSON and JSONB Data Handling** for document storage needs
- **Full-Text Search** implementation with PostgreSQL
- **Partitioning Strategies** for large datasets

## Best Practices
- Use appropriate index types for different query patterns (B-tree, GIN, GiST)
- Implement proper constraints for data integrity
- Utilize EXPLAIN ANALYZE for query performance analysis
- Leverage Common Table Expressions for complex queries
- Use the appropriate transaction isolation level for your use case
- Implement proper connection pooling
- Utilize partitioning for very large tables
- Implement regular VACUUM and ANALYZE maintenance
- Use prepared statements for query execution
- Leverage extensions for specialized functionality

## File Types
- SQL script files (.sql)
- Backup files (.dump, .backup)
- Configuration files (postgresql.conf, pg_hba.conf)
- Log files (PostgreSQL server logs)
- Data files (base directory files)
- WAL segment files
- Tablespace files
- SSL certificate files (.pem, .crt, .key)
- Extension control files (.control)
- PL/pgSQL function files (.sql)

## Related Packages
- PostgreSQL Server ^16.0.0
- psycopg2 ^2.9.9
- node-postgres ^8.11.3
- pg_dump/pg_restore ^16.0.0
- PostGIS ^3.4.0
- pgAdmin 4 ^7.8.0
- pgBouncer ^1.21.0
- pg_partman ^4.7.4
- TimescaleDB ^2.13.0
- pgvector ^0.5.1

## Differences From Previous Version
- **New APIs**:
  - SQL standard MERGE command implementation
  - Enhanced logical replication API with row filtering
  - Extended JSON processing functions
  
- **Enhanced Features**:
  - Improved parallel query execution
  - Better query planner with more accurate statistics
  - Incremental sort optimization
  - WAL compression for storage efficiency
  - Enhanced monitoring capabilities

## Custom Instructions
When working with PostgreSQL 16.0, focus on leveraging its advanced SQL capabilities and robust feature set to build powerful database applications. This version introduces significant performance improvements and new features that enhance PostgreSQL's already strong capabilities. For schema design, take advantage of PostgreSQL's rich data type system, including arrays, ranges, and JSON types, to model your data accurately. Use inheritance and partitioning for managing large datasets effectively. Implement appropriate indexing strategies based on your query patterns, using B-tree indexes for equality and range conditions, GIN indexes for full-text search and JSONB containment queries, GiST indexes for geometric data and similarity searches, and BRIN indexes for large tables with naturally clustered data. For complex queries, utilize Common Table Expressions (CTEs) and window functions to write more readable and maintainable SQL. Take advantage of the new MERGE command for upsert operations instead of the older ON CONFLICT syntax when appropriate. For storing document data, prefer JSONB over regular JSON for its indexing capabilities and more efficient storage. When implementing full-text search, use PostgreSQL's built-in text search functionality with appropriate language-specific dictionaries and configurations. For high availability and read scaling, configure logical replication with the enhanced row filtering capabilities in PostgreSQL 16. Optimize your transactions by choosing the appropriate isolation level based on your consistency requirements, and implement connection pooling using pgBouncer or similar tools to manage database connections efficiently.