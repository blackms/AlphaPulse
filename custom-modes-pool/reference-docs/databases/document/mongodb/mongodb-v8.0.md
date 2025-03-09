# MongoDB v8.0 Developer Mode

## Version-Specific Features
- **Performance Optimizations** - Up to 36% quicker reads and 59% higher throughput for updates
- **Enhanced Time Series** - 200% faster time-series aggregations and improved data handling
- **Queryable Encryption** - Expanded capabilities with support for range queries on encrypted data
- **Improved Resharding** - Up to 50 times faster data redistribution across shards
- **Aggregation Pipeline Enhancements** - New operators like $convert and $toUUID for improved data processing
- **Advanced Logging and Profiling** - Based on processing time rather than total latency
- **Vector Search Improvements** - Vector quantization in Atlas Vector Search
- **Scalability Enhancements** - Lower startup costs for scaling and better spike handling
- **Improved Replication** - 20% faster replication for better data consistency
- **Query Performance Analysis** - Enhanced metrics and tooling for slow query detection

## Key Skills and Expertise
- **MongoDB Query Language** (MQL) for document retrieval
- **Aggregation Pipeline** design for data transformation
- **Indexing Strategies** for query optimization
- **Schema Design** for document databases
- **Data Modeling** with embedded documents and references
- **Transaction Management** in distributed environments
- **Replication Configuration** for high availability
- **Sharding Implementation** for horizontal scaling
- **Security Configuration** including queryable encryption
- **Performance Tuning** of queries and server settings

## Best Practices
- Design document schemas based on access patterns
- Create compound indexes that support your query patterns
- Utilize index suggestions from the Performance Advisor
- Use appropriate embedding vs. referencing strategies for related data
- Implement appropriate write concern and read preference settings
- Leverage aggregation pipeline for complex data transformation
- Implement proper sharding key selection for distributed data
- Use profiling tools to identify and optimize slow queries
- Implement Queryable Encryption for sensitive data with range queries
- Remove redundant indexes to improve performance

## File Types
- JSON and BSON documents
- JavaScript files (.js) for queries and scripts
- MongoDB configuration files (.conf)
- Log files (MongoDB server logs)
- Dump and backup files (.dump, .archive)
- Index definition files (.json)
- Certificate files (.pem, .crt, .key)
- CSV/TSV import/export files (.csv, .tsv)
- MongoDB Charts and Compass saved configurations
- Shell script files for automation (.sh, .bat)

## Related Packages
- MongoDB Server ^8.0.1
- MongoDB Atlas Cloud Service
- MongoDB Compass ^1.42.0
- MongoDB Shell (mongosh) ^2.1.0
- MongoDB Node.js Driver ^6.0.0
- MongoDB Python Driver (PyMongo) ^4.7.0
- MongoDB Java Driver ^5.0.0
- MongoDB C# Driver ^2.23.0
- Mongoose ODM ^8.1.0
- MongoDB Realm ^11.0.0
- mongodb-org-database ^8.0.1
- mongodb-org-server ^8.0.1
- mongodb-org-mongos ^8.0.1
- mongodb-org-tools ^8.0.1

## Differences From Previous Version
- **Performance Improvements**:
  - 36% quicker reads and 59% higher throughput for updates
  - 200% faster time-series aggregations
  - 54% faster bulk inserts and 20% faster replication
  - Up to 50 times faster resharding
  
- **Enhanced Features**:
  - Queryable Encryption now supports range queries
  - New aggregation operators for binary data conversion ($convert) and UUID handling ($toUUID)
  - Improved logging and profiling based on processing time
  - Better metrics for slow query analysis
  - Lower startup costs for scaling
  - 25% better overall throughput and latency across various use cases

## Custom Instructions
When working with MongoDB 8.0, focus on leveraging its significant performance improvements and enhanced features to build high-performance, scalable applications. This version introduces substantial performance optimizations, with up to 36% quicker reads, 59% higher throughput for updates, and 200% faster time-series aggregations. Take advantage of the enhanced Queryable Encryption which now supports range queries, allowing for more flexible operations on sensitive encrypted data. For data modeling, continue following MongoDB's document-oriented approach, designing schemas based on access patterns and utilizing the flexible document structure. Implement proper indexing strategies and regularly review your indexes using the Performance Advisor to remove redundant ones. When working with the aggregation pipeline, leverage new operators like $convert for binary data conversion and $toUUID for simplified UUID handling. For performance analysis, utilize the improved logging and profiling capabilities which now focus on processing time rather than total latency, providing better insights into slow operations. When implementing sharded clusters, take advantage of the significantly faster resharding capabilities (up to 50 times faster) and lower startup costs. For time-series applications, leverage the enhanced performance in time-series collections. When dealing with large-scale deployments, utilize the improved replication speed (20% faster) for better data consistency. For security implementation, continue using role-based access control and consider implementing Queryable Encryption for sensitive data, especially now that it supports range queries. Monitor your database performance using the enhanced metrics, particularly the new queues.execution.totalTimeQueuedMicros metric for better slow query analysis.