# MongoDB v7.0 Developer Mode

## Version-Specific Features
- **Time Series Collections** - Enhanced support for time series data with improved indexing and querying
- **Queryable Encryption** - Client-side field level encryption with query capabilities on encrypted data
- **Serverless Instances** - Support for MongoDB Atlas serverless deployment model
- **Vector Search** - Built-in vector search capabilities for machine learning applications
- **Aggregation Pipeline Improvements** - New operators and performance optimizations
- **Cluster-to-Cluster Sync** - Bidirectional synchronization between MongoDB clusters
- **Secondary Indexing Enhancements** - More efficient indexing with reduced overhead
- **Atlas Search Facets** - Advanced search categorization capabilities
- **Wildcard Indexes** - Support for indexing across multiple document fields
- **Change Streams Enhancements** - Improved real-time data processing capabilities

## Key Skills and Expertise
- **MongoDB Query Language** (MQL) for document retrieval
- **Aggregation Pipeline** design for data transformation
- **Indexing Strategies** for query optimization
- **Schema Design** for document databases
- **Data Modeling** with embedded documents and references
- **Transaction Management** in distributed environments
- **Replication Configuration** for high availability
- **Sharding Implementation** for horizontal scaling
- **Security Configuration** including authentication and authorization
- **Performance Tuning** of queries and server settings

## Best Practices
- Design document schemas based on access patterns
- Create compound indexes that support your query patterns
- Use appropriate embedding vs. referencing strategies for related data
- Implement appropriate write concern and read preference settings
- Utilize change streams for real-time data processing
- Leverage aggregation pipeline for complex data transformation
- Implement proper sharding key selection for distributed data
- Use array operators for efficient array manipulation
- Implement proper authentication and access control
- Utilize optimistic concurrency for document updates

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
- MongoDB Server ^7.0.0
- MongoDB Atlas Cloud Service
- MongoDB Compass ^1.40.0
- MongoDB Shell ^2.0.0
- MongoDB Node.js Driver ^5.8.0
- MongoDB Python Driver (PyMongo) ^4.6.0
- MongoDB Java Driver ^4.11.0
- MongoDB C# Driver ^2.22.0
- Mongoose ODM ^8.0.0
- MongoDB Realm ^10.24.0

## Differences From Previous Version
- **New APIs**:
  - Vector search API for ML applications
  - Enhanced time series collections API
  - Queryable encryption interfaces
  
- **Enhanced Features**:
  - Improved aggregation pipeline performance
  - Better indexing efficiency and flexibility
  - Advanced Atlas Search capabilities
  - More powerful change streams
  - Enhanced security features

## Custom Instructions
When working with MongoDB 7.0, focus on leveraging its document-oriented architecture and powerful query capabilities to build scalable, flexible applications. This version introduces significant advancements in vector search, time series data, and encryption capabilities. For data modeling, design your schema based on access patterns rather than normalized relationships, taking advantage of MongoDB's flexible document structure. Use embedded documents when data is primarily accessed together and references when data is shared across multiple entities or when documents would grow too large. Implement appropriate indexing strategies based on your query patterns, utilizing compound indexes for queries with multiple conditions and covered indexes to avoid document lookups. For complex data processing, use the aggregation pipeline with its rich set of operators, taking advantage of the performance improvements in this version. When working with time series data, leverage the enhanced time series collections which provide optimized storage and querying for chronological data. For applications requiring machine learning integration, explore the new vector search capabilities to implement similarity search and semantic analysis. Implement proper security measures including authentication, role-based access control, and field-level encryption for sensitive data. For distributed deployments, carefully design your sharding strategy with appropriate shard keys to ensure even data distribution and efficient queries. When implementing real-time features, use change streams to react to data modifications. For database administration, monitor performance using MongoDB's built-in tools and establish proper backup procedures with appropriate retention policies.