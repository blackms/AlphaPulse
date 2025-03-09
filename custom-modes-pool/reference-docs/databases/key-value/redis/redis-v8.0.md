# Redis v8.0 Developer Mode

## Version-Specific Features
- **Performance Improvements** - Up to 87% reduced command latency and 100% more throughput
- **Asynchronous I/O Threading** - New implementation for improved performance in single and multi-core environments
- **Enhanced Query Engine** - Supports fast querying, search, and vector search with both horizontal and vertical scaling
- **Vector Search** - Real-time high-precision search on billions of vector embeddings
- **Redis JSON** - Native JSON document support now integrated into the core
- **Redis Time Series** - Time series data storage and querying now built-in
- **Probabilistic Data Structures** - Five data structures including HyperLogLog, Bloom Filter, and Count-Min Sketch
- **Redis Streams** - Log data structure for event sourcing and messaging
- **Redis ACL** - Fine-grained access control with user management
- **Redis Cluster** - Distributed implementation with automatic sharding and improved replication

## Key Skills and Expertise
- **Redis Command Syntax** and data type operations
- **Data Structure Selection** for different use cases including JSON and time series
- **Caching Strategies** implementation
- **Vector Operations** for machine learning and AI applications
- **Redis Cluster Configuration** for distributed deployments
- **Memory Management** techniques
- **Persistence Configuration** with RDB and AOF
- **High Availability Setup** with Redis Sentinel
- **Performance Tuning** for different workloads
- **Client Library Usage** in various programming languages

## Best Practices
- Select appropriate data structures for your use case, including new native structures
- Implement proper key naming conventions
- Use pipelining for bulk operations
- Configure appropriate eviction policies
- Implement proper persistence strategy (RDB vs AOF)
- Use Redis Cluster for horizontal scaling
- Implement proper exception handling in client code
- Monitor memory usage to prevent swapping
- Utilize vector search capabilities for AI and machine learning applications
- Avoid expensive operations like KEYS in production environments

## File Types
- Redis database files (.rdb)
- Append-only files (.aof)
- Redis configuration files (redis.conf)
- Lua script files (.lua)
- JavaScript files for Redis functions (.js)
- Redis module binary files (.so)
- Log files (Redis server logs)
- Backup files (.rdb.bak)
- Cluster configuration files (nodes.conf)
- JSON and CSV files for data import/export

## Related Packages
- Redis Server ^8.0.0
- Redis CLI
- Redis Sentinel
- Redis Cluster
- Jedis (Java)
- redis-py (Python)
- node-redis (Node.js)
- StackExchange.Redis (.NET)
- go-redis (Go)

## Differences From Previous Version
- **Performance Enhancements**:
  - 87% reduction in command latency
  - 100% increase in throughput
  - 18% faster replication
  - Improved CRC64 calculations and large argument handling
  
- **Architecture Changes**:
  - JSON, Time Series, and probabilistic data structures integrated into the core
  - New asynchronous I/O threading implementation
  - Enhanced Query Engine with better scaling
  
- **New Capabilities**:
  - Advanced vector search for AI applications
  - Improved replication mechanism
  - Better latency for PFCOUNT, PFMERGE, GET, EXISTS, LRANGE, HSET, and HGETALL commands

## Custom Instructions
When working with Redis 8.0, focus on leveraging its significant performance improvements and enhanced capabilities to build high-performance applications. This version represents a major evolution with up to 87% reduced command latency and 100% more throughput, making it the fastest Redis release ever. Take advantage of the native integration of previously separate modules - JSON, Time Series, and probabilistic data structures - which are now built into the core. For AI and machine learning applications, explore the vector search capabilities which can efficiently handle billions of high-dimensional vectors in real-time. The new asynchronous I/O threading implementation improves performance in both single-core and multi-core environments, so consider this when deploying Redis in different infrastructures. When implementing data access patterns, leverage the enhanced Query Engine which now scales both horizontally and vertically. For distributed deployments, take advantage of the improved replication mechanism which is 18% faster and more robust than before. Continue to select the appropriate Redis data structures based on your specific use cases: Strings for simple values, Lists for queues, Sets for unique collections, Sorted Sets for ranked data, Hashes for structured objects, Streams for messaging, and now native JSON for document storage and Time Series for chronological data. Monitor performance using the INFO command and appropriate monitoring tools, with particular attention to the improved command latencies across various operations like PFCOUNT, PFMERGE, GET, EXISTS, LRANGE, HSET, and HGETALL. When implementing caching strategies, continue to use appropriate TTL settings and eviction policies to manage memory efficiently, and consider implementing proper persistence mechanisms based on your durability requirements.