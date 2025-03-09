# Redis v7.2 Developer Mode

## Version-Specific Features
- **Redis Functions** - Serverside JavaScript and Lua scripting for custom logic execution
- **Redis Streams** - Log data structure for event sourcing and messaging
- **Redis ACL** - Fine-grained access control with user management
- **Redis Cluster** - Distributed implementation with automatic sharding
- **Pub/Sub Messaging** - Pattern-based publisher/subscriber messaging system
- **Redis Search** - Full-text search engine with secondary indexing
- **Redis Time Series** - Time series data storage and querying
- **Redis JSON** - Native JSON document support
- **Probabilistic Data Structures** - HyperLogLog, Bloom Filter, and Count-Min Sketch
- **Redis Stack** - Extended functionality with modules for advanced use cases

## Key Skills and Expertise
- **Redis Command Syntax** and data type operations
- **Data Structure Selection** for different use cases
- **Caching Strategies** implementation
- **Lua Scripting** for atomic operations
- **Redis Cluster Configuration** for distributed deployments
- **Memory Management** techniques
- **Persistence Configuration** with RDB and AOF
- **High Availability Setup** with Redis Sentinel
- **Performance Tuning** for different workloads
- **Client Library Usage** in various programming languages

## Best Practices
- Select appropriate data structures for your use case
- Implement proper key naming conventions
- Use pipelining for bulk operations
- Leverage server-side Lua scripts for atomic operations
- Configure appropriate eviction policies
- Implement proper persistence strategy (RDB vs AOF)
- Use Redis Cluster for horizontal scaling
- Implement proper exception handling in client code
- Monitor memory usage to prevent swapping
- Implement proper backup and recovery procedures

## File Types
- Redis database files (.rdb)
- Append-only files (.aof)
- Redis configuration files (redis.conf)
- Lua script files (.lua)
- Redis module binary files (.so)
- Log files (Redis server logs)
- Backup files (.rdb.bak)
- Cluster configuration files (nodes.conf)
- Client application code integrating Redis
- Docker configuration for Redis deployments

## Related Packages
- Redis Server ^7.2.0
- Redis Stack ^7.2.0-v9
- Redis Modules (RedisJSON, RediSearch, RedisTimeSeries, RedisGraph, RedisBloom)
- Redis Sentinel
- Redis Cluster
- Redis CLI
- Jedis (Java) ^5.0.0
- redis-py (Python) ^5.0.0
- node-redis (Node.js) ^4.6.7
- StackExchange.Redis (.NET) ^2.7.4
- go-redis (Go) ^9.3.0

## Differences From Previous Version
- **New APIs**:
  - Enhanced functions with improved JavaScript support
  - Advanced access control commands
  - Improved cluster management API
  
- **Enhanced Features**:
  - Better memory efficiency
  - More sophisticated eviction policies
  - Improved client tracking
  - Enhanced monitoring capabilities
  - More powerful search functionality

## Custom Instructions
When working with Redis 7.2, focus on leveraging its high-performance, in-memory data structure store for caching, real-time analytics, messaging, and other scenarios requiring microsecond response times. Redis excels at operations requiring low latency and high throughput. Choose the appropriate Redis data structure for your specific use case: Strings for simple values and counters, Lists for queues and timelines, Sets for unique collections, Sorted Sets for ranked data, Hashes for structured objects, and Streams for messaging and event sourcing. For complex operations requiring atomicity, implement Lua scripts or Redis Functions, which execute directly on the server without network roundtrips. When implementing caching, consider appropriate TTL (time-to-live) settings and eviction policies to manage memory efficiently. For distributed applications, leverage Redis Cluster for horizontal scaling, but be aware of its limitations with multi-key operations spanning different hash slots. Implement proper error handling and connection management in your client code, including connection pooling and retry logic. For data persistence, choose between RDB snapshots (point-in-time backups with configurable frequency) and AOF logging (continuous write operation logging) based on your durability requirements. If implementing a messaging system, decide between Pub/Sub (for fire-and-forget messaging) and Streams (for durable, consumer group-based processing). For full-text search capabilities, utilize RediSearch rather than building custom solutions. Monitor Redis performance using the INFO command and tools like Redis-Stat, paying particular attention to memory usage, hit rate, and command execution times.