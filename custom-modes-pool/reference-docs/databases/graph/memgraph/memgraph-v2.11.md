# Memgraph v2.11 Developer Mode

## Version-Specific Features
- **In-Memory Graph Database** - High-performance graph processing with in-memory architecture
- **Cypher Query Language** - Standard graph query language with Memgraph-specific extensions
- **Stream Processing** - Native integration with Kafka and Pulsar for real-time data processing
- **HTAP Capabilities** - Hybrid Transactional/Analytical Processing for real-time analytics
- **Custom Query Modules** - Extensibility through Python, C++, and Rust modules
- **Query Modules Library** - Pre-built algorithms and procedures for graph analytics
- **OpenCypher Compatibility** - Standardized graph query language support
- **Bolt Protocol** - Client communication using the Neo4j Bolt protocol
- **Enterprise Security** - Fine-grained access control and authentication
- **Monitoring and Metrics** - Prometheus integration for observability

## Key Skills and Expertise
- **Graph Data Modeling** for in-memory processing
- **Cypher Query Language** with Memgraph extensions
- **Streaming Data Integration** particularly with Kafka
- **Real-time Analytics** using graph algorithms
- **Python Module Development** for custom procedures
- **Performance Tuning** for in-memory graph operations
- **Graph Theory** concepts and algorithms
- **Stream Processing Patterns** for continuous computation
- **Distributed Systems** concepts for scaled deployments
- **Data Visualization** for graph exploration

## Best Practices
- Design graph schema for efficient in-memory representation
- Use query modules for complex algorithms rather than pure Cypher
- Implement efficient indexes for frequently queried properties
- Structure Cypher queries for optimal execution plans
- Use parameterized queries to prevent injection and improve performance
- Configure appropriate memory settings based on dataset size
- Leverage streaming capabilities for real-time data processing
- Implement custom procedures for operations not available in Cypher
- Use trigger functions for maintaining derived graph properties
- Monitor memory usage to stay within available resources

## File Types
- Memgraph database files
- Cypher script files (.cypher)
- Configuration files (memgraph.conf)
- Python module files (.py)
- C++ module files (.cpp, .h)
- Rust module files (.rs)
- CSV import files (.csv)
- Log files (Memgraph server logs)
- Backup files (.mg_backup)
- Data visualization exports (.json, .graphml)

## Related Packages
- Memgraph Platform ^2.11.0
- Memgraph Lab ^2.6.0
- MAGE (Memgraph Advanced Graph Extensions) ^1.13.0
- mgclient (C client) ^1.3.0
- GQLAlchemy (Python OGM) ^1.8.0
- pymgclient (Python driver) ^1.2.0
- Memgraph Bolt (JavaScript driver) ^0.6.0
- Kafka Connect Memgraph Sink ^1.0.0
- Docker ^20.10.0
- Kubernetes operators

## Differences From Previous Version
- **New APIs**:
  - Enhanced stream processing connectors
  - Extended Python query module API
  - New graph algorithm implementations
  
- **Enhanced Features**:
  - Improved query performance for complex traversals
  - Better memory management for large graphs
  - Enhanced Cypher query planner
  - More sophisticated monitoring capabilities
  - Improved security features with fine-grained access control

## Custom Instructions
When working with Memgraph 2.11, focus on leveraging its in-memory architecture and streaming capabilities for high-performance, real-time graph applications. Memgraph excels at use cases requiring low-latency graph operations on continuously updated data, such as fraud detection, network monitoring, and recommendation systems with real-time components. Begin by designing your graph model with in-memory performance in mind, structuring your data to minimize traversal depth for common queries. Use appropriate indexes on properties used in filtering operations, but be judicious as indexes consume additional memory. For complex analytical operations, leverage the MAGE (Memgraph Advanced Graph Extensions) library which provides optimized implementations of common graph algorithms rather than writing them in pure Cypher. When implementing stream processing, use the Memgraph Kafka or Pulsar connectors to ingest continuous data and design transformation modules that efficiently update the graph structure as new events arrive. For extending Memgraph's capabilities, develop custom query modules in Python for rapid development or C++/Rust for maximum performance. Structure your Cypher queries to take advantage of Memgraph's query planner, using EXPLAIN to analyze execution plans. Monitor your instance's memory usage and query performance using the built-in metrics and Prometheus integration. For production deployments, configure appropriate replication for high availability and implement regular backup procedures using Memgraph's backup facilities. When visualizing your graph data, utilize Memgraph Lab for development and exploration, or connect to third-party visualization tools for production dashboards.