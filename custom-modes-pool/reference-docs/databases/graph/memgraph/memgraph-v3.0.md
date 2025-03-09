# Memgraph v3.0 Developer Mode

## Version-Specific Features
- **Vector Search** - Core capability for similarity and relevance-based graph search in a unified system
- **GraphRAG Support** - Combines vector search with knowledge graphs for enhanced AI reasoning
- **In-Memory Graph Database** - High-performance graph processing with in-memory architecture
- **Dynamic Algorithms** - Real-time data analysis for high-throughput use cases without LLM retraining
- **Cypher Query Language** - Standard graph query language with Memgraph-specific extensions
- **Stream Processing** - Native integration with Kafka and Pulsar for real-time data processing
- **HTAP Capabilities** - Hybrid Transactional/Analytical Processing for real-time analytics
- **Custom Query Modules** - Extensibility through Python, C++, and Rust modules
- **GraphChat Enhancements** - Natural language queries translated to Cypher with LLM context integration
- **Improved Performance** - Optimized replication recovery and faster query execution

## Key Skills and Expertise
- **Vector Embeddings** for similarity-based search
- **Graph Data Modeling** for in-memory processing
- **AI Integration** particularly with LLMs and RAG workflows
- **Cypher Query Language** with Memgraph extensions
- **Streaming Data Integration** particularly with Kafka
- **Real-time Analytics** using graph algorithms
- **Python Module Development** for custom procedures
- **Performance Tuning** for in-memory graph operations
- **Graph Theory** concepts and algorithms
- **Stream Processing Patterns** for continuous computation

## Best Practices
- Leverage vector search for similarity-based queries and AI applications
- Use GraphRAG to enhance LLM reasoning with graph context
- Implement IN_MEMORY_ANALYTICAL storage mode for faster data import
- Utilize EXPLAIN clause to optimize query performance
- Design graph schema for efficient in-memory representation
- Use query modules for complex algorithms rather than pure Cypher
- Implement efficient indexes for frequently queried properties
- Structure Cypher queries for optimal execution plans
- Use parameterized queries to prevent injection and improve performance
- Leverage streaming capabilities for real-time data processing

## File Types
- Memgraph database files
- Cypher script files (.cypher)
- CSV import files (.csv)
- JSON data files (.json)
- CYPHERL query files (.cypherl)
- Configuration files (memgraph.conf)
- Python module files (.py)
- C++ module files (.cpp, .h)
- Rust module files (.rs)
- Log files (Memgraph server logs)
- Backup files (.mg_backup)
- Data visualization exports (.json, .graphml)

## Related Packages
- Memgraph Platform ^3.0.0
- Memgraph Lab ^3.0.0
- MAGE (Memgraph Advanced Graph Extensions) ^2.0.0
- GQLAlchemy (Python OGM) ^2.0.0
- pymgclient (Python driver)
- Memgraph Bolt (JavaScript driver)
- DeepSeek models integration
- Kafka Connect Memgraph Sink
- Docker
- Kubernetes operators

## Differences From Previous Version
- **New APIs**:
  - Vector search for similarity-based operations
  - GraphRAG support for enhanced AI reasoning
  - Dynamic algorithms for real-time analysis
  - Natural language interface through GraphChat
  
- **Enhanced Features**:
  - Improved query performance
  - Optimized replication recovery
  - Better performance under heavy load
  - Enhanced security with updated Python libraries
  - LLM integration for natural language queries

## Custom Instructions
When working with Memgraph 3.0, focus on leveraging its new vector search capabilities and GraphRAG support to build advanced AI applications with enhanced reasoning abilities. This version represents a significant evolution in combining graph database technology with AI capabilities, particularly for enterprise-specific contexts. Begin by designing your vector embeddings strategy to represent your graph data in a way that captures semantic meaning and enables similarity-based search. Utilize the GraphRAG capabilities to enhance language models by providing them with relevant graph context, which reduces hallucinations and improves reasoning. Take advantage of the dynamic algorithms for real-time analysis of streaming data without requiring LLM retraining. When designing your graph model, continue to focus on in-memory performance, structuring your data to minimize traversal depth for common queries. Use the IN_MEMORY_ANALYTICAL storage mode for faster data import operations. For query optimization, always use the EXPLAIN clause to understand the execution plan before running complex queries. Leverage the enhanced GraphChat in Memgraph Lab 3.0 to explore your graph using natural language questions, which are translated into Cypher queries and executed against your knowledge graph. For extending Memgraph's capabilities, continue developing custom query modules in Python for rapid development or C++/Rust for maximum performance. Monitor your instance's memory usage and query performance using the built-in metrics and integrate with Prometheus for comprehensive observability. For production deployments, benefit from the improved replication recovery for more efficient failover handling, and implement regular backup procedures using Memgraph's backup facilities.