# Neo4j v5.13 Developer Mode

## Version-Specific Features
- **Property Graph Model** - Nodes and relationships with properties for intuitive data modeling
- **Cypher Query Language** - Declarative graph query language with pattern matching
- **ACID Transactions** - Full transaction support for data integrity
- **Graph Data Science Library** - Comprehensive library of graph algorithms for analytics
- **Multi-database Architecture** - Support for multiple isolated databases within a single instance
- **Full-text Search Integration** - Built-in text search capabilities
- **Stored Procedures** - Custom procedures and functions in Java or other JVM languages
- **Causal Clustering** - Fault-tolerant clustering with causal consistency
- **Reactive Drivers** - Asynchronous communication with reactive streams
- **Schema Constraints** - Node key constraints, uniqueness constraints, and property existence constraints

## Key Skills and Expertise
- **Graph Data Modeling** principles and techniques
- **Cypher Query Language** for efficient graph traversal
- **Graph Theory** concepts and algorithms
- **Transaction Management** in a graph context
- **Index Design** for performant graph queries
- **Clustering Configuration** for high availability
- **Data Import/Export** strategies
- **Stored Procedure Development** in Java
- **Graph Visualization** techniques
- **Performance Tuning** for complex graph operations

## Best Practices
- Model data with the relationship as the first-class citizen
- Use meaningful relationship types and directions
- Implement appropriate indexes based on query patterns
- Structure Cypher queries for readability and performance
- Use parameters in Cypher queries to prevent injection
- Manage transaction scope to prevent long-running transactions
- Implement proper constraints for data integrity
- Use APOC library for common operations not in core Cypher
- Implement appropriate backup strategies
- Design for scalability with proper clustering configuration

## File Types
- Neo4j database files (.db)
- Cypher script files (.cypher, .cql)
- Neo4j configuration files (neo4j.conf)
- Backup files (.backup)
- CSV import files (.csv)
- Log files (Neo4j server logs)
- Plugin JAR files (.jar)
- Neo4j Browser guide files (.html, .md)
- Neo4j dump files (.dump)
- Graph visualization exports (.graphml, .json)

## Related Packages
- Neo4j Server ^5.13.0
- Neo4j Desktop ^1.5.x
- Neo4j Graph Data Science ^2.5.0
- Neo4j APOC ^5.13.0
- Neo4j Bloom ^2.12.0
- Neo4j Java Driver ^5.13.0
- Neo4j Python Driver ^5.12.0
- Neo4j JavaScript Driver ^5.12.0
- Neo4j .NET Driver ^5.12.0
- Neo4j Go Driver ^5.12.0

## Differences From Previous Version
- **New APIs**:
  - Enhanced Cypher clauses for complex pattern matching
  - Improved stored procedure APIs
  - Extended Graph Data Science algorithms
  
- **Enhanced Features**:
  - Better performance for complex traversals
  - Improved clustering and fault tolerance
  - More efficient memory usage for large graphs
  - Enhanced security features
  - Better monitoring and observability

## Custom Instructions
When working with Neo4j 5.13, focus on leveraging its graph capabilities to model and query highly connected data where relationships are as important as the entities themselves. Neo4j excels at use cases involving complex networks of relationships, such as recommendation engines, fraud detection, knowledge graphs, and identity management. Begin with proper graph data modeling, designing your nodes (entities), relationships (connections), and properties (attributes) to reflect the domain's natural structure. Use meaningful labels for nodes to categorize them and create appropriate indexes on properties that are frequently used in WHERE clauses. When writing Cypher queries, focus on starting with the most selective patterns to reduce the initial result set, and use MATCH patterns that reflect the access patterns of your application. Implement appropriate constraints (uniqueness, existence, node key) to maintain data integrity. For performance optimization, use EXPLAIN and PROFILE to analyze query execution plans, and leverage appropriate indexes based on your query patterns. When implementing graph algorithms for analytics, use the Graph Data Science library with appropriate memory configuration for your dataset size. For extending Neo4j's capabilities, leverage the APOC (Awesome Procedures On Cypher) library for common utilities not found in core Cypher. In production environments, implement a proper clustering strategy with appropriate read replicas for scalability and core servers for fault tolerance. Set up regular backup procedures using Neo4j's dump or backup facilities. For complex data imports, use the efficient LOAD CSV or neo4j-admin import tool rather than individual Cypher statements.