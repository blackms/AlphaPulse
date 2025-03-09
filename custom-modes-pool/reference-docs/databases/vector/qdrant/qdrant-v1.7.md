# Qdrant v1.7 Developer Mode

## Version-Specific Features
- **Vector Similarity Search** - Efficient nearest neighbor search for high-dimensional vectors
- **Multiple Vector Indexes** - Support for HNSW, IVF, and quantization indexes for performance optimization
- **Payload Filtering** - Advanced filtering of search results based on metadata
- **Query Planning** - Automatic optimization of search queries with filters
- **Multitenancy** - Collection-based isolation for multiple applications
- **Distributed Architecture** - Horizontal scaling with sharding and replication
- **Consistent Snapshots** - Point-in-time backup and recovery
- **REST API and gRPC** - Multiple integration interfaces
- **Batched Updates** - Efficient bulk operations for vectors and payloads
- **Full observability** - Prometheus metrics, OpenTelemetry tracing, and structured logging

## Key Skills and Expertise
- **Vector Space Models** and embedding generation
- **Similarity Search Algorithms** particularly HNSW and IVF
- **Distributed Systems** concepts for scaled deployments
- **Query Optimization** for vector searches
- **Metadata Management** with payload filtering
- **High-dimensional Data** principles and challenges
- **REST API Integration** for vector operations
- **gRPC Communication** for high-performance scenarios
- **Machine Learning Integration** particularly for embeddings
- **Docker and Kubernetes** for containerized deployments

## Best Practices
- Choose appropriate vector index parameters based on your dataset size and query latency requirements
- Implement proper vector normalization before insertion
- Use payload indexes for efficient filtering
- Structure payloads to support your query patterns
- Implement batch operations for bulk data loading
- Set appropriate shard sizes and replication factors for distributed deployments
- Use scrolling API for large result sets
- Configure proper resource limits for collections
- Implement regular snapshots for data backup
- Monitor performance metrics using Prometheus integration

## File Types
- Vector data files (.npy, .bin)
- Collection snapshot files (.snapshot)
- Configuration files (config.yaml)
- Backup archives (.tar.gz)
- Schema definition files (.json)
- Docker configuration (Dockerfile, docker-compose.yml)
- REST API specification files (OpenAPI)
- Client application code integrating Qdrant
- Log files (Qdrant server logs)
- Metrics data (Prometheus, Grafana dashboards)

## Related Packages
- Qdrant Server ^1.7.0
- qdrant-client (Python) ^1.7.0
- qdrant-js (Node.js) ^1.7.0
- gRPC clients (various languages)
- sentence-transformers ^2.2.2
- OpenAI text-embedding-ada-002
- FAISS ^1.7.4
- Kubernetes operators
- Docker ^20.10.0
- Prometheus and Grafana

## Differences From Previous Version
- **New APIs**:
  - Enhanced filtering capabilities with advanced query planning
  - Improved batch operations API
  - New snapshot management endpoints
  
- **Enhanced Features**:
  - Better performance for filtered searches
  - More efficient memory usage for large collections
  - Improved distributed coordination
  - Enhanced observability with detailed metrics
  - Better consistency guarantees for distributed deployments

## Custom Instructions
When working with Qdrant 1.7, focus on leveraging its high-performance vector search capabilities for similarity-based retrieval in AI applications. Qdrant excels at storing and searching vector embeddings with associated payload data, making it ideal for semantic search, recommendation systems, and other machine learning applications. Start by designing your vector space model, determining the appropriate embedding model (like OpenAI's text-embedding-ada-002 or sentence-transformers) and dimension size. When creating collections, carefully configure your vector parameters including dimension and distance metric (cosine, euclidean, or dot product) based on your embedding model. For performance optimization, select the appropriate index type: HNSW for better recall at the cost of memory usage, or quantization for memory efficiency with slightly lower precision. Structure your payload data to support your filtering needs, creating payload indexes for frequently filtered fields. When implementing search queries, balance similarity thresholds with filtering constraints, and use the query planning capabilities to optimize complex searches. For production deployments, implement a distributed Qdrant cluster with appropriate sharding for horizontal scaling and replication for high availability. Monitor your instance using the built-in Prometheus metrics, tracking key indicators like query latency, memory usage, and disk I/O. Implement regular snapshots for data backup and disaster recovery. When integrating with application code, choose between the REST API for simplicity or gRPC for higher performance, and leverage the client libraries available for your programming language. For large-scale data operations, always use batched updates to minimize network overhead and improve throughput.