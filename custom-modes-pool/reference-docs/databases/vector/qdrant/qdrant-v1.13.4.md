# Qdrant v1.13.4 Developer Mode

## Version-Specific Features
- **GPU-Accelerated Indexing** - Fast HNSW indexing with support for NVIDIA, AMD, and Intel GPUs
- **Strict Mode** - Enforced operation restrictions on collections for enhanced control
- **HNSW Graph Compression** - Reduced storage usage via Delta Encoding
- **Named Vector Filtering** - New has_vector filtering condition for collections with named vectors
- **Custom Storage** - Constant-time reads/writes of payloads and sparse vectors
- **Vector Similarity Search** - Efficient nearest neighbor search for high-dimensional vectors
- **Multiple Vector Indexes** - Support for HNSW, IVF, and quantization indexes for performance optimization
- **Query Planning** - Automatic optimization of search queries with filters
- **Distributed Architecture** - Horizontal scaling with sharding and replication with improved consensus
- **Consistent Snapshots** - Point-in-time backup and recovery

## Key Skills and Expertise
- **Vector Space Models** and embedding generation
- **Similarity Search Algorithms** particularly HNSW and IVF
- **GPU Computing** for accelerated vector operations
- **Distributed Systems** concepts for scaled deployments
- **Query Optimization** for vector searches
- **Metadata Management** with payload filtering
- **High-dimensional Data** principles and challenges
- **REST API Integration** for vector operations
- **gRPC Communication** for high-performance scenarios
- **Machine Learning Integration** particularly for embeddings

## Best Practices
- Leverage GPU-accelerated indexing for large datasets to reduce indexing times
- Implement Strict Mode for collections requiring tight operational control
- Use HNSW Graph Compression to reduce storage requirements
- Utilize Named Vector Filtering for collections with heterogeneous data
- Optimize search parameters like hnsw_ef and exact to balance speed and precision
- Adjust the number of segments based on priority (latency vs. throughput)
- Implement proper vector normalization before insertion
- Use payload indexes for efficient filtering
- Implement batch operations for bulk data loading
- Monitor performance metrics using Prometheus integration

## File Types
- Vector data files (.npy, .bin)
- Collection snapshot files (.snapshot)
- Configuration files (config.yaml, .toml, .json, .ini)
- TLS certificate files (.pem)
- Backup archives (.tar.gz)
- Schema definition files (.json)
- Docker configuration (Dockerfile, docker-compose.yml)
- REST API specification files (OpenAPI)
- Client application code integrating Qdrant
- Python wheel files (.whl) for client libraries

## Related Packages
- Qdrant Server ^1.13.4
- qdrant-client (Python) ^1.13.2
- qdrant-js (Node.js)
- grpcio (for gRPC communication)
- pydantic (for data validation)
- httpx (for HTTP communication)
- sentence-transformers
- OpenAI embedding models
- Docker
- Prometheus and Grafana

## Differences From Previous Version
- **New APIs**:
  - GPU-accelerated indexing on NVIDIA, AMD, and Intel GPUs
  - Strict Mode for enforcing operation restrictions
  - Named Vector Filtering with has_vector condition
  - Custom Storage with constant-time operations
  
- **Enhanced Features**:
  - HNSW Graph Compression via Delta Encoding
  - Maximum points limit in collections with strict mode
  - Improved consensus compaction for faster peer joining and recovery
  - Enhanced data consistency measures
  - Better rate limiting and error handling in REST API

## Custom Instructions
When working with Qdrant 1.13.4, focus on leveraging its advanced vector database capabilities, particularly the new GPU-accelerated indexing which significantly reduces indexing times across NVIDIA, AMD, and Intel GPUs. This version introduces Strict Mode, allowing you to enforce operation restrictions on collections for enhanced control and set maximum points limits. Take advantage of HNSW Graph Compression using Delta Encoding to reduce storage requirements while maintaining search performance. For collections with heterogeneous data, utilize the new Named Vector Filtering with the has_vector condition to efficiently query based on vector availability. The Custom Storage feature enables constant-time reads and writes of payloads and sparse vectors, improving performance for complex data structures. When designing your vector search implementation, carefully balance parameters like hnsw_ef and exact based on your precision vs. speed requirements. For large-scale deployments, leverage the improved consensus mechanisms for faster cluster operations, peer joining, and recovery. Continue using best practices from previous versions including proper vector normalization, effective payload indexing, and batched operations for bulk data. For monitoring, implement comprehensive observability using the Prometheus integration to track performance metrics. When integrating with client applications, select the appropriate client library for your language, with the Python qdrant-client (version 1.13.2) being the most feature-complete. For production environments, implement a robust backup strategy using the snapshot functionality, and consider using Docker and Kubernetes for containerized deployments with the available Helm charts.