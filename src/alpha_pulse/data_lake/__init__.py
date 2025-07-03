"""
Data Lake module for AlphaPulse.

Provides scalable storage and processing infrastructure for
historical market data with Bronze/Silver/Gold architecture.
"""

from alpha_pulse.data_lake.lake_manager import (
    DataLakeManager,
    DataLakeConfig,
    StorageBackend,
    DataFormat
)

from alpha_pulse.data_lake.storage_layers import (
    DataLayer,
    BronzeLayer,
    SilverLayer,
    GoldLayer
)

from alpha_pulse.data_lake.data_catalog import (
    DataCatalog,
    DatasetMetadata,
    DatasetSchema,
    DatasetType,
    DatasetStatus
)

from alpha_pulse.data_lake.partitioning_strategy import (
    PartitioningStrategy,
    PartitionConfig,
    PartitionScheme
)

from alpha_pulse.data_lake.compression_handler import (
    CompressionHandler,
    CompressionProfile,
    CompressionType
)

from alpha_pulse.data_lake.ingestion_pipeline import (
    IngestionPipeline,
    BatchIngestionPipeline,
    StreamingIngestionPipeline,
    IncrementalIngestionPipeline,
    IngestionOrchestrator,
    IngestionConfig,
    IngestionJob,
    IngestionMode,
    IngestionStatus
)

from alpha_pulse.utils.data_lake_utils import DataLakeUtils

from alpha_pulse.config.data_lake_config import (
    get_data_lake_config,
    DataLakeConfigFactory,
    DataLakeEnvironment,
    StorageClass
)

__all__ = [
    # Lake Manager
    "DataLakeManager",
    "DataLakeConfig",
    "StorageBackend",
    "DataFormat",
    
    # Storage Layers
    "DataLayer",
    "BronzeLayer",
    "SilverLayer",
    "GoldLayer",
    
    # Data Catalog
    "DataCatalog",
    "DatasetMetadata",
    "DatasetSchema",
    "DatasetType",
    "DatasetStatus",
    
    # Partitioning
    "PartitioningStrategy",
    "PartitionConfig",
    "PartitionScheme",
    
    # Compression
    "CompressionHandler",
    "CompressionProfile",
    "CompressionType",
    
    # Ingestion
    "IngestionPipeline",
    "BatchIngestionPipeline",
    "StreamingIngestionPipeline",
    "IncrementalIngestionPipeline",
    "IngestionOrchestrator",
    "IngestionConfig",
    "IngestionJob",
    "IngestionMode",
    "IngestionStatus",
    
    # Utilities
    "DataLakeUtils",
    
    # Configuration
    "get_data_lake_config",
    "DataLakeConfigFactory",
    "DataLakeEnvironment",
    "StorageClass"
]