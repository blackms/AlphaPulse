"""
Data Lake Manager for AlphaPulse.

Orchestrates the data lake operations including:
- Layer management (Bronze/Silver/Gold)
- Storage backend abstraction
- Data lifecycle management
- Query optimization
"""

import os
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import json
from abc import ABC, abstractmethod

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Optional Delta Lake support
try:
    from delta import DeltaTable, configure_spark_with_delta_pip
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    DELTA_AVAILABLE = True
except ImportError:
    DELTA_AVAILABLE = False
    DeltaTable = None
    configure_spark_with_delta_pip = None
    SparkSession = None
    SparkDataFrame = None

from alpha_pulse.config.config_loader import ConfigLoader
from alpha_pulse.data_lake.storage_layers import (
    DataLayer, BronzeLayer, SilverLayer, GoldLayer
)
from alpha_pulse.data_lake.partitioning_strategy import PartitioningStrategy
from alpha_pulse.data_lake.compression_handler import CompressionHandler


logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Supported storage backends."""
    AWS_S3 = "aws_s3"
    AZURE_DATA_LAKE = "azure_data_lake"
    GCP_CLOUD_STORAGE = "gcp_cloud_storage"
    LOCAL_FILE_SYSTEM = "local_file_system"


class DataFormat(Enum):
    """Supported data formats."""
    PARQUET = "parquet"
    DELTA = "delta"
    ORC = "orc"
    AVRO = "avro"
    JSON = "json"
    CSV = "csv"


@dataclass
class DataLakeConfig:
    """Configuration for data lake."""
    storage_backend: StorageBackend
    root_path: str
    bronze_path: str = "bronze"
    silver_path: str = "silver"
    gold_path: str = "gold"
    default_format: DataFormat = DataFormat.PARQUET
    compression: str = "snappy"
    enable_delta: bool = True
    retention_days: Dict[str, int] = field(default_factory=lambda: {
        "bronze": 2555,  # 7 years
        "silver": 1825,  # 5 years
        "gold": -1       # Permanent
    })
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DataLakeConfig':
        """Create from configuration dictionary."""
        return cls(
            storage_backend=StorageBackend(config.get("storage_backend", "local_file_system")),
            root_path=config["root_path"],
            bronze_path=config.get("bronze_path", "bronze"),
            silver_path=config.get("silver_path", "silver"),
            gold_path=config.get("gold_path", "gold"),
            default_format=DataFormat(config.get("default_format", "parquet")),
            compression=config.get("compression", "snappy"),
            enable_delta=config.get("enable_delta", True),
            retention_days=config.get("retention_days", {
                "bronze": 2555,
                "silver": 1825,
                "gold": -1
            })
        )


class StorageBackendInterface(ABC):
    """Abstract interface for storage backends."""
    
    @abstractmethod
    def read(self, path: str, format: DataFormat, **kwargs) -> pd.DataFrame:
        """Read data from storage."""
        pass
    
    @abstractmethod
    def write(self, df: pd.DataFrame, path: str, format: DataFormat, **kwargs):
        """Write data to storage."""
        pass
    
    @abstractmethod
    def list_files(self, path: str, pattern: Optional[str] = None) -> List[str]:
        """List files in path."""
        pass
    
    @abstractmethod
    def delete(self, path: str):
        """Delete file or directory."""
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        pass
    
    @abstractmethod
    def get_metadata(self, path: str) -> Dict[str, Any]:
        """Get file/directory metadata."""
        pass


class LocalFileSystemBackend(StorageBackendInterface):
    """Local file system storage backend."""
    
    def __init__(self, root_path: str):
        """Initialize local backend."""
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
    
    def read(self, path: str, format: DataFormat, **kwargs) -> pd.DataFrame:
        """Read data from local storage."""
        full_path = self.root_path / path
        
        if format == DataFormat.PARQUET:
            return pd.read_parquet(full_path, **kwargs)
        elif format == DataFormat.CSV:
            return pd.read_csv(full_path, **kwargs)
        elif format == DataFormat.JSON:
            return pd.read_json(full_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format for local backend: {format}")
    
    def write(self, df: pd.DataFrame, path: str, format: DataFormat, **kwargs):
        """Write data to local storage."""
        full_path = self.root_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == DataFormat.PARQUET:
            df.to_parquet(full_path, **kwargs)
        elif format == DataFormat.CSV:
            df.to_csv(full_path, **kwargs)
        elif format == DataFormat.JSON:
            df.to_json(full_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format for local backend: {format}")
    
    def list_files(self, path: str, pattern: Optional[str] = None) -> List[str]:
        """List files in path."""
        full_path = self.root_path / path
        if not full_path.exists():
            return []
        
        if pattern:
            files = list(full_path.glob(pattern))
        else:
            files = list(full_path.iterdir())
        
        return [str(f.relative_to(self.root_path)) for f in files if f.is_file()]
    
    def delete(self, path: str):
        """Delete file or directory."""
        full_path = self.root_path / path
        if full_path.is_file():
            full_path.unlink()
        elif full_path.is_dir():
            import shutil
            shutil.rmtree(full_path)
    
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        return (self.root_path / path).exists()
    
    def get_metadata(self, path: str) -> Dict[str, Any]:
        """Get file/directory metadata."""
        full_path = self.root_path / path
        if not full_path.exists():
            return {}
        
        stat = full_path.stat()
        return {
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "created": datetime.fromtimestamp(stat.st_ctime),
            "is_file": full_path.is_file(),
            "is_dir": full_path.is_dir()
        }


class S3Backend(StorageBackendInterface):
    """AWS S3 storage backend."""
    
    def __init__(self, bucket: str, region: str = "us-east-1"):
        """Initialize S3 backend."""
        import boto3
        self.s3 = boto3.client('s3', region_name=region)
        self.bucket = bucket
    
    def read(self, path: str, format: DataFormat, **kwargs) -> pd.DataFrame:
        """Read data from S3."""
        s3_path = f"s3://{self.bucket}/{path}"
        
        if format == DataFormat.PARQUET:
            return pd.read_parquet(s3_path, **kwargs)
        elif format == DataFormat.CSV:
            return pd.read_csv(s3_path, **kwargs)
        elif format == DataFormat.JSON:
            return pd.read_json(s3_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format for S3: {format}")
    
    def write(self, df: pd.DataFrame, path: str, format: DataFormat, **kwargs):
        """Write data to S3."""
        s3_path = f"s3://{self.bucket}/{path}"
        
        if format == DataFormat.PARQUET:
            df.to_parquet(s3_path, **kwargs)
        elif format == DataFormat.CSV:
            df.to_csv(s3_path, **kwargs)
        elif format == DataFormat.JSON:
            df.to_json(s3_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format for S3: {format}")
    
    def list_files(self, path: str, pattern: Optional[str] = None) -> List[str]:
        """List files in S3 path."""
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=path
        )
        
        if 'Contents' not in response:
            return []
        
        files = [obj['Key'] for obj in response['Contents']]
        
        if pattern:
            import fnmatch
            files = [f for f in files if fnmatch.fnmatch(f, pattern)]
        
        return files
    
    def delete(self, path: str):
        """Delete S3 object or prefix."""
        # List all objects with prefix
        objects = self.list_files(path)
        
        if objects:
            # Delete objects in batches
            delete_objects = [{'Key': obj} for obj in objects]
            self.s3.delete_objects(
                Bucket=self.bucket,
                Delete={'Objects': delete_objects}
            )
    
    def exists(self, path: str) -> bool:
        """Check if S3 path exists."""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=path)
            return True
        except:
            # Check if it's a prefix
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=path,
                MaxKeys=1
            )
            return 'Contents' in response
    
    def get_metadata(self, path: str) -> Dict[str, Any]:
        """Get S3 object metadata."""
        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=path)
            return {
                "size": response['ContentLength'],
                "modified": response['LastModified'],
                "etag": response['ETag'],
                "content_type": response.get('ContentType', 'unknown')
            }
        except:
            return {}


class DataLakeManager:
    """Main data lake manager."""
    
    def __init__(self, config: DataLakeConfig):
        """Initialize data lake manager."""
        self.config = config
        self._initialize_backend()
        self._initialize_spark()
        self._initialize_layers()
        self.compression_handler = CompressionHandler(config.compression)
        self.partitioning_strategy = PartitioningStrategy()
        
        logger.info(f"Data Lake initialized with {config.storage_backend.value} backend")
    
    def _initialize_backend(self):
        """Initialize storage backend."""
        if self.config.storage_backend == StorageBackend.LOCAL_FILE_SYSTEM:
            self.backend = LocalFileSystemBackend(self.config.root_path)
        elif self.config.storage_backend == StorageBackend.AWS_S3:
            # Extract bucket and prefix from root_path
            parts = self.config.root_path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            self.backend = S3Backend(bucket)
        else:
            raise ValueError(f"Unsupported backend: {self.config.storage_backend}")
    
    def _initialize_spark(self):
        """Initialize Spark session with Delta support."""
        if self.config.enable_delta:
            if not DELTA_AVAILABLE:
                raise ImportError(
                    "Delta Lake support requires pyspark and delta-spark packages. "
                    "Install with: pip install alpha-pulse[datalake] or "
                    "poetry install --extras datalake"
                )

            builder = SparkSession.builder \
                .appName("AlphaPulseDataLake") \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

            # Configure for S3 if needed
            if self.config.storage_backend == StorageBackend.AWS_S3:
                builder = builder \
                    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
                    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")

            self.spark = configure_spark_with_delta_pip(builder).getOrCreate()
        else:
            self.spark = None
    
    def _initialize_layers(self):
        """Initialize data layers."""
        self.bronze_layer = BronzeLayer(
            path=f"{self.config.root_path}/{self.config.bronze_path}",
            backend=self.backend,
            retention_days=self.config.retention_days["bronze"]
        )
        
        self.silver_layer = SilverLayer(
            path=f"{self.config.root_path}/{self.config.silver_path}",
            backend=self.backend,
            retention_days=self.config.retention_days["silver"],
            spark_session=self.spark
        )
        
        self.gold_layer = GoldLayer(
            path=f"{self.config.root_path}/{self.config.gold_path}",
            backend=self.backend,
            retention_days=self.config.retention_days["gold"],
            spark_session=self.spark
        )
    
    def ingest_raw_data(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], str],
        source: str,
        dataset_name: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Ingest raw data into bronze layer."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Convert data to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, str):
            # Assume it's a file path
            df = pd.read_csv(data)
        else:
            df = data
        
        # Add metadata columns
        df['_ingestion_timestamp'] = timestamp
        df['_source'] = source
        df['_dataset'] = dataset_name
        
        if metadata:
            df['_metadata'] = json.dumps(metadata)
        
        # Generate partition path
        partition_path = self.partitioning_strategy.generate_time_partition(
            timestamp, 
            level="day"
        )
        
        # Write to bronze layer
        path = self.bronze_layer.write_raw_data(
            df=df,
            source=source,
            dataset_name=dataset_name,
            partition_path=partition_path,
            format=self.config.default_format,
            compression=self.config.compression
        )
        
        logger.info(f"Ingested {len(df)} records to bronze layer: {path}")
        return path
    
    def process_to_silver(
        self,
        dataset_name: str,
        processing_func: Callable[[pd.DataFrame], pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> str:
        """Process data from bronze to silver layer."""
        # Read from bronze
        bronze_data = self.bronze_layer.read_dataset(
            dataset_name=dataset_name,
            start_date=start_date,
            end_date=end_date
        )
        
        if bronze_data.empty:
            logger.warning(f"No data found in bronze layer for {dataset_name}")
            return ""
        
        # Apply processing function
        processed_data = processing_func(bronze_data)
        
        # Add processing metadata
        processed_data['_processing_timestamp'] = datetime.utcnow()
        processed_data['_processing_version'] = "1.0"
        
        # Write to silver layer
        path = self.silver_layer.write_processed_data(
            df=processed_data,
            dataset_name=dataset_name,
            format=DataFormat.DELTA if self.config.enable_delta else self.config.default_format
        )
        
        logger.info(f"Processed {len(processed_data)} records to silver layer: {path}")
        return path
    
    def create_gold_dataset(
        self,
        dataset_name: str,
        aggregation_func: Callable[[pd.DataFrame], pd.DataFrame],
        source_datasets: List[str],
        business_date: Optional[datetime] = None
    ) -> str:
        """Create business-ready dataset in gold layer."""
        # Read from silver layer
        dfs = []
        for source in source_datasets:
            df = self.silver_layer.read_dataset(source)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            logger.warning(f"No data found in silver layer for {source_datasets}")
            return ""
        
        # Combine datasets
        combined_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        
        # Apply aggregation
        gold_data = aggregation_func(combined_df)
        
        # Write to gold layer
        path = self.gold_layer.write_business_dataset(
            df=gold_data,
            dataset_name=dataset_name,
            business_date=business_date
        )
        
        logger.info(f"Created gold dataset with {len(gold_data)} records: {path}")
        return path
    
    def query_data(
        self,
        query: str,
        layer: str = "silver",
        format: str = "pandas"
    ) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """Query data from specified layer."""
        if not self.spark:
            raise RuntimeError("Spark not initialized. Enable Delta Lake support.")
        
        # Register layer as temporary view
        if layer == "bronze":
            layer_path = f"{self.config.root_path}/{self.config.bronze_path}"
        elif layer == "silver":
            layer_path = f"{self.config.root_path}/{self.config.silver_path}"
        elif layer == "gold":
            layer_path = f"{self.config.root_path}/{self.config.gold_path}"
        else:
            raise ValueError(f"Invalid layer: {layer}")
        
        # Read data into Spark
        if self.config.default_format == DataFormat.DELTA:
            df = self.spark.read.format("delta").load(layer_path)
        else:
            df = self.spark.read.parquet(layer_path)
        
        df.createOrReplaceTempView(f"{layer}_view")
        
        # Execute query
        result = self.spark.sql(query)
        
        # Return in requested format
        if format == "pandas":
            return result.toPandas()
        else:
            return result
    
    def optimize_storage(self, layer: str = "all"):
        """Optimize storage through compaction and cleanup."""
        layers_to_optimize = []
        
        if layer == "all":
            layers_to_optimize = [self.bronze_layer, self.silver_layer, self.gold_layer]
        elif layer == "bronze":
            layers_to_optimize = [self.bronze_layer]
        elif layer == "silver":
            layers_to_optimize = [self.silver_layer]
        elif layer == "gold":
            layers_to_optimize = [self.gold_layer]
        
        for data_layer in layers_to_optimize:
            logger.info(f"Optimizing {data_layer.__class__.__name__}")
            
            # Compact small files
            data_layer.compact_small_files()
            
            # Clean up old data based on retention
            data_layer.cleanup_old_data()
            
            # Update statistics
            if hasattr(data_layer, 'update_statistics'):
                data_layer.update_statistics()
    
    def get_lake_statistics(self) -> Dict[str, Any]:
        """Get data lake statistics."""
        stats = {
            "bronze": self.bronze_layer.get_statistics(),
            "silver": self.silver_layer.get_statistics(),
            "gold": self.gold_layer.get_statistics(),
            "total_size": 0,
            "total_files": 0,
            "total_datasets": 0
        }
        
        # Aggregate totals
        for layer in ["bronze", "silver", "gold"]:
            layer_stats = stats[layer]
            stats["total_size"] += layer_stats.get("total_size", 0)
            stats["total_files"] += layer_stats.get("file_count", 0)
            stats["total_datasets"] += layer_stats.get("dataset_count", 0)
        
        return stats
    
    def shutdown(self):
        """Shutdown data lake manager."""
        if self.spark:
            self.spark.stop()
        logger.info("Data Lake Manager shutdown complete")