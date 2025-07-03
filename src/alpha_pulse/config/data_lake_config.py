"""
Configuration for data lake infrastructure.

Defines settings for storage, processing, and lifecycle management
of the multi-layered data lake architecture.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path


class StorageClass(Enum):
    """Storage class tiers for cost optimization."""
    STANDARD = "standard"
    INFREQUENT_ACCESS = "infrequent_access"
    GLACIER = "glacier"
    DEEP_ARCHIVE = "deep_archive"


class DataLakeEnvironment(Enum):
    """Data lake environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class StorageTierConfig:
    """Configuration for storage tiering."""
    name: str
    storage_class: StorageClass
    transition_days: int
    min_object_size_mb: float = 0.128  # 128KB
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "storage_class": self.storage_class.value,
            "transition_days": self.transition_days,
            "min_object_size_mb": self.min_object_size_mb
        }


@dataclass
class LayerConfig:
    """Configuration for a data lake layer."""
    name: str
    path: str
    retention_days: int
    format: str = "parquet"
    compression: str = "snappy"
    partition_strategy: str = "time_based"
    storage_tiers: List[StorageTierConfig] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "retention_days": self.retention_days,
            "format": self.format,
            "compression": self.compression,
            "partition_strategy": self.partition_strategy,
            "storage_tiers": [tier.to_dict() for tier in self.storage_tiers]
        }


@dataclass
class DataLakeConfig:
    """Main data lake configuration."""
    environment: DataLakeEnvironment
    storage_backend: str
    root_path: str
    bronze_config: LayerConfig
    silver_config: LayerConfig
    gold_config: LayerConfig
    enable_delta_lake: bool = True
    enable_data_catalog: bool = True
    enable_lineage_tracking: bool = True
    enable_quality_monitoring: bool = True
    
    # Performance settings
    max_concurrent_jobs: int = 10
    default_batch_size: int = 10000
    checkpoint_interval: int = 1000
    
    # Cost optimization
    enable_auto_tiering: bool = True
    enable_lifecycle_policies: bool = True
    compact_small_files: bool = True
    small_file_threshold_mb: float = 128
    
    # Monitoring
    metrics_enabled: bool = True
    alerting_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "environment": self.environment.value,
            "storage_backend": self.storage_backend,
            "root_path": self.root_path,
            "layers": {
                "bronze": self.bronze_config.to_dict(),
                "silver": self.silver_config.to_dict(),
                "gold": self.gold_config.to_dict()
            },
            "features": {
                "delta_lake": self.enable_delta_lake,
                "data_catalog": self.enable_data_catalog,
                "lineage_tracking": self.enable_lineage_tracking,
                "quality_monitoring": self.enable_quality_monitoring
            },
            "performance": {
                "max_concurrent_jobs": self.max_concurrent_jobs,
                "default_batch_size": self.default_batch_size,
                "checkpoint_interval": self.checkpoint_interval
            },
            "cost_optimization": {
                "auto_tiering": self.enable_auto_tiering,
                "lifecycle_policies": self.enable_lifecycle_policies,
                "compact_small_files": self.compact_small_files,
                "small_file_threshold_mb": self.small_file_threshold_mb
            },
            "monitoring": {
                "metrics_enabled": self.metrics_enabled,
                "alerting_enabled": self.alerting_enabled
            }
        }


class DataLakeConfigFactory:
    """Factory for creating data lake configurations."""
    
    @staticmethod
    def create_default_config(
        environment: DataLakeEnvironment = DataLakeEnvironment.DEVELOPMENT
    ) -> DataLakeConfig:
        """Create default data lake configuration."""
        # Determine root path based on environment
        if environment == DataLakeEnvironment.PRODUCTION:
            root_path = os.environ.get("DATA_LAKE_ROOT", "s3://alphapulse-data-lake-prod")
            storage_backend = "aws_s3"
        else:
            root_path = os.environ.get("DATA_LAKE_ROOT", str(Path.home() / "alphapulse_data_lake"))
            storage_backend = "local_file_system"
        
        # Bronze layer config
        bronze_config = LayerConfig(
            name="bronze",
            path="bronze",
            retention_days=2555,  # 7 years
            format="parquet",
            compression="snappy",
            partition_strategy="time_based",
            storage_tiers=[
                StorageTierConfig(
                    name="hot",
                    storage_class=StorageClass.STANDARD,
                    transition_days=0
                ),
                StorageTierConfig(
                    name="warm",
                    storage_class=StorageClass.INFREQUENT_ACCESS,
                    transition_days=30
                ),
                StorageTierConfig(
                    name="cold",
                    storage_class=StorageClass.GLACIER,
                    transition_days=90
                ),
                StorageTierConfig(
                    name="archive",
                    storage_class=StorageClass.DEEP_ARCHIVE,
                    transition_days=365
                )
            ] if environment == DataLakeEnvironment.PRODUCTION else []
        )
        
        # Silver layer config
        silver_config = LayerConfig(
            name="silver",
            path="silver",
            retention_days=1825,  # 5 years
            format="delta" if environment != DataLakeEnvironment.DEVELOPMENT else "parquet",
            compression="zstd",
            partition_strategy="composite",
            storage_tiers=[
                StorageTierConfig(
                    name="hot",
                    storage_class=StorageClass.STANDARD,
                    transition_days=0
                ),
                StorageTierConfig(
                    name="warm",
                    storage_class=StorageClass.INFREQUENT_ACCESS,
                    transition_days=60
                ),
                StorageTierConfig(
                    name="cold",
                    storage_class=StorageClass.GLACIER,
                    transition_days=180
                )
            ] if environment == DataLakeEnvironment.PRODUCTION else []
        )
        
        # Gold layer config
        gold_config = LayerConfig(
            name="gold",
            path="gold",
            retention_days=-1,  # Permanent
            format="parquet",
            compression="snappy",
            partition_strategy="business_domain",
            storage_tiers=[
                StorageTierConfig(
                    name="active",
                    storage_class=StorageClass.STANDARD,
                    transition_days=0
                ),
                StorageTierConfig(
                    name="reference",
                    storage_class=StorageClass.INFREQUENT_ACCESS,
                    transition_days=90
                )
            ] if environment == DataLakeEnvironment.PRODUCTION else []
        )
        
        return DataLakeConfig(
            environment=environment,
            storage_backend=storage_backend,
            root_path=root_path,
            bronze_config=bronze_config,
            silver_config=silver_config,
            gold_config=gold_config,
            enable_delta_lake=environment != DataLakeEnvironment.DEVELOPMENT,
            enable_data_catalog=True,
            enable_lineage_tracking=True,
            enable_quality_monitoring=True,
            max_concurrent_jobs=20 if environment == DataLakeEnvironment.PRODUCTION else 5,
            default_batch_size=50000 if environment == DataLakeEnvironment.PRODUCTION else 10000,
            enable_auto_tiering=environment == DataLakeEnvironment.PRODUCTION,
            enable_lifecycle_policies=environment == DataLakeEnvironment.PRODUCTION
        )
    
    @staticmethod
    def create_from_dict(config_dict: Dict[str, Any]) -> DataLakeConfig:
        """Create configuration from dictionary."""
        # Parse layer configs
        bronze_dict = config_dict["layers"]["bronze"]
        bronze_config = LayerConfig(
            name=bronze_dict["name"],
            path=bronze_dict["path"],
            retention_days=bronze_dict["retention_days"],
            format=bronze_dict.get("format", "parquet"),
            compression=bronze_dict.get("compression", "snappy"),
            partition_strategy=bronze_dict.get("partition_strategy", "time_based"),
            storage_tiers=[
                StorageTierConfig(
                    name=tier["name"],
                    storage_class=StorageClass(tier["storage_class"]),
                    transition_days=tier["transition_days"],
                    min_object_size_mb=tier.get("min_object_size_mb", 0.128)
                )
                for tier in bronze_dict.get("storage_tiers", [])
            ]
        )
        
        silver_dict = config_dict["layers"]["silver"]
        silver_config = LayerConfig(
            name=silver_dict["name"],
            path=silver_dict["path"],
            retention_days=silver_dict["retention_days"],
            format=silver_dict.get("format", "delta"),
            compression=silver_dict.get("compression", "zstd"),
            partition_strategy=silver_dict.get("partition_strategy", "composite"),
            storage_tiers=[
                StorageTierConfig(
                    name=tier["name"],
                    storage_class=StorageClass(tier["storage_class"]),
                    transition_days=tier["transition_days"],
                    min_object_size_mb=tier.get("min_object_size_mb", 0.128)
                )
                for tier in silver_dict.get("storage_tiers", [])
            ]
        )
        
        gold_dict = config_dict["layers"]["gold"]
        gold_config = LayerConfig(
            name=gold_dict["name"],
            path=gold_dict["path"],
            retention_days=gold_dict["retention_days"],
            format=gold_dict.get("format", "parquet"),
            compression=gold_dict.get("compression", "snappy"),
            partition_strategy=gold_dict.get("partition_strategy", "business_domain"),
            storage_tiers=[
                StorageTierConfig(
                    name=tier["name"],
                    storage_class=StorageClass(tier["storage_class"]),
                    transition_days=tier["transition_days"],
                    min_object_size_mb=tier.get("min_object_size_mb", 0.128)
                )
                for tier in gold_dict.get("storage_tiers", [])
            ]
        )
        
        return DataLakeConfig(
            environment=DataLakeEnvironment(config_dict["environment"]),
            storage_backend=config_dict["storage_backend"],
            root_path=config_dict["root_path"],
            bronze_config=bronze_config,
            silver_config=silver_config,
            gold_config=gold_config,
            enable_delta_lake=config_dict["features"]["delta_lake"],
            enable_data_catalog=config_dict["features"]["data_catalog"],
            enable_lineage_tracking=config_dict["features"]["lineage_tracking"],
            enable_quality_monitoring=config_dict["features"]["quality_monitoring"],
            max_concurrent_jobs=config_dict["performance"]["max_concurrent_jobs"],
            default_batch_size=config_dict["performance"]["default_batch_size"],
            checkpoint_interval=config_dict["performance"]["checkpoint_interval"],
            enable_auto_tiering=config_dict["cost_optimization"]["auto_tiering"],
            enable_lifecycle_policies=config_dict["cost_optimization"]["lifecycle_policies"],
            compact_small_files=config_dict["cost_optimization"]["compact_small_files"],
            small_file_threshold_mb=config_dict["cost_optimization"]["small_file_threshold_mb"],
            metrics_enabled=config_dict["monitoring"]["metrics_enabled"],
            alerting_enabled=config_dict["monitoring"]["alerting_enabled"]
        )


# Pre-configured environments
DEVELOPMENT_CONFIG = DataLakeConfigFactory.create_default_config(
    DataLakeEnvironment.DEVELOPMENT
)

STAGING_CONFIG = DataLakeConfigFactory.create_default_config(
    DataLakeEnvironment.STAGING
)

PRODUCTION_CONFIG = DataLakeConfigFactory.create_default_config(
    DataLakeEnvironment.PRODUCTION
)


def get_data_lake_config(environment: Optional[str] = None) -> DataLakeConfig:
    """Get data lake configuration for specified environment."""
    if environment is None:
        environment = os.environ.get("DATA_LAKE_ENV", "development")
    
    environment = environment.lower()
    
    if environment == "production":
        return PRODUCTION_CONFIG
    elif environment == "staging":
        return STAGING_CONFIG
    else:
        return DEVELOPMENT_CONFIG