"""
Common types and enums for the data lake module.

This module contains shared types used across data lake components
to avoid circular imports between lake_manager and storage_layers.
"""

from enum import Enum


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
