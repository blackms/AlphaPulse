"""
Tests for data lake manager functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from alpha_pulse.data_lake import (
    DataLakeManager,
    DataLakeConfig,
    StorageBackend,
    DataFormat,
    DataLakeConfigFactory,
    DataLakeEnvironment
)


class TestDataLakeManager:
    """Test data lake manager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return DataLakeConfig(
            storage_backend=StorageBackend.LOCAL_FILE_SYSTEM,
            root_path=temp_dir,
            bronze_path="bronze",
            silver_path="silver",
            gold_path="gold",
            enable_delta=False  # Disable for tests
        )
    
    @pytest.fixture
    def lake_manager(self, config):
        """Create lake manager instance."""
        return DataLakeManager(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        return pd.DataFrame({
            'timestamp': dates,
            'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], 100),
            'price': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })
    
    def test_initialization(self, lake_manager, temp_dir):
        """Test lake manager initialization."""
        assert lake_manager.config.storage_backend == StorageBackend.LOCAL_FILE_SYSTEM
        assert lake_manager.config.root_path == temp_dir
        
        # Check layer paths exist
        assert Path(temp_dir, "bronze").exists()
        assert Path(temp_dir, "silver").exists()
        assert Path(temp_dir, "gold").exists()
    
    def test_ingest_raw_data(self, lake_manager, sample_data):
        """Test raw data ingestion to bronze layer."""
        path = lake_manager.ingest_raw_data(
            data=sample_data,
            source="test_source",
            dataset_name="test_dataset",
            metadata={"test": "metadata"}
        )
        
        assert path
        assert "bronze" in path
        assert "test_source" in path
        assert "test_dataset" in path
        
        # Verify file exists
        full_path = Path(lake_manager.config.root_path) / path
        assert full_path.exists()
    
    def test_process_to_silver(self, lake_manager, sample_data):
        """Test processing data to silver layer."""
        # First ingest to bronze
        lake_manager.ingest_raw_data(
            data=sample_data,
            source="test_source",
            dataset_name="test_dataset"
        )
        
        # Define processing function
        def process_func(df):
            df['price_change'] = df['price'].diff()
            return df
        
        # Process to silver
        path = lake_manager.process_to_silver(
            dataset_name="test_dataset",
            processing_func=process_func
        )
        
        assert path
        assert "silver" in path
        assert "test_dataset" in path
    
    def test_create_gold_dataset(self, lake_manager, sample_data):
        """Test creating business dataset in gold layer."""
        # Setup bronze and silver data
        lake_manager.ingest_raw_data(
            data=sample_data,
            source="test_source",
            dataset_name="test_dataset"
        )
        
        lake_manager.process_to_silver(
            dataset_name="test_dataset",
            processing_func=lambda df: df
        )
        
        # Create gold dataset
        def aggregate_func(df):
            return df.groupby('symbol').agg({
                'price': ['mean', 'std'],
                'volume': 'sum'
            }).reset_index()
        
        path = lake_manager.create_gold_dataset(
            dataset_name="symbol_summary",
            aggregation_func=aggregate_func,
            source_datasets=["test_dataset"]
        )
        
        assert path
        assert "gold" in path
        assert "symbol_summary" in path
    
    def test_get_lake_statistics(self, lake_manager, sample_data):
        """Test getting lake statistics."""
        # Add some data
        lake_manager.ingest_raw_data(
            data=sample_data,
            source="test_source",
            dataset_name="test_dataset"
        )
        
        stats = lake_manager.get_lake_statistics()
        
        assert "bronze" in stats
        assert "silver" in stats
        assert "gold" in stats
        assert stats["total_files"] > 0
        assert stats["total_size"] > 0
    
    def test_optimize_storage(self, lake_manager, sample_data):
        """Test storage optimization."""
        # Create multiple small files
        for i in range(5):
            small_data = sample_data.iloc[i*20:(i+1)*20]
            lake_manager.ingest_raw_data(
                data=small_data,
                source="test_source",
                dataset_name="test_dataset",
                timestamp=datetime.utcnow() + timedelta(hours=i)
            )
        
        # Get stats before optimization
        stats_before = lake_manager.get_lake_statistics()
        
        # Optimize storage
        lake_manager.optimize_storage(layer="bronze")
        
        # Files might be compacted (depending on implementation)
        stats_after = lake_manager.get_lake_statistics()
        assert stats_after["total_files"] <= stats_before["total_files"]


class TestDataLakeConfig:
    """Test data lake configuration."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = DataLakeConfigFactory.create_default_config(
            DataLakeEnvironment.DEVELOPMENT
        )
        
        assert config.environment == DataLakeEnvironment.DEVELOPMENT
        assert config.storage_backend == "local_file_system"
        assert config.bronze_config.retention_days == 2555
        assert config.silver_config.retention_days == 1825
        assert config.gold_config.retention_days == -1
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = DataLakeConfigFactory.create_default_config()
        config_dict = config.to_dict()
        
        assert config_dict["environment"] == "development"
        assert "layers" in config_dict
        assert "bronze" in config_dict["layers"]
        assert "features" in config_dict
        assert config_dict["features"]["data_catalog"] is True
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "environment": "production",
            "storage_backend": "aws_s3",
            "root_path": "s3://test-bucket",
            "layers": {
                "bronze": {
                    "name": "bronze",
                    "path": "bronze",
                    "retention_days": 2555,
                    "format": "parquet",
                    "compression": "snappy",
                    "partition_strategy": "time_based",
                    "storage_tiers": []
                },
                "silver": {
                    "name": "silver",
                    "path": "silver",
                    "retention_days": 1825,
                    "format": "delta",
                    "compression": "zstd",
                    "partition_strategy": "composite",
                    "storage_tiers": []
                },
                "gold": {
                    "name": "gold",
                    "path": "gold",
                    "retention_days": -1,
                    "format": "parquet",
                    "compression": "snappy",
                    "partition_strategy": "business_domain",
                    "storage_tiers": []
                }
            },
            "features": {
                "delta_lake": True,
                "data_catalog": True,
                "lineage_tracking": True,
                "quality_monitoring": True
            },
            "performance": {
                "max_concurrent_jobs": 20,
                "default_batch_size": 50000,
                "checkpoint_interval": 1000
            },
            "cost_optimization": {
                "auto_tiering": True,
                "lifecycle_policies": True,
                "compact_small_files": True,
                "small_file_threshold_mb": 128
            },
            "monitoring": {
                "metrics_enabled": True,
                "alerting_enabled": True
            }
        }
        
        config = DataLakeConfigFactory.create_from_dict(config_dict)
        
        assert config.environment == DataLakeEnvironment.PRODUCTION
        assert config.storage_backend == "aws_s3"
        assert config.root_path == "s3://test-bucket"
        assert config.enable_delta_lake is True