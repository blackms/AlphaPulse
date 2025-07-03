"""
Tests for data lake ingestion pipelines.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
from pathlib import Path

from alpha_pulse.data_lake import (
    BatchIngestionPipeline,
    IngestionConfig,
    IngestionMode,
    IngestionStatus,
    DataLakeManager,
    DataLakeConfig,
    StorageBackend,
    PartitionConfig,
    PartitionScheme
)


class TestBatchIngestionPipeline:
    """Test batch ingestion pipeline."""
    
    @pytest.fixture
    def lake_manager(self):
        """Create test lake manager."""
        temp_dir = tempfile.mkdtemp()
        config = DataLakeConfig(
            storage_backend=StorageBackend.LOCAL_FILE_SYSTEM,
            root_path=temp_dir,
            enable_delta=False
        )
        yield DataLakeManager(config)
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create sample CSV file."""
        # Generate sample data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'symbol': np.random.choice(['AAPL', 'GOOGL'], 100),
            'price': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        Path(temp_file.name).unlink()
    
    @pytest.fixture
    def ingestion_config(self):
        """Create test ingestion configuration."""
        return IngestionConfig(
            source_name="test_source",
            dataset_name="test_dataset",
            mode=IngestionMode.BATCH,
            validation_enabled=True,
            batch_size=50,
            partition_config=PartitionConfig(
                scheme=PartitionScheme.TIME_BASED,
                columns=['timestamp'],
                time_granularity='day'
            )
        )
    
    @pytest.mark.asyncio
    async def test_batch_ingestion_from_csv(
        self, 
        lake_manager, 
        sample_csv_file, 
        ingestion_config
    ):
        """Test batch ingestion from CSV file."""
        pipeline = BatchIngestionPipeline(
            lake_manager=lake_manager,
            config=ingestion_config,
            source_path=sample_csv_file
        )
        
        # Validate source
        is_valid = await pipeline.validate_source()
        assert is_valid is True
        
        # Execute ingestion
        job = await pipeline.ingest()
        
        assert job.status == IngestionStatus.COMPLETED
        assert job.records_processed == 100
        assert job.records_failed == 0
        assert job.metrics['duration_seconds'] > 0
        assert job.metrics['success_rate'] == 1.0
    
    @pytest.mark.asyncio
    async def test_batch_ingestion_with_validation(
        self,
        lake_manager,
        ingestion_config
    ):
        """Test batch ingestion with data validation."""
        # Create data with some invalid records
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
            'symbol': ['AAPL'] * 10,
            'price': [100, 110, -10, 120, 130, 140, 150, 160, 170, 180],  # Negative price
            'volume': [1000] * 10
        })
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        pipeline = BatchIngestionPipeline(
            lake_manager=lake_manager,
            config=ingestion_config,
            source_path=temp_file.name
        )
        
        job = await pipeline.ingest()
        
        # Should still complete but with cleaned data
        assert job.status == IngestionStatus.COMPLETED
        assert job.records_processed <= 10  # Some records might be filtered
        
        # Cleanup
        Path(temp_file.name).unlink()
    
    @pytest.mark.asyncio
    async def test_batch_ingestion_failure(
        self,
        lake_manager,
        ingestion_config
    ):
        """Test batch ingestion failure handling."""
        pipeline = BatchIngestionPipeline(
            lake_manager=lake_manager,
            config=ingestion_config,
            source_path="/non/existent/file.csv"
        )
        
        # Should fail validation
        is_valid = await pipeline.validate_source()
        assert is_valid is False
        
        # Ingestion should fail
        job = await pipeline.ingest()
        assert job.status == IngestionStatus.FAILED
        assert job.error_message is not None


class TestIngestionConfig:
    """Test ingestion configuration."""
    
    def test_config_creation(self):
        """Test creating ingestion configuration."""
        config = IngestionConfig(
            source_name="test_source",
            dataset_name="test_dataset",
            mode=IngestionMode.BATCH,
            batch_size=10000,
            max_retries=5
        )
        
        assert config.source_name == "test_source"
        assert config.dataset_name == "test_dataset"
        assert config.mode == IngestionMode.BATCH
        assert config.batch_size == 10000
        assert config.max_retries == 5
        assert config.validation_enabled is True  # Default
    
    def test_partition_config(self):
        """Test partition configuration."""
        partition_config = PartitionConfig(
            scheme=PartitionScheme.COMPOSITE,
            columns=['timestamp', 'symbol'],
            time_granularity='hour',
            hash_buckets=10
        )
        
        # Should validate successfully
        partition_config.validate()
        
        # Test invalid config
        invalid_config = PartitionConfig(
            scheme=PartitionScheme.HASH_BASED,
            columns=['symbol']
            # Missing hash_buckets
        )
        
        with pytest.raises(ValueError):
            invalid_config.validate()