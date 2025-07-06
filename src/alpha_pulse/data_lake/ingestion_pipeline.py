"""
Data ingestion pipelines for the data lake.

Handles batch and streaming ingestion with validation,
transformation, and monitoring.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import json
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from aiokafka import AIOKafkaConsumer

from alpha_pulse.data_lake.lake_manager import DataLakeManager, DataFormat
from alpha_pulse.data_lake.partitioning_strategy import PartitionConfig, PartitionScheme
from alpha_pulse.data.quality.data_validator import DataValidator
from alpha_pulse.data.quality.quality_metrics import QualityMetricsService
from alpha_pulse.models.streaming_message import StreamingMessage, MarketDataMessage
from alpha_pulse.monitoring.metrics_collector import MetricsCollector


logger = logging.getLogger(__name__)


class IngestionMode(Enum):
    """Ingestion modes."""
    BATCH = "batch"
    STREAMING = "streaming"
    HYBRID = "hybrid"
    INCREMENTAL = "incremental"


class IngestionStatus(Enum):
    """Ingestion job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class IngestionConfig:
    """Configuration for data ingestion."""
    source_name: str
    dataset_name: str
    mode: IngestionMode
    format: DataFormat = DataFormat.PARQUET
    partition_config: Optional[PartitionConfig] = None
    validation_enabled: bool = True
    quality_threshold: float = 0.8
    batch_size: int = 10000
    max_retries: int = 3
    retry_delay_seconds: int = 60
    checkpoint_interval: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionJob:
    """Represents an ingestion job."""
    job_id: str
    config: IngestionConfig
    status: IngestionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    records_processed: int = 0
    records_failed: int = 0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint: Optional[Dict[str, Any]] = None


class IngestionPipeline(ABC):
    """Abstract base class for ingestion pipelines."""
    
    def __init__(
        self,
        lake_manager: DataLakeManager,
        config: IngestionConfig
    ):
        """Initialize ingestion pipeline."""
        self.lake_manager = lake_manager
        self.config = config
        self.validator = DataValidator() if config.validation_enabled else None
        self.quality_service = QualityMetricsService() if config.validation_enabled else None
        self.metrics_collector = MetricsCollector()
        self.current_job: Optional[IngestionJob] = None
    
    @abstractmethod
    async def ingest(self) -> IngestionJob:
        """Execute ingestion."""
        pass
    
    @abstractmethod
    async def validate_source(self) -> bool:
        """Validate data source availability."""
        pass
    
    def _create_job(self) -> IngestionJob:
        """Create new ingestion job."""
        return IngestionJob(
            job_id=f"{self.config.source_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            config=self.config,
            status=IngestionStatus.PENDING,
            started_at=datetime.utcnow()
        )
    
    async def _validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """Validate data and return cleaned data with quality score."""
        if not self.validator:
            return df, 1.0
        
        # Run validation
        validation_result = self.validator.validate_dataframe(
            df=df,
            dataset_name=self.config.dataset_name,
            rules=self._get_validation_rules()
        )
        
        # Calculate quality score
        quality_metrics = self.quality_service.calculate_metrics(
            df=df,
            dataset_name=self.config.dataset_name
        )
        
        quality_score = quality_metrics.overall_score
        
        # Clean data based on validation results
        if validation_result.is_valid and quality_score >= self.config.quality_threshold:
            return df, quality_score
        else:
            # Apply data cleaning
            cleaned_df = self._clean_data(df, validation_result)
            return cleaned_df, quality_score
    
    def _get_validation_rules(self) -> List[Dict[str, Any]]:
        """Get validation rules for dataset."""
        # Default rules - can be overridden
        return [
            {
                "name": "no_future_dates",
                "type": "custom",
                "function": lambda df: ~(df.select_dtypes(include=['datetime']).apply(
                    lambda x: x > datetime.utcnow()
                ).any().any())
            },
            {
                "name": "no_negative_values",
                "type": "range",
                "columns": ["price", "volume", "quantity"],
                "min_value": 0
            }
        ]
    
    def _clean_data(self, df: pd.DataFrame, validation_result: Any) -> pd.DataFrame:
        """Clean data based on validation results."""
        cleaned_df = df.copy()
        
        # Remove rows with critical errors
        if hasattr(validation_result, 'invalid_rows'):
            cleaned_df = cleaned_df.drop(validation_result.invalid_rows)
        
        # Handle missing values
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(0)
        
        # Handle string columns
        string_columns = cleaned_df.select_dtypes(include=['object']).columns
        cleaned_df[string_columns] = cleaned_df[string_columns].fillna('')
        
        return cleaned_df
    
    def _update_metrics(self, job: IngestionJob):
        """Update ingestion metrics."""
        duration = (datetime.utcnow() - job.started_at).total_seconds()
        
        job.metrics.update({
            "duration_seconds": duration,
            "records_per_second": job.records_processed / duration if duration > 0 else 0,
            "success_rate": job.records_processed / (job.records_processed + job.records_failed) 
                           if (job.records_processed + job.records_failed) > 0 else 0
        })
        
        # Send to metrics collector
        self.metrics_collector.record_metric(
            "data_lake.ingestion.completed",
            1,
            tags={
                "source": self.config.source_name,
                "dataset": self.config.dataset_name,
                "status": job.status.value
            }
        )


class BatchIngestionPipeline(IngestionPipeline):
    """Batch ingestion pipeline for files and databases."""
    
    def __init__(
        self,
        lake_manager: DataLakeManager,
        config: IngestionConfig,
        source_path: Optional[str] = None,
        source_query: Optional[str] = None,
        connection_string: Optional[str] = None
    ):
        """Initialize batch ingestion pipeline."""
        super().__init__(lake_manager, config)
        self.source_path = source_path
        self.source_query = source_query
        self.connection_string = connection_string
    
    async def validate_source(self) -> bool:
        """Validate batch data source."""
        if self.source_path:
            # Check if file exists
            return Path(self.source_path).exists()
        elif self.source_query and self.connection_string:
            # Test database connection
            try:
                import sqlalchemy
                engine = sqlalchemy.create_engine(self.connection_string)
                with engine.connect() as conn:
                    conn.execute("SELECT 1")
                return True
            except Exception as e:
                logger.error(f"Failed to validate database source: {e}")
                return False
        else:
            return False
    
    async def ingest(self) -> IngestionJob:
        """Execute batch ingestion."""
        job = self._create_job()
        job.status = IngestionStatus.RUNNING
        self.current_job = job
        
        try:
            # Validate source
            if not await self.validate_source():
                raise ValueError("Invalid data source")
            
            # Load data
            df = await self._load_data()
            
            if df.empty:
                raise ValueError("No data to ingest")
            
            # Process in batches
            total_rows = len(df)
            for start_idx in range(0, total_rows, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx]
                
                # Validate and clean batch
                cleaned_batch, quality_score = await self._validate_data(batch_df)
                
                if quality_score < self.config.quality_threshold:
                    logger.warning(
                        f"Batch quality score {quality_score} below threshold "
                        f"{self.config.quality_threshold}"
                    )
                
                # Ingest batch
                try:
                    path = self.lake_manager.ingest_raw_data(
                        data=cleaned_batch,
                        source=self.config.source_name,
                        dataset_name=self.config.dataset_name,
                        metadata={
                            "batch_number": start_idx // self.config.batch_size,
                            "quality_score": quality_score,
                            "original_rows": len(batch_df),
                            "cleaned_rows": len(cleaned_batch)
                        }
                    )
                    
                    job.records_processed += len(cleaned_batch)
                    
                    # Checkpoint
                    if job.records_processed % self.config.checkpoint_interval == 0:
                        job.checkpoint = {
                            "last_index": end_idx,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    
                except Exception as e:
                    logger.error(f"Failed to ingest batch: {e}")
                    job.records_failed += len(batch_df)
            
            # Mark job as completed
            job.status = IngestionStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Batch ingestion failed: {e}")
            job.status = IngestionStatus.FAILED
            job.error_message = str(e)
        
        finally:
            self._update_metrics(job)
        
        return job
    
    async def _load_data(self) -> pd.DataFrame:
        """Load data from source."""
        if self.source_path:
            # Load from file
            file_path = Path(self.source_path)
            
            if file_path.suffix == '.csv':
                return pd.read_csv(file_path)
            elif file_path.suffix == '.parquet':
                return pd.read_parquet(file_path)
            elif file_path.suffix == '.json':
                return pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        elif self.source_query and self.connection_string:
            # Load from database
            import sqlalchemy
            engine = sqlalchemy.create_engine(self.connection_string)
            return pd.read_sql_query(self.source_query, engine)
        
        else:
            raise ValueError("No valid data source specified")


class StreamingIngestionPipeline(IngestionPipeline):
    """Streaming ingestion pipeline for real-time data."""
    
    def __init__(
        self,
        lake_manager: DataLakeManager,
        config: IngestionConfig,
        kafka_config: Dict[str, Any],
        topic: str,
        message_handler: Optional[Callable[[StreamingMessage], pd.DataFrame]] = None
    ):
        """Initialize streaming ingestion pipeline."""
        super().__init__(lake_manager, config)
        self.kafka_config = kafka_config
        self.topic = topic
        self.message_handler = message_handler or self._default_message_handler
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.buffer: List[pd.DataFrame] = []
        self.last_flush_time = datetime.utcnow()
    
    async def validate_source(self) -> bool:
        """Validate streaming source."""
        try:
            # Test Kafka connection
            consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=self.kafka_config['bootstrap_servers'],
                group_id=f"{self.config.source_name}_validator",
                auto_offset_reset='latest'
            )
            await consumer.start()
            await consumer.stop()
            return True
        except Exception as e:
            logger.error(f"Failed to validate Kafka source: {e}")
            return False
    
    async def ingest(self) -> IngestionJob:
        """Execute streaming ingestion."""
        job = self._create_job()
        job.status = IngestionStatus.RUNNING
        self.current_job = job
        
        try:
            # Validate source
            if not await self.validate_source():
                raise ValueError("Invalid streaming source")
            
            # Create consumer
            self.consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=self.kafka_config['bootstrap_servers'],
                group_id=self.kafka_config.get('group_id', f"{self.config.source_name}_consumer"),
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest'
            )
            
            await self.consumer.start()
            
            # Consume messages
            async for msg in self.consumer:
                try:
                    # Convert message to DataFrame
                    df = self.message_handler(msg.value)
                    
                    if df is not None and not df.empty:
                        self.buffer.append(df)
                        job.records_processed += len(df)
                    
                    # Check if we should flush buffer
                    if self._should_flush():
                        await self._flush_buffer(job)
                    
                except Exception as e:
                    logger.error(f"Failed to process message: {e}")
                    job.records_failed += 1
                
                # Check if job should complete
                if job.records_processed >= self.config.metadata.get('max_records', float('inf')):
                    break
            
            # Final flush
            if self.buffer:
                await self._flush_buffer(job)
            
            job.status = IngestionStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Streaming ingestion failed: {e}")
            job.status = IngestionStatus.FAILED
            job.error_message = str(e)
        
        finally:
            if self.consumer:
                await self.consumer.stop()
            self._update_metrics(job)
        
        return job
    
    def _default_message_handler(self, message: Dict[str, Any]) -> pd.DataFrame:
        """Default message handler."""
        # Convert single message to DataFrame
        return pd.DataFrame([message])
    
    def _should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        # Flush based on size or time
        buffer_size = sum(len(df) for df in self.buffer)
        time_since_flush = (datetime.utcnow() - self.last_flush_time).total_seconds()
        
        return (
            buffer_size >= self.config.batch_size or
            time_since_flush >= 60  # Flush every minute
        )
    
    async def _flush_buffer(self, job: IngestionJob):
        """Flush buffer to data lake."""
        if not self.buffer:
            return
        
        # Combine buffer
        combined_df = pd.concat(self.buffer, ignore_index=True)
        self.buffer.clear()
        
        # Validate and clean
        cleaned_df, quality_score = await self._validate_data(combined_df)
        
        # Ingest to lake
        try:
            path = self.lake_manager.ingest_raw_data(
                data=cleaned_df,
                source=self.config.source_name,
                dataset_name=self.config.dataset_name,
                metadata={
                    "stream_batch": True,
                    "quality_score": quality_score,
                    "flush_time": datetime.utcnow().isoformat()
                }
            )
            
            self.last_flush_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to flush buffer: {e}")
            job.records_failed += len(combined_df)


class IncrementalIngestionPipeline(BatchIngestionPipeline):
    """Incremental ingestion pipeline for change data capture."""
    
    def __init__(
        self,
        lake_manager: DataLakeManager,
        config: IngestionConfig,
        watermark_column: str,
        state_store: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize incremental ingestion pipeline."""
        super().__init__(lake_manager, config, **kwargs)
        self.watermark_column = watermark_column
        self.state_store = state_store or {}
    
    async def ingest(self) -> IngestionJob:
        """Execute incremental ingestion."""
        # Get last watermark
        last_watermark = self.state_store.get(
            f"{self.config.source_name}_{self.config.dataset_name}_watermark"
        )
        
        # Modify query to include watermark filter
        if last_watermark and self.source_query:
            self.source_query = f"""
            {self.source_query}
            WHERE {self.watermark_column} > '{last_watermark}'
            ORDER BY {self.watermark_column}
            """
        
        # Execute batch ingestion
        job = await super().ingest()
        
        # Update watermark if successful
        if job.status == IngestionStatus.COMPLETED and job.records_processed > 0:
            # Get new watermark from ingested data
            df = await self._load_data()
            if not df.empty and self.watermark_column in df.columns:
                new_watermark = df[self.watermark_column].max()
                self.state_store[
                    f"{self.config.source_name}_{self.config.dataset_name}_watermark"
                ] = new_watermark
                
                logger.info(f"Updated watermark to: {new_watermark}")
        
        return job


class IngestionOrchestrator:
    """Orchestrates multiple ingestion pipelines."""
    
    def __init__(self, lake_manager: DataLakeManager):
        """Initialize orchestrator."""
        self.lake_manager = lake_manager
        self.pipelines: Dict[str, IngestionPipeline] = {}
        self.jobs: List[IngestionJob] = []
        self.scheduler_task: Optional[asyncio.Task] = None
    
    def register_pipeline(self, name: str, pipeline: IngestionPipeline):
        """Register an ingestion pipeline."""
        self.pipelines[name] = pipeline
        logger.info(f"Registered pipeline: {name}")
    
    async def run_pipeline(self, name: str) -> IngestionJob:
        """Run a specific pipeline."""
        if name not in self.pipelines:
            raise ValueError(f"Pipeline {name} not found")
        
        pipeline = self.pipelines[name]
        job = await pipeline.ingest()
        self.jobs.append(job)
        
        return job
    
    async def run_all_pipelines(self, parallel: bool = True) -> List[IngestionJob]:
        """Run all registered pipelines."""
        if parallel:
            # Run pipelines in parallel
            tasks = [
                self.run_pipeline(name)
                for name in self.pipelines
            ]
            jobs = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            return [job for job in jobs if isinstance(job, IngestionJob)]
        else:
            # Run pipelines sequentially
            jobs = []
            for name in self.pipelines:
                try:
                    job = await self.run_pipeline(name)
                    jobs.append(job)
                except Exception as e:
                    logger.error(f"Failed to run pipeline {name}: {e}")
            
            return jobs
    
    def get_job_status(self, job_id: str) -> Optional[IngestionJob]:
        """Get status of a specific job."""
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        return None
    
    def get_pipeline_stats(self, pipeline_name: str) -> Dict[str, Any]:
        """Get statistics for a pipeline."""
        pipeline_jobs = [
            job for job in self.jobs
            if job.config.source_name == pipeline_name
        ]
        
        if not pipeline_jobs:
            return {}
        
        successful_jobs = [
            job for job in pipeline_jobs
            if job.status == IngestionStatus.COMPLETED
        ]
        
        total_records = sum(job.records_processed for job in pipeline_jobs)
        total_failures = sum(job.records_failed for job in pipeline_jobs)
        
        return {
            "total_jobs": len(pipeline_jobs),
            "successful_jobs": len(successful_jobs),
            "success_rate": len(successful_jobs) / len(pipeline_jobs) if pipeline_jobs else 0,
            "total_records_processed": total_records,
            "total_records_failed": total_failures,
            "average_duration_seconds": sum(
                job.metrics.get("duration_seconds", 0) for job in pipeline_jobs
            ) / len(pipeline_jobs) if pipeline_jobs else 0
        }
    
    async def start_scheduler(self, schedule: Dict[str, str]):
        """Start scheduled pipeline execution."""
        async def scheduler_loop():
            while True:
                try:
                    # Check schedules
                    for pipeline_name, cron_expr in schedule.items():
                        if self._should_run(pipeline_name, cron_expr):
                            await self.run_pipeline(pipeline_name)
                    
                    # Sleep for a minute
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
        
        self.scheduler_task = asyncio.create_task(scheduler_loop())
    
    def _should_run(self, pipeline_name: str, cron_expr: str) -> bool:
        """Check if pipeline should run based on schedule."""
        # Simplified cron check (in practice, use a proper cron library)
        # For now, just check if it's time for hourly/daily runs
        current_time = datetime.utcnow()
        
        if cron_expr == "hourly":
            # Run at the start of each hour
            return current_time.minute == 0
        elif cron_expr == "daily":
            # Run at midnight
            return current_time.hour == 0 and current_time.minute == 0
        
        return False
    
    async def stop_scheduler(self):
        """Stop scheduled execution."""
        if self.scheduler_task:
            self.scheduler_task.cancel()
            await self.scheduler_task