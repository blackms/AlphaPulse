"""
Storage layers for the data lake architecture.

Implements Bronze, Silver, and Gold layers with specific
processing and storage strategies for each.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from delta.tables import DeltaTable

from alpha_pulse.data_lake.lake_manager import DataFormat


logger = logging.getLogger(__name__)


class DataLayer(ABC):
    """Abstract base class for data layers."""
    
    def __init__(
        self,
        path: str,
        backend: Any,
        retention_days: int = -1,
        spark_session: Optional[SparkSession] = None
    ):
        """Initialize data layer."""
        self.path = path
        self.backend = backend
        self.retention_days = retention_days
        self.spark = spark_session
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure layer directory exists."""
        if not self.backend.exists(self.path):
            # Create directory structure
            if hasattr(self.backend, 'root_path'):
                # Local filesystem
                Path(self.backend.root_path / self.path).mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get layer statistics."""
        pass
    
    @abstractmethod
    def cleanup_old_data(self):
        """Clean up data older than retention period."""
        pass
    
    def compact_small_files(self, threshold_mb: float = 128):
        """Compact small files into larger ones."""
        logger.info(f"Compacting small files in {self.path}")
        
        # List all parquet files
        files = self.backend.list_files(self.path, pattern="*.parquet")
        
        # Group by directory
        file_groups = {}
        for file in files:
            dir_path = str(Path(file).parent)
            if dir_path not in file_groups:
                file_groups[dir_path] = []
            file_groups[dir_path].append(file)
        
        # Process each directory
        for dir_path, file_list in file_groups.items():
            # Check file sizes
            small_files = []
            total_size = 0
            
            for file in file_list:
                metadata = self.backend.get_metadata(file)
                size_mb = metadata.get('size', 0) / (1024 * 1024)
                
                if size_mb < threshold_mb:
                    small_files.append(file)
                    total_size += size_mb
            
            # Compact if we have multiple small files
            if len(small_files) > 1 and total_size > threshold_mb:
                self._compact_files(small_files, dir_path)
    
    def _compact_files(self, files: List[str], output_dir: str):
        """Compact multiple files into one."""
        try:
            # Read all files
            dfs = []
            for file in files:
                df = self.backend.read(file, DataFormat.PARQUET)
                dfs.append(df)
            
            # Combine
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Write compacted file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_file = f"{output_dir}/compacted_{timestamp}.parquet"
            
            self.backend.write(
                combined_df,
                output_file,
                DataFormat.PARQUET,
                compression='snappy'
            )
            
            # Delete original files
            for file in files:
                self.backend.delete(file)
            
            logger.info(f"Compacted {len(files)} files into {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to compact files: {e}")


class BronzeLayer(DataLayer):
    """Bronze layer for raw data storage."""
    
    def write_raw_data(
        self,
        df: pd.DataFrame,
        source: str,
        dataset_name: str,
        partition_path: str,
        format: DataFormat = DataFormat.PARQUET,
        compression: str = "snappy"
    ) -> str:
        """Write raw data to bronze layer."""
        # Construct full path
        full_path = f"{self.path}/{source}/{dataset_name}/{partition_path}"
        
        # Generate filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"data_{timestamp}.{format.value}"
        file_path = f"{full_path}/{filename}"
        
        # Write data
        if format == DataFormat.PARQUET:
            self.backend.write(
                df,
                file_path,
                format,
                compression=compression,
                index=False
            )
        else:
            self.backend.write(df, file_path, format)
        
        # Write metadata
        metadata = {
            "source": source,
            "dataset": dataset_name,
            "records": len(df),
            "columns": list(df.columns),
            "ingestion_time": datetime.utcnow().isoformat(),
            "file_size": df.memory_usage(deep=True).sum(),
            "schema": df.dtypes.to_dict()
        }
        
        metadata_path = f"{full_path}/_metadata.json"
        self._write_metadata(metadata, metadata_path)
        
        return file_path
    
    def read_dataset(
        self,
        dataset_name: str,
        source: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Read dataset from bronze layer."""
        # Build path pattern
        if source:
            pattern = f"{self.path}/{source}/{dataset_name}"
        else:
            pattern = f"{self.path}/*/{dataset_name}"
        
        # List matching files
        files = self.backend.list_files(pattern, "*.parquet")
        
        # Filter by date if specified
        if start_date or end_date:
            files = self._filter_files_by_date(files, start_date, end_date)
        
        if not files:
            return pd.DataFrame()
        
        # Read and combine files
        dfs = []
        for file in files:
            try:
                df = self.backend.read(file, DataFormat.PARQUET)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to read {file}: {e}")
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _filter_files_by_date(
        self,
        files: List[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[str]:
        """Filter files by date range."""
        filtered = []
        
        for file in files:
            # Extract date from path (assuming year/month/day structure)
            parts = file.split('/')
            try:
                # Find year, month, day in path
                year = month = day = None
                for part in parts:
                    if part.startswith('year='):
                        year = int(part.split('=')[1])
                    elif part.startswith('month='):
                        month = int(part.split('=')[1])
                    elif part.startswith('day='):
                        day = int(part.split('=')[1])
                
                if year and month and day:
                    file_date = datetime(year, month, day)
                    
                    if start_date and file_date < start_date:
                        continue
                    if end_date and file_date > end_date:
                        continue
                    
                    filtered.append(file)
            except:
                # If date extraction fails, include the file
                filtered.append(file)
        
        return filtered
    
    def _write_metadata(self, metadata: Dict[str, Any], path: str):
        """Write metadata file."""
        # Convert to JSON string
        json_data = json.dumps(metadata, indent=2, default=str)
        
        # Write using backend (as a simple text file)
        if hasattr(self.backend, 'root_path'):
            # Local filesystem
            full_path = Path(self.backend.root_path) / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(json_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bronze layer statistics."""
        stats = {
            "total_size": 0,
            "file_count": 0,
            "dataset_count": 0,
            "sources": set(),
            "oldest_data": None,
            "newest_data": None
        }
        
        # List all files
        files = self.backend.list_files(self.path, pattern="*.parquet")
        stats["file_count"] = len(files)
        
        # Analyze files
        datasets = set()
        sources = set()
        
        for file in files:
            # Extract source and dataset from path
            parts = file.split('/')
            if len(parts) >= 3:
                sources.add(parts[1])
                datasets.add(parts[2])
            
            # Get file metadata
            metadata = self.backend.get_metadata(file)
            stats["total_size"] += metadata.get("size", 0)
            
            # Track dates
            modified = metadata.get("modified")
            if modified:
                if not stats["oldest_data"] or modified < stats["oldest_data"]:
                    stats["oldest_data"] = modified
                if not stats["newest_data"] or modified > stats["newest_data"]:
                    stats["newest_data"] = modified
        
        stats["dataset_count"] = len(datasets)
        stats["sources"] = list(sources)
        
        return stats
    
    def cleanup_old_data(self):
        """Clean up data older than retention period."""
        if self.retention_days <= 0:
            return  # No cleanup needed
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        logger.info(f"Cleaning up bronze data older than {cutoff_date}")
        
        # List all files
        files = self.backend.list_files(self.path, pattern="*.parquet")
        
        deleted_count = 0
        for file in files:
            metadata = self.backend.get_metadata(file)
            modified = metadata.get("modified")
            
            if modified and modified < cutoff_date:
                try:
                    self.backend.delete(file)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file}: {e}")
        
        logger.info(f"Deleted {deleted_count} old files from bronze layer")


class SilverLayer(DataLayer):
    """Silver layer for processed and validated data."""
    
    def write_processed_data(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        format: DataFormat = DataFormat.DELTA,
        partition_cols: Optional[List[str]] = None
    ) -> str:
        """Write processed data to silver layer."""
        # Default partitioning by date if available
        if partition_cols is None and '_processing_timestamp' in df.columns:
            # Extract date parts for partitioning
            df['year'] = pd.to_datetime(df['_processing_timestamp']).dt.year
            df['month'] = pd.to_datetime(df['_processing_timestamp']).dt.month
            df['day'] = pd.to_datetime(df['_processing_timestamp']).dt.day
            partition_cols = ['year', 'month', 'day']
        
        # Construct path
        dataset_path = f"{self.path}/{dataset_name}"
        
        if format == DataFormat.DELTA and self.spark:
            # Use Delta Lake
            spark_df = self.spark.createDataFrame(df)
            
            if self.backend.exists(dataset_path):
                # Append to existing Delta table
                spark_df.write \
                    .format("delta") \
                    .mode("append") \
                    .option("mergeSchema", "true") \
                    .save(dataset_path)
            else:
                # Create new Delta table
                writer = spark_df.write \
                    .format("delta") \
                    .mode("overwrite") \
                    .option("overwriteSchema", "true")
                
                if partition_cols:
                    writer = writer.partitionBy(*partition_cols)
                
                writer.save(dataset_path)
            
            # Optimize Delta table
            from delta.tables import DeltaTable
            delta_table = DeltaTable.forPath(self.spark, dataset_path)
            delta_table.optimize().executeCompaction()
            
        else:
            # Use Parquet with partitioning
            if partition_cols:
                # Save with partitions
                table = pa.Table.from_pandas(df)
                pq.write_to_dataset(
                    table,
                    root_path=dataset_path,
                    partition_cols=partition_cols,
                    compression='snappy',
                    existing_data_behavior='overwrite_or_ignore'
                )
            else:
                # Save as single file
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                file_path = f"{dataset_path}/data_{timestamp}.parquet"
                self.backend.write(df, file_path, DataFormat.PARQUET)
        
        # Update catalog
        self._update_catalog(dataset_name, df)
        
        return dataset_path
    
    def read_dataset(
        self,
        dataset_name: str,
        columns: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        version: Optional[int] = None
    ) -> pd.DataFrame:
        """Read dataset from silver layer."""
        dataset_path = f"{self.path}/{dataset_name}"
        
        if not self.backend.exists(dataset_path):
            logger.warning(f"Dataset {dataset_name} not found in silver layer")
            return pd.DataFrame()
        
        if self.spark:
            # Check if it's a Delta table
            try:
                if version is not None:
                    # Read specific version
                    df = self.spark.read \
                        .format("delta") \
                        .option("versionAsOf", version) \
                        .load(dataset_path)
                else:
                    df = self.spark.read.format("delta").load(dataset_path)
            except:
                # Fall back to Parquet
                df = self.spark.read.parquet(dataset_path)
            
            # Apply column selection
            if columns:
                df = df.select(*columns)
            
            # Apply filter
            if filter_expr:
                df = df.filter(filter_expr)
            
            return df.toPandas()
        else:
            # Read Parquet files
            return pd.read_parquet(dataset_path, columns=columns)
    
    def _update_catalog(self, dataset_name: str, df: pd.DataFrame):
        """Update data catalog with dataset information."""
        catalog_path = f"{self.path}/_catalog.json"
        
        # Load existing catalog
        catalog = {}
        if self.backend.exists(catalog_path):
            # Read catalog (simplified for local backend)
            if hasattr(self.backend, 'root_path'):
                with open(Path(self.backend.root_path) / catalog_path, 'r') as f:
                    catalog = json.load(f)
        
        # Update catalog entry
        catalog[dataset_name] = {
            "last_updated": datetime.utcnow().isoformat(),
            "record_count": len(df),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "size_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Write updated catalog
        if hasattr(self.backend, 'root_path'):
            catalog_full_path = Path(self.backend.root_path) / catalog_path
            catalog_full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(catalog_full_path, 'w') as f:
                json.dump(catalog, f, indent=2, default=str)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get silver layer statistics."""
        stats = {
            "total_size": 0,
            "file_count": 0,
            "dataset_count": 0,
            "delta_tables": 0,
            "parquet_tables": 0
        }
        
        # List all datasets
        datasets = set()
        if self.backend.exists(self.path):
            items = self.backend.list_files(self.path)
            
            # Extract dataset names
            for item in items:
                parts = item.split('/')
                if len(parts) >= 2 and not parts[1].startswith('_'):
                    datasets.add(parts[1])
        
        stats["dataset_count"] = len(datasets)
        
        # Analyze each dataset
        for dataset in datasets:
            dataset_path = f"{self.path}/{dataset}"
            
            # Check if Delta table
            delta_log_path = f"{dataset_path}/_delta_log"
            if self.backend.exists(delta_log_path):
                stats["delta_tables"] += 1
            else:
                stats["parquet_tables"] += 1
            
            # Count files and size
            files = self.backend.list_files(dataset_path, "*.parquet")
            stats["file_count"] += len(files)
            
            for file in files:
                metadata = self.backend.get_metadata(file)
                stats["total_size"] += metadata.get("size", 0)
        
        return stats
    
    def cleanup_old_data(self):
        """Clean up old data from silver layer."""
        if self.retention_days <= 0:
            return
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        logger.info(f"Cleaning up silver data older than {cutoff_date}")
        
        # For Delta tables, use time travel to delete old versions
        if self.spark:
            datasets = self._list_datasets()
            
            for dataset in datasets:
                dataset_path = f"{self.path}/{dataset}"
                
                try:
                    # Check if Delta table
                    delta_table = DeltaTable.forPath(self.spark, dataset_path)
                    
                    # Delete old versions
                    delta_table.vacuum(self.retention_days * 24)  # Convert days to hours
                    
                except Exception as e:
                    # Not a Delta table or error occurred
                    logger.debug(f"Could not vacuum {dataset}: {e}")
    
    def _list_datasets(self) -> List[str]:
        """List all datasets in silver layer."""
        datasets = []
        if self.backend.exists(self.path):
            items = self.backend.list_files(self.path)
            
            # Extract unique dataset names
            dataset_set = set()
            for item in items:
                parts = item.split('/')
                if len(parts) >= 2 and not parts[1].startswith('_'):
                    dataset_set.add(parts[1])
            
            datasets = list(dataset_set)
        
        return datasets


class GoldLayer(DataLayer):
    """Gold layer for business-ready datasets."""
    
    def write_business_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        business_date: Optional[datetime] = None,
        format: DataFormat = DataFormat.PARQUET,
        optimize_for_bi: bool = True
    ) -> str:
        """Write business-ready dataset to gold layer."""
        if business_date is None:
            business_date = datetime.utcnow().date()
        
        # Add business metadata
        df['_business_date'] = business_date
        df['_created_timestamp'] = datetime.utcnow()
        
        # Construct path with business domain structure
        dataset_path = f"{self.path}/{dataset_name}"
        
        if optimize_for_bi:
            # Optimize for BI tools
            df = self._optimize_for_bi(df)
        
        if format == DataFormat.DELTA and self.spark:
            # Use Delta with Z-ordering for optimal query performance
            spark_df = self.spark.createDataFrame(df)
            
            # Write with overwrite mode for gold layer (latest version)
            spark_df.write \
                .format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(dataset_path)
            
            # Apply Z-ordering on commonly queried columns
            delta_table = DeltaTable.forPath(self.spark, dataset_path)
            
            # Identify columns for Z-ordering (date and key dimensions)
            z_order_cols = self._identify_z_order_columns(df)
            if z_order_cols:
                delta_table.optimize().executeZOrderBy(*z_order_cols)
        else:
            # Write as optimized Parquet
            file_path = f"{dataset_path}/data_{business_date.strftime('%Y%m%d')}.parquet"
            
            # Use optimized Parquet settings
            self.backend.write(
                df,
                file_path,
                DataFormat.PARQUET,
                compression='snappy',
                engine='pyarrow',
                index=False,
                row_group_size=50000  # Optimize for query performance
            )
        
        # Update business catalog
        self._update_business_catalog(dataset_name, df, business_date)
        
        return dataset_path
    
    def _optimize_for_bi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataset for BI tool consumption."""
        # Convert data types for better BI compatibility
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to categorical for better performance
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
        
        # Sort by common dimensions for better compression
        sort_cols = []
        for col in ['date', 'business_date', 'symbol', 'exchange']:
            if col in df.columns:
                sort_cols.append(col)
        
        if sort_cols:
            df = df.sort_values(sort_cols)
        
        return df
    
    def _identify_z_order_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns for Z-ordering."""
        z_order_cols = []
        
        # Common patterns for Z-ordering
        priority_patterns = [
            ('date', 'symbol'),
            ('business_date', 'symbol'),
            ('timestamp', 'symbol'),
            ('year', 'month', 'symbol')
        ]
        
        for pattern in priority_patterns:
            if all(col in df.columns for col in pattern):
                z_order_cols = list(pattern)
                break
        
        return z_order_cols[:4]  # Delta supports up to 4 columns for Z-ordering
    
    def read_business_dataset(
        self,
        dataset_name: str,
        business_date: Optional[datetime] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Read business dataset from gold layer."""
        dataset_path = f"{self.path}/{dataset_name}"
        
        if not self.backend.exists(dataset_path):
            logger.warning(f"Dataset {dataset_name} not found in gold layer")
            return pd.DataFrame()
        
        if business_date:
            # Read specific business date
            file_path = f"{dataset_path}/data_{business_date.strftime('%Y%m%d')}.parquet"
            if self.backend.exists(file_path):
                return self.backend.read(file_path, DataFormat.PARQUET, columns=columns)
        
        # Read latest version
        files = self.backend.list_files(dataset_path, "*.parquet")
        if not files:
            return pd.DataFrame()
        
        # Get most recent file
        latest_file = sorted(files)[-1]
        return self.backend.read(latest_file, DataFormat.PARQUET, columns=columns)
    
    def _update_business_catalog(
        self,
        dataset_name: str,
        df: pd.DataFrame,
        business_date: datetime
    ):
        """Update business dataset catalog."""
        catalog_path = f"{self.path}/_business_catalog.json"
        
        # Load existing catalog
        catalog = {}
        if self.backend.exists(catalog_path):
            if hasattr(self.backend, 'root_path'):
                with open(Path(self.backend.root_path) / catalog_path, 'r') as f:
                    catalog = json.load(f)
        
        # Update catalog entry
        if dataset_name not in catalog:
            catalog[dataset_name] = {
                "created": datetime.utcnow().isoformat(),
                "versions": []
            }
        
        catalog[dataset_name]["last_updated"] = datetime.utcnow().isoformat()
        catalog[dataset_name]["latest_business_date"] = business_date.isoformat()
        catalog[dataset_name]["versions"].append({
            "business_date": business_date.isoformat(),
            "created": datetime.utcnow().isoformat(),
            "record_count": len(df),
            "columns": list(df.columns)
        })
        
        # Keep only last 10 versions in catalog
        catalog[dataset_name]["versions"] = catalog[dataset_name]["versions"][-10:]
        
        # Write updated catalog
        if hasattr(self.backend, 'root_path'):
            catalog_full_path = Path(self.backend.root_path) / catalog_path
            catalog_full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(catalog_full_path, 'w') as f:
                json.dump(catalog, f, indent=2, default=str)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get gold layer statistics."""
        stats = {
            "total_size": 0,
            "file_count": 0,
            "dataset_count": 0,
            "business_datasets": []
        }
        
        # Load business catalog
        catalog_path = f"{self.path}/_business_catalog.json"
        if self.backend.exists(catalog_path) and hasattr(self.backend, 'root_path'):
            with open(Path(self.backend.root_path) / catalog_path, 'r') as f:
                catalog = json.load(f)
            
            stats["dataset_count"] = len(catalog)
            stats["business_datasets"] = list(catalog.keys())
        
        # Calculate sizes
        files = self.backend.list_files(self.path, "*.parquet")
        stats["file_count"] = len(files)
        
        for file in files:
            metadata = self.backend.get_metadata(file)
            stats["total_size"] += metadata.get("size", 0)
        
        return stats
    
    def cleanup_old_data(self):
        """Gold layer typically doesn't delete data (permanent storage)."""
        if self.retention_days > 0:
            logger.info("Gold layer data is permanent - skipping cleanup")
        return