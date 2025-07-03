"""
Partitioning strategies for data lake storage.

Provides intelligent partitioning schemes for optimal query performance
and storage efficiency.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum
import hashlib
import logging

import pandas as pd


logger = logging.getLogger(__name__)


class PartitionScheme(Enum):
    """Supported partitioning schemes."""
    TIME_BASED = "time_based"
    SYMBOL_BASED = "symbol_based"
    HASH_BASED = "hash_based"
    RANGE_BASED = "range_based"
    LIST_BASED = "list_based"
    COMPOSITE = "composite"
    DYNAMIC = "dynamic"


@dataclass
class PartitionConfig:
    """Configuration for partitioning."""
    scheme: PartitionScheme
    columns: List[str]
    time_granularity: Optional[str] = "day"  # hour, day, month, year
    hash_buckets: Optional[int] = 10
    range_size: Optional[int] = None
    list_values: Optional[List[Any]] = None
    max_partition_size_mb: float = 1024  # 1GB default
    min_partition_size_mb: float = 128   # 128MB default
    
    def validate(self):
        """Validate partition configuration."""
        if not self.columns:
            raise ValueError("At least one partition column required")
        
        if self.scheme == PartitionScheme.HASH_BASED and not self.hash_buckets:
            raise ValueError("Hash-based partitioning requires hash_buckets")
        
        if self.scheme == PartitionScheme.LIST_BASED and not self.list_values:
            raise ValueError("List-based partitioning requires list_values")


class PartitioningStrategy:
    """Manages data partitioning strategies."""
    
    def __init__(self):
        """Initialize partitioning strategy."""
        self.partition_stats = {}
    
    def generate_partition_path(
        self,
        data: pd.DataFrame,
        config: PartitionConfig
    ) -> Dict[int, str]:
        """Generate partition paths for DataFrame rows."""
        config.validate()
        
        if config.scheme == PartitionScheme.TIME_BASED:
            return self._time_based_partition(data, config)
        elif config.scheme == PartitionScheme.SYMBOL_BASED:
            return self._symbol_based_partition(data, config)
        elif config.scheme == PartitionScheme.HASH_BASED:
            return self._hash_based_partition(data, config)
        elif config.scheme == PartitionScheme.COMPOSITE:
            return self._composite_partition(data, config)
        elif config.scheme == PartitionScheme.DYNAMIC:
            return self._dynamic_partition(data, config)
        else:
            raise ValueError(f"Unsupported partition scheme: {config.scheme}")
    
    def generate_time_partition(
        self,
        timestamp: datetime,
        level: str = "day"
    ) -> str:
        """Generate time-based partition path."""
        if level == "hour":
            return (f"year={timestamp.year}/"
                   f"month={timestamp.month:02d}/"
                   f"day={timestamp.day:02d}/"
                   f"hour={timestamp.hour:02d}")
        elif level == "day":
            return (f"year={timestamp.year}/"
                   f"month={timestamp.month:02d}/"
                   f"day={timestamp.day:02d}")
        elif level == "month":
            return (f"year={timestamp.year}/"
                   f"month={timestamp.month:02d}")
        elif level == "year":
            return f"year={timestamp.year}"
        else:
            raise ValueError(f"Invalid time level: {level}")
    
    def generate_symbol_partition(self, symbol: str, prefix_length: int = 2) -> str:
        """Generate symbol-based partition path."""
        # Clean symbol
        clean_symbol = symbol.replace("/", "_").replace("-", "_").upper()
        
        # Use prefix for distribution
        if len(clean_symbol) >= prefix_length:
            prefix = clean_symbol[:prefix_length]
        else:
            prefix = clean_symbol.ljust(prefix_length, '0')
        
        return f"symbol_prefix={prefix}/symbol={clean_symbol}"
    
    def generate_hash_partition(self, key: str, buckets: int = 10) -> str:
        """Generate hash-based partition path."""
        # Create hash of key
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        bucket = hash_value % buckets
        
        return f"hash_bucket={bucket:02d}"
    
    def _time_based_partition(
        self,
        data: pd.DataFrame,
        config: PartitionConfig
    ) -> Dict[int, str]:
        """Generate time-based partitions."""
        partitions = {}
        time_col = config.columns[0]
        
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found in data")
        
        # Convert to datetime if needed
        time_series = pd.to_datetime(data[time_col])
        
        for idx, timestamp in enumerate(time_series):
            partitions[idx] = self.generate_time_partition(
                timestamp, 
                config.time_granularity
            )
        
        return partitions
    
    def _symbol_based_partition(
        self,
        data: pd.DataFrame,
        config: PartitionConfig
    ) -> Dict[int, str]:
        """Generate symbol-based partitions."""
        partitions = {}
        symbol_col = config.columns[0]
        
        if symbol_col not in data.columns:
            raise ValueError(f"Symbol column '{symbol_col}' not found in data")
        
        for idx, symbol in enumerate(data[symbol_col]):
            partitions[idx] = self.generate_symbol_partition(str(symbol))
        
        return partitions
    
    def _hash_based_partition(
        self,
        data: pd.DataFrame,
        config: PartitionConfig
    ) -> Dict[int, str]:
        """Generate hash-based partitions."""
        partitions = {}
        
        # Create composite key from columns
        key_columns = [data[col].astype(str) for col in config.columns]
        
        for idx in range(len(data)):
            # Combine column values for hash key
            key_parts = [col.iloc[idx] for col in key_columns]
            key = "_".join(key_parts)
            
            partitions[idx] = self.generate_hash_partition(
                key, 
                config.hash_buckets
            )
        
        return partitions
    
    def _composite_partition(
        self,
        data: pd.DataFrame,
        config: PartitionConfig
    ) -> Dict[int, str]:
        """Generate composite partitions (e.g., time + symbol)."""
        partitions = {}
        
        # Assume first column is time, second is symbol
        if len(config.columns) < 2:
            raise ValueError("Composite partitioning requires at least 2 columns")
        
        time_col = config.columns[0]
        symbol_col = config.columns[1]
        
        time_series = pd.to_datetime(data[time_col])
        
        for idx in range(len(data)):
            time_part = self.generate_time_partition(
                time_series.iloc[idx],
                config.time_granularity
            )
            symbol_part = self.generate_symbol_partition(
                str(data[symbol_col].iloc[idx])
            )
            
            partitions[idx] = f"{time_part}/{symbol_part}"
        
        return partitions
    
    def _dynamic_partition(
        self,
        data: pd.DataFrame,
        config: PartitionConfig
    ) -> Dict[int, str]:
        """Generate dynamic partitions based on data characteristics."""
        # Analyze data distribution
        analysis = self.analyze_data_distribution(data, config.columns)
        
        # Choose best partitioning strategy
        if analysis["recommendation"] == "time":
            return self._time_based_partition(data, config)
        elif analysis["recommendation"] == "hash":
            return self._hash_based_partition(data, config)
        else:
            # Default to hash-based
            return self._hash_based_partition(data, config)
    
    def analyze_data_distribution(
        self,
        data: pd.DataFrame,
        columns: List[str]
    ) -> Dict[str, Any]:
        """Analyze data distribution for optimal partitioning."""
        analysis = {
            "total_rows": len(data),
            "columns": {},
            "recommendation": None
        }
        
        for col in columns:
            if col not in data.columns:
                continue
            
            col_analysis = {
                "unique_values": data[col].nunique(),
                "null_count": data[col].isnull().sum(),
                "data_type": str(data[col].dtype)
            }
            
            # Check if it's a time column
            if "date" in col.lower() or "time" in col.lower():
                try:
                    time_series = pd.to_datetime(data[col])
                    col_analysis["is_time"] = True
                    col_analysis["time_range"] = {
                        "min": time_series.min(),
                        "max": time_series.max(),
                        "span_days": (time_series.max() - time_series.min()).days
                    }
                except:
                    col_analysis["is_time"] = False
            
            # Check cardinality
            cardinality_ratio = col_analysis["unique_values"] / len(data)
            col_analysis["cardinality_ratio"] = cardinality_ratio
            
            if cardinality_ratio < 0.01:
                col_analysis["cardinality"] = "low"
            elif cardinality_ratio < 0.1:
                col_analysis["cardinality"] = "medium"
            else:
                col_analysis["cardinality"] = "high"
            
            analysis["columns"][col] = col_analysis
        
        # Make recommendation
        has_time_column = any(
            c.get("is_time", False) for c in analysis["columns"].values()
        )
        
        if has_time_column:
            analysis["recommendation"] = "time"
        else:
            # Look for low cardinality columns
            low_cardinality_cols = [
                col for col, stats in analysis["columns"].items()
                if stats.get("cardinality") == "low"
            ]
            
            if low_cardinality_cols:
                analysis["recommendation"] = "list"
            else:
                analysis["recommendation"] = "hash"
        
        return analysis
    
    def optimize_partitions(
        self,
        current_partitions: Dict[str, int],
        target_size_mb: float = 512
    ) -> List[Tuple[List[str], str]]:
        """Optimize partitions by merging small ones."""
        # Group partitions by size
        small_partitions = []
        good_partitions = []
        
        mb_per_row = 0.001  # Rough estimate
        
        for partition, row_count in current_partitions.items():
            estimated_size_mb = row_count * mb_per_row
            
            if estimated_size_mb < target_size_mb * 0.5:
                small_partitions.append((partition, row_count, estimated_size_mb))
            else:
                good_partitions.append((partition, row_count, estimated_size_mb))
        
        # Merge small partitions
        merged_partitions = []
        current_group = []
        current_size = 0
        
        for partition, row_count, size_mb in sorted(small_partitions):
            if current_size + size_mb > target_size_mb:
                # Start new group
                if current_group:
                    merged_partitions.append((
                        [p[0] for p in current_group],
                        f"merged_{len(merged_partitions)}"
                    ))
                current_group = [(partition, row_count, size_mb)]
                current_size = size_mb
            else:
                current_group.append((partition, row_count, size_mb))
                current_size += size_mb
        
        # Add last group
        if current_group:
            merged_partitions.append((
                [p[0] for p in current_group],
                f"merged_{len(merged_partitions)}"
            ))
        
        logger.info(
            f"Optimized {len(small_partitions)} small partitions "
            f"into {len(merged_partitions)} merged partitions"
        )
        
        return merged_partitions
    
    def suggest_partition_scheme(
        self,
        data_sample: pd.DataFrame,
        expected_volume: int,
        query_patterns: Optional[List[str]] = None
    ) -> PartitionConfig:
        """Suggest optimal partition scheme based on data and usage."""
        # Analyze data
        numeric_cols = data_sample.select_dtypes(include=['number']).columns.tolist()
        string_cols = data_sample.select_dtypes(include=['object']).columns.tolist()
        date_cols = []
        
        # Identify date columns
        for col in data_sample.columns:
            if "date" in col.lower() or "time" in col.lower():
                try:
                    pd.to_datetime(data_sample[col])
                    date_cols.append(col)
                except:
                    pass
        
        # Analyze query patterns if provided
        filter_cols = set()
        if query_patterns:
            for pattern in query_patterns:
                # Simple pattern extraction
                pattern_lower = pattern.lower()
                for col in data_sample.columns:
                    if col.lower() in pattern_lower:
                        filter_cols.add(col)
        
        # Make recommendation
        if date_cols and (not filter_cols or any(d in filter_cols for d in date_cols)):
            # Time-based partitioning
            time_col = date_cols[0]
            
            # Determine granularity based on volume
            if expected_volume > 1e9:  # > 1 billion rows
                granularity = "hour"
            elif expected_volume > 1e7:  # > 10 million rows
                granularity = "day"
            else:
                granularity = "month"
            
            # Check if composite partitioning would help
            if filter_cols and len(filter_cols) > 1:
                other_cols = list(filter_cols - {time_col})[:1]  # Max 1 additional
                return PartitionConfig(
                    scheme=PartitionScheme.COMPOSITE,
                    columns=[time_col] + other_cols,
                    time_granularity=granularity
                )
            else:
                return PartitionConfig(
                    scheme=PartitionScheme.TIME_BASED,
                    columns=[time_col],
                    time_granularity=granularity
                )
        
        elif filter_cols:
            # Use most selective column
            selectivity = {}
            for col in filter_cols:
                if col in data_sample.columns:
                    selectivity[col] = data_sample[col].nunique() / len(data_sample)
            
            if selectivity:
                # Sort by selectivity (lower is better for partitioning)
                best_col = min(selectivity.items(), key=lambda x: x[1])[0]
                
                if selectivity[best_col] < 0.01:  # Very low cardinality
                    return PartitionConfig(
                        scheme=PartitionScheme.LIST_BASED,
                        columns=[best_col],
                        list_values=data_sample[best_col].unique().tolist()
                    )
                else:
                    return PartitionConfig(
                        scheme=PartitionScheme.HASH_BASED,
                        columns=[best_col],
                        hash_buckets=min(100, expected_volume // 1000000)
                    )
        
        # Default to hash-based on primary key or first column
        return PartitionConfig(
            scheme=PartitionScheme.HASH_BASED,
            columns=[data_sample.columns[0]],
            hash_buckets=min(50, expected_volume // 1000000)
        )