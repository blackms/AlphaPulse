"""
Utility functions for data lake operations.

Provides helper functions for common data lake tasks including
data movement, format conversion, and metadata management.
"""

import os
import shutil
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Optional Delta Lake support
try:
    from delta import DeltaTable
    DELTA_AVAILABLE = True
except ImportError:
    DELTA_AVAILABLE = False
    DeltaTable = None


logger = logging.getLogger(__name__)


class DataLakeUtils:
    """Utility functions for data lake operations."""
    
    @staticmethod
    def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
        """Calculate hash of a file."""
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def estimate_dataframe_size(df: pd.DataFrame) -> Dict[str, float]:
        """Estimate DataFrame size in memory and on disk."""
        memory_size = df.memory_usage(deep=True).sum()
        
        # Estimate compressed sizes
        estimates = {
            "memory_mb": memory_size / (1024 * 1024),
            "csv_mb": memory_size * 0.3 / (1024 * 1024),  # Rough estimate
            "parquet_snappy_mb": memory_size * 0.2 / (1024 * 1024),
            "parquet_gzip_mb": memory_size * 0.15 / (1024 * 1024),
            "json_mb": memory_size * 0.4 / (1024 * 1024)
        }
        
        return estimates
    
    @staticmethod
    def convert_format(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        input_format: str,
        output_format: str,
        **kwargs
    ) -> bool:
        """Convert file between different formats."""
        try:
            # Read data
            if input_format == "csv":
                df = pd.read_csv(input_path, **kwargs.get('read_options', {}))
            elif input_format == "parquet":
                df = pd.read_parquet(input_path, **kwargs.get('read_options', {}))
            elif input_format == "json":
                df = pd.read_json(input_path, **kwargs.get('read_options', {}))
            elif input_format == "excel":
                df = pd.read_excel(input_path, **kwargs.get('read_options', {}))
            else:
                raise ValueError(f"Unsupported input format: {input_format}")
            
            # Write data
            if output_format == "csv":
                df.to_csv(output_path, index=False, **kwargs.get('write_options', {}))
            elif output_format == "parquet":
                df.to_parquet(output_path, index=False, **kwargs.get('write_options', {}))
            elif output_format == "json":
                df.to_json(output_path, **kwargs.get('write_options', {}))
            elif output_format == "excel":
                df.to_excel(output_path, index=False, **kwargs.get('write_options', {}))
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            return True
            
        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            return False
    
    @staticmethod
    def split_large_file(
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        chunk_size_mb: float = 128,
        format: str = "parquet"
    ) -> List[str]:
        """Split large file into smaller chunks."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        
        try:
            # Read file in chunks
            if format == "csv":
                chunk_iter = pd.read_csv(input_path, chunksize=10000)
            elif format == "parquet":
                # For parquet, read the whole file and split
                df = pd.read_parquet(input_path)
                chunk_size_rows = int(chunk_size_mb * 1024 * 1024 / 
                                     (df.memory_usage(deep=True).sum() / len(df)))
                chunk_iter = [df[i:i+chunk_size_rows] 
                             for i in range(0, len(df), chunk_size_rows)]
            else:
                raise ValueError(f"Unsupported format for splitting: {format}")
            
            # Write chunks
            for i, chunk in enumerate(chunk_iter):
                output_file = output_dir / f"chunk_{i:04d}.{format}"
                
                if format == "csv":
                    chunk.to_csv(output_file, index=False)
                elif format == "parquet":
                    chunk.to_parquet(output_file, index=False)
                
                output_files.append(str(output_file))
            
            logger.info(f"Split {input_path} into {len(output_files)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to split file: {e}")
        
        return output_files
    
    @staticmethod
    def merge_files(
        input_files: List[Union[str, Path]],
        output_path: Union[str, Path],
        format: str = "parquet",
        remove_duplicates: bool = False
    ) -> bool:
        """Merge multiple files into one."""
        try:
            dfs = []
            
            # Read all files
            for file in input_files:
                if format == "csv":
                    df = pd.read_csv(file)
                elif format == "parquet":
                    df = pd.read_parquet(file)
                elif format == "json":
                    df = pd.read_json(file)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                dfs.append(df)
            
            # Combine
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Remove duplicates if requested
            if remove_duplicates:
                combined_df = combined_df.drop_duplicates()
            
            # Write output
            if format == "csv":
                combined_df.to_csv(output_path, index=False)
            elif format == "parquet":
                combined_df.to_parquet(output_path, index=False)
            elif format == "json":
                combined_df.to_json(output_path)
            
            logger.info(f"Merged {len(input_files)} files into {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to merge files: {e}")
            return False
    
    @staticmethod
    def optimize_parquet_file(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        row_group_size: int = 50000,
        compression: str = "snappy",
        column_encoding: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Optimize Parquet file for better performance."""
        try:
            # Read parquet file
            table = pq.read_table(input_path)
            
            # Get original stats
            original_size = Path(input_path).stat().st_size
            
            # Optimize schema
            optimized_schema = DataLakeUtils._optimize_arrow_schema(table.schema)
            
            # Convert with optimized schema
            optimized_table = table.cast(optimized_schema)
            
            # Write with optimization
            pq.write_table(
                optimized_table,
                output_path,
                row_group_size=row_group_size,
                compression=compression,
                use_dictionary=True,
                column_encoding=column_encoding or {}
            )
            
            # Get new stats
            new_size = Path(output_path).stat().st_size
            
            return {
                "original_size_mb": original_size / (1024 * 1024),
                "optimized_size_mb": new_size / (1024 * 1024),
                "compression_ratio": original_size / new_size if new_size > 0 else 0,
                "space_saved_pct": (1 - new_size / original_size) * 100 if original_size > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize parquet file: {e}")
            return {}
    
    @staticmethod
    def _optimize_arrow_schema(schema: pa.Schema) -> pa.Schema:
        """Optimize Arrow schema for better storage."""
        fields = []
        
        for field in schema:
            # Optimize data types
            if pa.types.is_string(field.type):
                # Convert to dictionary encoding for low cardinality strings
                fields.append(pa.field(field.name, pa.dictionary(pa.int32(), pa.string())))
            elif pa.types.is_float64(field.type):
                # Downcast to float32 if possible
                fields.append(pa.field(field.name, pa.float32()))
            else:
                fields.append(field)
        
        return pa.schema(fields)
    
    @staticmethod
    def create_external_table_ddl(
        table_name: str,
        location: str,
        schema: Dict[str, str],
        format: str = "parquet",
        partitions: Optional[List[str]] = None,
        engine: str = "spark"
    ) -> str:
        """Generate DDL for external table creation."""
        # Build column definitions
        columns = []
        partition_cols = set(partitions or [])
        
        for col_name, col_type in schema.items():
            if col_name not in partition_cols:
                columns.append(f"{col_name} {col_type}")
        
        columns_str = ",\n  ".join(columns)
        
        if engine == "spark":
            ddl = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
  {columns_str}
)
USING {format}
"""
            if partitions:
                partition_str = ", ".join(partitions)
                ddl += f"PARTITIONED BY ({partition_str})\n"
            
            ddl += f"LOCATION '{location}'"
            
        elif engine == "hive":
            ddl = f"""
CREATE EXTERNAL TABLE IF NOT EXISTS {table_name} (
  {columns_str}
)
"""
            if partitions:
                partition_cols_str = ",\n  ".join(
                    f"{col} {schema[col]}" for col in partitions
                )
                ddl += f"PARTITIONED BY (\n  {partition_cols_str}\n)\n"
            
            if format == "parquet":
                ddl += "STORED AS PARQUET\n"
            elif format == "orc":
                ddl += "STORED AS ORC\n"
            
            ddl += f"LOCATION '{location}'"
            
        elif engine == "presto":
            ddl = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
  {columns_str}
)
WITH (
  format = '{format}',
  external_location = '{location}'
"""
            if partitions:
                ddl += f",\n  partitioned_by = ARRAY[{', '.join(repr(p) for p in partitions)}]"
            
            ddl += "\n)"
        
        return ddl.strip()
    
    @staticmethod
    def validate_schema_compatibility(
        schema1: Dict[str, str],
        schema2: Dict[str, str]
    ) -> Tuple[bool, List[str]]:
        """Validate if two schemas are compatible."""
        issues = []
        
        # Check for missing columns
        schema1_cols = set(schema1.keys())
        schema2_cols = set(schema2.keys())
        
        missing_in_2 = schema1_cols - schema2_cols
        missing_in_1 = schema2_cols - schema1_cols
        
        if missing_in_2:
            issues.append(f"Columns missing in schema2: {missing_in_2}")
        if missing_in_1:
            issues.append(f"Columns missing in schema1: {missing_in_1}")
        
        # Check type compatibility
        common_cols = schema1_cols & schema2_cols
        for col in common_cols:
            type1 = schema1[col]
            type2 = schema2[col]
            
            if not DataLakeUtils._are_types_compatible(type1, type2):
                issues.append(f"Type mismatch for column {col}: {type1} vs {type2}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def _are_types_compatible(type1: str, type2: str) -> bool:
        """Check if two data types are compatible."""
        # Normalize types
        type1 = type1.lower()
        type2 = type2.lower()
        
        if type1 == type2:
            return True
        
        # Define compatible type pairs
        compatible_pairs = [
            ("int", "bigint"),
            ("float", "double"),
            ("string", "varchar"),
            ("timestamp", "datetime")
        ]
        
        for t1, t2 in compatible_pairs:
            if (type1 == t1 and type2 == t2) or (type1 == t2 and type2 == t1):
                return True
        
        return False
    
    @staticmethod
    def generate_data_sample(
        file_path: Union[str, Path],
        sample_size: int = 1000,
        format: str = "parquet"
    ) -> pd.DataFrame:
        """Generate a sample from a data file."""
        try:
            if format == "parquet":
                # For parquet, we can read specific row groups
                parquet_file = pq.ParquetFile(file_path)
                total_rows = parquet_file.metadata.num_rows
                
                if total_rows <= sample_size:
                    return pd.read_parquet(file_path)
                
                # Sample row groups
                num_row_groups = parquet_file.num_row_groups
                rows_per_group = total_rows // num_row_groups
                groups_to_read = min(
                    num_row_groups,
                    max(1, sample_size // rows_per_group)
                )
                
                # Read sample
                table = parquet_file.read_row_groups(list(range(groups_to_read)))
                df = table.to_pandas()
                
                # Further sample if needed
                if len(df) > sample_size:
                    df = df.sample(n=sample_size)
                
                return df
                
            elif format == "csv":
                # For CSV, read first N rows
                return pd.read_csv(file_path, nrows=sample_size)
            
            else:
                # For other formats, read all and sample
                if format == "json":
                    df = pd.read_json(file_path)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                if len(df) > sample_size:
                    return df.sample(n=sample_size)
                return df
                
        except Exception as e:
            logger.error(f"Failed to generate sample: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def calculate_table_statistics(
        df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate comprehensive statistics for a DataFrame."""
        stats = {}
        
        for col in df.columns:
            col_stats = {
                "count": df[col].count(),
                "null_count": df[col].isnull().sum(),
                "null_percentage": df[col].isnull().sum() / len(df) * 100,
                "unique_count": df[col].nunique(),
                "data_type": str(df[col].dtype)
            }
            
            # Numeric statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                    "percentiles": {
                        "25%": float(df[col].quantile(0.25)),
                        "50%": float(df[col].quantile(0.50)),
                        "75%": float(df[col].quantile(0.75))
                    }
                })
            
            # String statistics
            elif pd.api.types.is_string_dtype(df[col]):
                col_stats.update({
                    "avg_length": df[col].str.len().mean(),
                    "max_length": df[col].str.len().max(),
                    "min_length": df[col].str.len().min()
                })
                
                # Top values
                top_values = df[col].value_counts().head(10)
                col_stats["top_values"] = top_values.to_dict()
            
            # Date statistics
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_stats.update({
                    "min_date": df[col].min().isoformat() if pd.notna(df[col].min()) else None,
                    "max_date": df[col].max().isoformat() if pd.notna(df[col].max()) else None,
                    "date_range_days": (df[col].max() - df[col].min()).days if pd.notna(df[col].min()) else None
                })
            
            stats[col] = col_stats
        
        return stats
    
    @staticmethod
    def parallel_file_operation(
        files: List[Union[str, Path]],
        operation: Callable,
        max_workers: int = 4,
        **kwargs
    ) -> List[Any]:
        """Execute file operation in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(operation, file, **kwargs): file
                for file in files
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    results.append((file, result))
                except Exception as e:
                    logger.error(f"Operation failed for {file}: {e}")
                    results.append((file, None))
        
        return results