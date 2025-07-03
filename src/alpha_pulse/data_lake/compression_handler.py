"""
Compression handler for data lake storage.

Manages compression strategies for different data types and formats
to optimize storage costs and query performance.
"""

from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import logging
import zlib
import gzip
import bz2
import lzma
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np


logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    SNAPPY = "snappy"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    BROTLI = "brotli"
    BZ2 = "bz2"
    LZMA = "lzma"


@dataclass
class CompressionProfile:
    """Compression profile for different use cases."""
    name: str
    compression_type: CompressionType
    compression_level: Optional[int] = None
    target_file_size_mb: float = 128
    optimize_for: str = "balanced"  # speed, ratio, balanced
    
    def get_parquet_compression(self) -> str:
        """Get compression format for Parquet."""
        if self.compression_type == CompressionType.NONE:
            return None
        elif self.compression_type in [CompressionType.SNAPPY, CompressionType.GZIP, 
                                     CompressionType.LZ4, CompressionType.ZSTD,
                                     CompressionType.BROTLI]:
            return self.compression_type.value
        else:
            # Default to snappy for unsupported types
            return "snappy"


class CompressionHandler:
    """Handles data compression for storage optimization."""
    
    # Predefined compression profiles
    PROFILES = {
        "hot_data": CompressionProfile(
            name="hot_data",
            compression_type=CompressionType.SNAPPY,
            optimize_for="speed"
        ),
        "warm_data": CompressionProfile(
            name="warm_data",
            compression_type=CompressionType.ZSTD,
            compression_level=3,
            optimize_for="balanced"
        ),
        "cold_data": CompressionProfile(
            name="cold_data",
            compression_type=CompressionType.GZIP,
            compression_level=6,
            optimize_for="ratio"
        ),
        "archive": CompressionProfile(
            name="archive",
            compression_type=CompressionType.LZMA,
            compression_level=9,
            optimize_for="ratio"
        )
    }
    
    def __init__(self, default_compression: str = "snappy"):
        """Initialize compression handler."""
        self.default_compression = CompressionType(default_compression)
        self.compression_stats = {}
    
    def compress_dataframe(
        self,
        df: pd.DataFrame,
        profile: Optional[CompressionProfile] = None,
        format: str = "parquet"
    ) -> bytes:
        """Compress DataFrame to bytes."""
        if profile is None:
            profile = self.PROFILES["hot_data"]
        
        if format == "parquet":
            return self._compress_parquet(df, profile)
        elif format == "csv":
            return self._compress_csv(df, profile)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _compress_parquet(
        self,
        df: pd.DataFrame,
        profile: CompressionProfile
    ) -> bytes:
        """Compress DataFrame to Parquet bytes."""
        import io
        
        # Convert to Arrow table
        table = pa.Table.from_pandas(df)
        
        # Create buffer
        buffer = io.BytesIO()
        
        # Write with compression
        pq.write_table(
            table,
            buffer,
            compression=profile.get_parquet_compression(),
            compression_level=profile.compression_level
        )
        
        return buffer.getvalue()
    
    def _compress_csv(
        self,
        df: pd.DataFrame,
        profile: CompressionProfile
    ) -> bytes:
        """Compress DataFrame to CSV bytes."""
        csv_str = df.to_csv(index=False)
        csv_bytes = csv_str.encode('utf-8')
        
        if profile.compression_type == CompressionType.GZIP:
            return gzip.compress(
                csv_bytes,
                compresslevel=profile.compression_level or 6
            )
        elif profile.compression_type == CompressionType.BZ2:
            return bz2.compress(
                csv_bytes,
                compresslevel=profile.compression_level or 9
            )
        elif profile.compression_type == CompressionType.LZMA:
            return lzma.compress(
                csv_bytes,
                preset=profile.compression_level or 6
            )
        else:
            return csv_bytes
    
    def analyze_compression_ratio(
        self,
        df: pd.DataFrame,
        test_all: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze compression ratios for different algorithms."""
        original_size = df.memory_usage(deep=True).sum()
        results = {}
        
        # Test different compression types
        if test_all:
            test_profiles = list(self.PROFILES.values())
        else:
            test_profiles = [
                self.PROFILES["hot_data"],
                self.PROFILES["warm_data"],
                self.PROFILES["cold_data"]
            ]
        
        for profile in test_profiles:
            try:
                import time
                start_time = time.time()
                
                compressed = self.compress_dataframe(df, profile)
                compressed_size = len(compressed)
                compression_time = time.time() - start_time
                
                results[profile.name] = {
                    "compression_type": profile.compression_type.value,
                    "original_size_mb": original_size / (1024 * 1024),
                    "compressed_size_mb": compressed_size / (1024 * 1024),
                    "compression_ratio": original_size / compressed_size,
                    "space_savings_pct": (1 - compressed_size / original_size) * 100,
                    "compression_time_sec": compression_time
                }
            except Exception as e:
                logger.error(f"Failed to test {profile.name}: {e}")
        
        # Find best compression
        if results:
            best_ratio = max(results.items(), key=lambda x: x[1]["compression_ratio"])
            best_speed = min(results.items(), key=lambda x: x[1]["compression_time_sec"])
            
            results["recommendation"] = {
                "best_ratio": best_ratio[0],
                "best_speed": best_speed[0],
                "balanced": self._find_balanced_compression(results)
            }
        
        return results
    
    def _find_balanced_compression(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> str:
        """Find balanced compression considering ratio and speed."""
        scores = {}
        
        # Normalize metrics
        max_ratio = max(r["compression_ratio"] for r in results.values() if "compression_ratio" in r)
        min_time = min(r["compression_time_sec"] for r in results.values() if "compression_time_sec" in r)
        
        for name, metrics in results.items():
            if "compression_ratio" not in metrics:
                continue
            
            # Score = 0.6 * ratio_score + 0.4 * speed_score
            ratio_score = metrics["compression_ratio"] / max_ratio
            speed_score = min_time / metrics["compression_time_sec"]
            
            scores[name] = 0.6 * ratio_score + 0.4 * speed_score
        
        return max(scores.items(), key=lambda x: x[1])[0] if scores else "hot_data"
    
    def suggest_compression(
        self,
        data_profile: Dict[str, Any],
        access_pattern: str = "balanced"
    ) -> CompressionProfile:
        """Suggest compression based on data profile and access pattern."""
        # Access patterns: hot, warm, cold, archive
        if access_pattern == "hot":
            return self.PROFILES["hot_data"]
        elif access_pattern == "archive":
            return self.PROFILES["archive"]
        
        # Analyze data characteristics
        avg_row_size = data_profile.get("avg_row_size", 1000)
        num_columns = data_profile.get("num_columns", 10)
        data_types = data_profile.get("data_types", {})
        
        # High cardinality string data compresses well with ZSTD
        string_ratio = data_types.get("object", 0) / num_columns if num_columns > 0 else 0
        
        if string_ratio > 0.5:
            # Lots of string data
            if access_pattern == "warm":
                return self.PROFILES["warm_data"]
            else:
                return self.PROFILES["cold_data"]
        
        # Numeric data with patterns compresses well
        if avg_row_size < 500:
            # Small rows, prioritize speed
            return self.PROFILES["hot_data"]
        else:
            # Larger rows, balance compression
            return self.PROFILES["warm_data"]
    
    def estimate_storage_cost(
        self,
        uncompressed_size_gb: float,
        compression_profile: CompressionProfile,
        storage_class: str = "standard"
    ) -> Dict[str, float]:
        """Estimate storage costs with compression."""
        # Typical compression ratios
        compression_ratios = {
            CompressionType.NONE: 1.0,
            CompressionType.SNAPPY: 2.5,
            CompressionType.GZIP: 4.0,
            CompressionType.LZ4: 2.8,
            CompressionType.ZSTD: 3.5,
            CompressionType.BROTLI: 4.5,
            CompressionType.BZ2: 4.2,
            CompressionType.LZMA: 5.0
        }
        
        # Storage costs per GB per month (example rates)
        storage_costs = {
            "standard": 0.023,      # $0.023/GB/month
            "infrequent": 0.0125,   # $0.0125/GB/month
            "archive": 0.004        # $0.004/GB/month
        }
        
        compression_ratio = compression_ratios.get(
            compression_profile.compression_type, 
            2.5
        )
        
        compressed_size_gb = uncompressed_size_gb / compression_ratio
        monthly_cost = compressed_size_gb * storage_costs.get(storage_class, 0.023)
        
        return {
            "uncompressed_size_gb": uncompressed_size_gb,
            "compressed_size_gb": compressed_size_gb,
            "compression_ratio": compression_ratio,
            "space_savings_pct": (1 - 1/compression_ratio) * 100,
            "monthly_cost_usd": monthly_cost,
            "annual_cost_usd": monthly_cost * 12,
            "storage_class": storage_class
        }
    
    def compress_file(
        self,
        input_path: Path,
        output_path: Path,
        profile: Optional[CompressionProfile] = None
    ) -> Dict[str, Any]:
        """Compress a file using specified profile."""
        if profile is None:
            profile = self.PROFILES["warm_data"]
        
        input_size = input_path.stat().st_size
        
        # Determine compression based on file extension
        if input_path.suffix == '.parquet':
            # Recompress Parquet file
            df = pd.read_parquet(input_path)
            compressed = self.compress_dataframe(df, profile, format="parquet")
            output_path.write_bytes(compressed)
        else:
            # General file compression
            data = input_path.read_bytes()
            
            if profile.compression_type == CompressionType.GZIP:
                compressed = gzip.compress(data, compresslevel=profile.compression_level or 6)
            elif profile.compression_type == CompressionType.BZ2:
                compressed = bz2.compress(data, compresslevel=profile.compression_level or 9)
            elif profile.compression_type == CompressionType.LZMA:
                compressed = lzma.compress(data, preset=profile.compression_level or 6)
            else:
                compressed = data
            
            output_path.write_bytes(compressed)
        
        output_size = output_path.stat().st_size
        
        return {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "input_size_mb": input_size / (1024 * 1024),
            "output_size_mb": output_size / (1024 * 1024),
            "compression_ratio": input_size / output_size if output_size > 0 else 0,
            "space_savings_pct": (1 - output_size / input_size) * 100 if input_size > 0 else 0
        }
    
    def decompress_file(
        self,
        input_path: Path,
        output_path: Path,
        compression_type: Optional[CompressionType] = None
    ) -> bool:
        """Decompress a file."""
        try:
            # Auto-detect compression if not specified
            if compression_type is None:
                if input_path.suffix == '.gz':
                    compression_type = CompressionType.GZIP
                elif input_path.suffix == '.bz2':
                    compression_type = CompressionType.BZ2
                elif input_path.suffix == '.xz':
                    compression_type = CompressionType.LZMA
                else:
                    # Try to detect from content
                    compression_type = self._detect_compression(input_path)
            
            compressed_data = input_path.read_bytes()
            
            if compression_type == CompressionType.GZIP:
                decompressed = gzip.decompress(compressed_data)
            elif compression_type == CompressionType.BZ2:
                decompressed = bz2.decompress(compressed_data)
            elif compression_type == CompressionType.LZMA:
                decompressed = lzma.decompress(compressed_data)
            else:
                decompressed = compressed_data
            
            output_path.write_bytes(decompressed)
            return True
            
        except Exception as e:
            logger.error(f"Failed to decompress {input_path}: {e}")
            return False
    
    def _detect_compression(self, file_path: Path) -> CompressionType:
        """Detect compression type from file content."""
        with open(file_path, 'rb') as f:
            header = f.read(10)
        
        # Check magic numbers
        if header.startswith(b'\x1f\x8b'):  # GZIP
            return CompressionType.GZIP
        elif header.startswith(b'BZh'):      # BZ2
            return CompressionType.BZ2
        elif header.startswith(b'\xfd7zXZ'): # LZMA/XZ
            return CompressionType.LZMA
        else:
            return CompressionType.NONE
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            "total_compressions": len(self.compression_stats),
            "by_type": self._aggregate_stats_by_type(),
            "average_ratio": self._calculate_average_ratio(),
            "total_space_saved_mb": self._calculate_total_savings()
        }
    
    def _aggregate_stats_by_type(self) -> Dict[str, int]:
        """Aggregate statistics by compression type."""
        by_type = {}
        for stats in self.compression_stats.values():
            comp_type = stats.get("compression_type", "unknown")
            by_type[comp_type] = by_type.get(comp_type, 0) + 1
        return by_type
    
    def _calculate_average_ratio(self) -> float:
        """Calculate average compression ratio."""
        ratios = [
            stats.get("compression_ratio", 1.0)
            for stats in self.compression_stats.values()
        ]
        return sum(ratios) / len(ratios) if ratios else 1.0
    
    def _calculate_total_savings(self) -> float:
        """Calculate total space saved in MB."""
        total_saved = 0
        for stats in self.compression_stats.values():
            original = stats.get("original_size_mb", 0)
            compressed = stats.get("compressed_size_mb", 0)
            total_saved += (original - compressed)
        return total_saved