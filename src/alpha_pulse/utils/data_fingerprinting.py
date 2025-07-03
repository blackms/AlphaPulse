"""
Data fingerprinting utilities for content-based versioning and change detection.

Provides:
- Multiple fingerprinting algorithms
- Change detection between datasets
- Statistical fingerprints for numerical data
- Schema fingerprinting
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger


@dataclass
class FingerprintResult:
    """Result of fingerprinting operation."""
    fingerprint: str
    algorithm: str
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fingerprint": self.fingerprint,
            "algorithm": self.algorithm,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class DataFingerprinter:
    """Comprehensive data fingerprinting system."""
    
    def __init__(self):
        self.supported_algorithms = {
            "sha256": self._sha256_fingerprint,
            "md5": self._md5_fingerprint,
            "statistical": self._statistical_fingerprint,
            "schema": self._schema_fingerprint,
            "sample": self._sample_fingerprint,
            "composite": self._composite_fingerprint
        }
    
    async def calculate_fingerprint(
        self,
        data: Any,
        algorithm: str = "sha256",
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Calculate fingerprint for data.
        
        Args:
            data: Data to fingerprint
            algorithm: Fingerprinting algorithm to use
            options: Algorithm-specific options
        
        Returns:
            Fingerprint string
        """
        if algorithm not in self.supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        try:
            fingerprint_func = self.supported_algorithms[algorithm]
            return fingerprint_func(data, options or {})
        except Exception as e:
            logger.error(f"Fingerprinting failed: {e}")
            raise
    
    def _sha256_fingerprint(self, data: Any, options: Dict[str, Any]) -> str:
        """Calculate SHA-256 fingerprint."""
        # Convert data to bytes
        data_bytes = self._serialize_data(data)
        
        # Calculate hash
        hasher = hashlib.sha256()
        hasher.update(data_bytes)
        
        return hasher.hexdigest()
    
    def _md5_fingerprint(self, data: Any, options: Dict[str, Any]) -> str:
        """Calculate MD5 fingerprint (faster but less secure)."""
        data_bytes = self._serialize_data(data)
        
        hasher = hashlib.md5()
        hasher.update(data_bytes)
        
        return hasher.hexdigest()
    
    def _statistical_fingerprint(self, data: Any, options: Dict[str, Any]) -> str:
        """Calculate statistical fingerprint for numerical data."""
        stats = {}
        
        if isinstance(data, pd.DataFrame):
            # DataFrame statistics
            stats["shape"] = data.shape
            stats["dtypes"] = data.dtypes.to_dict()
            
            # Numerical column statistics
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_stats = {
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "q25": float(data[col].quantile(0.25)),
                    "q50": float(data[col].quantile(0.50)),
                    "q75": float(data[col].quantile(0.75)),
                    "null_count": int(data[col].isnull().sum())
                }
                stats[f"col_{col}"] = col_stats
        
        elif isinstance(data, (list, np.ndarray)):
            # Array statistics
            arr = np.array(data)
            if arr.dtype.kind in 'biufc':  # Numeric types
                stats = {
                    "shape": arr.shape,
                    "dtype": str(arr.dtype),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "sum": float(np.sum(arr))
                }
        
        # Convert stats to fingerprint
        stats_str = json.dumps(stats, sort_keys=True)
        return hashlib.sha256(stats_str.encode()).hexdigest()[:16]
    
    def _schema_fingerprint(self, data: Any, options: Dict[str, Any]) -> str:
        """Calculate schema fingerprint."""
        schema = {}
        
        if isinstance(data, pd.DataFrame):
            # DataFrame schema
            schema["columns"] = list(data.columns)
            schema["dtypes"] = {col: str(dtype) for col, dtype in data.dtypes.items()}
            schema["index_type"] = str(type(data.index))
            
        elif isinstance(data, dict):
            # Dictionary schema
            schema["keys"] = sorted(data.keys())
            schema["types"] = {k: type(v).__name__ for k, v in data.items()}
            
        elif isinstance(data, list) and data:
            # List schema (based on first element)
            schema["length"] = len(data)
            schema["element_type"] = type(data[0]).__name__
            
            if isinstance(data[0], dict):
                schema["element_keys"] = sorted(data[0].keys())
        
        # Convert schema to fingerprint
        schema_str = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
    
    def _sample_fingerprint(self, data: Any, options: Dict[str, Any]) -> str:
        """Calculate fingerprint based on data sample."""
        sample_size = options.get("sample_size", 1000)
        seed = options.get("seed", 42)
        
        if isinstance(data, pd.DataFrame):
            # Sample rows
            n_rows = min(sample_size, len(data))
            if n_rows < len(data):
                sample = data.sample(n=n_rows, random_state=seed)
            else:
                sample = data
            
            # Use first/last rows and sample
            parts = [
                data.head(10).to_string(),
                data.tail(10).to_string(),
                sample.to_string()
            ]
            sample_str = "\n".join(parts)
            
        elif isinstance(data, (list, np.ndarray)):
            # Sample elements
            arr = np.array(data)
            if len(arr) > sample_size:
                np.random.seed(seed)
                indices = np.random.choice(len(arr), sample_size, replace=False)
                sample = arr[indices]
            else:
                sample = arr
            
            sample_str = str(sample)
        
        else:
            sample_str = str(data)[:sample_size]
        
        return hashlib.sha256(sample_str.encode()).hexdigest()[:16]
    
    def _composite_fingerprint(self, data: Any, options: Dict[str, Any]) -> str:
        """Calculate composite fingerprint using multiple methods."""
        components = []
        
        # Content hash
        components.append(self._sha256_fingerprint(data, {})[:8])
        
        # Schema hash
        components.append(self._schema_fingerprint(data, {})[:8])
        
        # Statistical hash (if applicable)
        try:
            components.append(self._statistical_fingerprint(data, {})[:8])
        except:
            components.append("00000000")
        
        # Sample hash
        components.append(self._sample_fingerprint(data, options)[:8])
        
        return "-".join(components)
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data to bytes for hashing."""
        if isinstance(data, bytes):
            return data
        
        elif isinstance(data, str):
            return data.encode('utf-8')
        
        elif isinstance(data, pd.DataFrame):
            # Serialize DataFrame consistently
            return data.to_csv(index=False).encode('utf-8')
        
        elif isinstance(data, np.ndarray):
            # Serialize numpy array
            return data.tobytes()
        
        elif isinstance(data, (list, dict)):
            # Serialize as JSON
            return json.dumps(data, sort_keys=True, default=str).encode('utf-8')
        
        else:
            # Fallback to string representation
            return str(data).encode('utf-8')
    
    async def calculate_content_diff(
        self,
        data1: Any,
        data2: Any
    ) -> Dict[str, Any]:
        """
        Calculate content differences between two datasets.
        
        Args:
            data1: First dataset
            data2: Second dataset
        
        Returns:
            Dictionary describing differences
        """
        diff = {
            "has_changes": False,
            "change_type": "none",
            "details": {}
        }
        
        # Check if same type
        if type(data1) != type(data2):
            diff["has_changes"] = True
            diff["change_type"] = "type_change"
            diff["details"]["old_type"] = type(data1).__name__
            diff["details"]["new_type"] = type(data2).__name__
            return diff
        
        if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
            diff = self._compare_dataframes(data1, data2)
        
        elif isinstance(data1, dict) and isinstance(data2, dict):
            diff = self._compare_dicts(data1, data2)
        
        elif isinstance(data1, (list, np.ndarray)) and isinstance(data2, (list, np.ndarray)):
            diff = self._compare_arrays(data1, data2)
        
        else:
            # Simple comparison
            if data1 != data2:
                diff["has_changes"] = True
                diff["change_type"] = "value_change"
        
        return diff
    
    def _compare_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        """Compare two DataFrames."""
        diff = {
            "has_changes": False,
            "change_type": "none",
            "details": {}
        }
        
        # Shape comparison
        if df1.shape != df2.shape:
            diff["has_changes"] = True
            diff["change_type"] = "shape_change"
            diff["details"]["old_shape"] = df1.shape
            diff["details"]["new_shape"] = df2.shape
        
        # Column comparison
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        
        if cols1 != cols2:
            diff["has_changes"] = True
            diff["change_type"] = "schema_change"
            diff["details"]["added_columns"] = list(cols2 - cols1)
            diff["details"]["removed_columns"] = list(cols1 - cols2)
        
        # Data comparison for common columns
        common_cols = cols1 & cols2
        if common_cols and df1.shape[0] == df2.shape[0]:
            for col in common_cols:
                if not df1[col].equals(df2[col]):
                    diff["has_changes"] = True
                    if diff["change_type"] == "none":
                        diff["change_type"] = "content_change"
                    
                    # Sample differences
                    mask = df1[col] != df2[col]
                    n_changes = mask.sum()
                    
                    if "column_changes" not in diff["details"]:
                        diff["details"]["column_changes"] = {}
                    
                    diff["details"]["column_changes"][col] = {
                        "changes": int(n_changes),
                        "change_ratio": float(n_changes / len(df1))
                    }
        
        return diff
    
    def _compare_dicts(self, dict1: dict, dict2: dict) -> Dict[str, Any]:
        """Compare two dictionaries."""
        diff = {
            "has_changes": False,
            "change_type": "none",
            "details": {}
        }
        
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        
        # Key differences
        if keys1 != keys2:
            diff["has_changes"] = True
            diff["change_type"] = "key_change"
            diff["details"]["added_keys"] = list(keys2 - keys1)
            diff["details"]["removed_keys"] = list(keys1 - keys2)
        
        # Value differences
        common_keys = keys1 & keys2
        changed_values = []
        
        for key in common_keys:
            if dict1[key] != dict2[key]:
                changed_values.append(key)
        
        if changed_values:
            diff["has_changes"] = True
            if diff["change_type"] == "none":
                diff["change_type"] = "value_change"
            diff["details"]["changed_values"] = changed_values
        
        return diff
    
    def _compare_arrays(
        self,
        arr1: Union[list, np.ndarray],
        arr2: Union[list, np.ndarray]
    ) -> Dict[str, Any]:
        """Compare two arrays."""
        diff = {
            "has_changes": False,
            "change_type": "none",
            "details": {}
        }
        
        a1 = np.array(arr1)
        a2 = np.array(arr2)
        
        # Shape comparison
        if a1.shape != a2.shape:
            diff["has_changes"] = True
            diff["change_type"] = "shape_change"
            diff["details"]["old_shape"] = a1.shape
            diff["details"]["new_shape"] = a2.shape
            return diff
        
        # Content comparison
        if not np.array_equal(a1, a2):
            diff["has_changes"] = True
            diff["change_type"] = "content_change"
            
            # Calculate change statistics
            if a1.dtype.kind in 'biufc' and a2.dtype.kind in 'biufc':
                changes = a1 != a2
                diff["details"]["change_count"] = int(np.sum(changes))
                diff["details"]["change_ratio"] = float(np.mean(changes))
                
                # Statistical changes
                if a1.size > 0:
                    diff["details"]["mean_change"] = float(np.mean(a2) - np.mean(a1))
                    diff["details"]["std_change"] = float(np.std(a2) - np.std(a1))
        
        return diff
    
    def calculate_similarity_score(
        self,
        fingerprint1: str,
        fingerprint2: str,
        algorithm: str = "sha256"
    ) -> float:
        """
        Calculate similarity score between two fingerprints.
        
        Args:
            fingerprint1: First fingerprint
            fingerprint2: Second fingerprint
            algorithm: Algorithm used for fingerprints
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if fingerprint1 == fingerprint2:
            return 1.0
        
        # For hash-based fingerprints, use Hamming distance
        if algorithm in ["sha256", "md5"]:
            # Convert to binary
            bin1 = bin(int(fingerprint1[:16], 16))[2:].zfill(64)
            bin2 = bin(int(fingerprint2[:16], 16))[2:].zfill(64)
            
            # Calculate Hamming distance
            distance = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
            similarity = 1.0 - (distance / len(bin1))
            
            return similarity
        
        # For composite fingerprints
        elif algorithm == "composite" and "-" in fingerprint1 and "-" in fingerprint2:
            parts1 = fingerprint1.split("-")
            parts2 = fingerprint2.split("-")
            
            if len(parts1) == len(parts2):
                # Average similarity of components
                similarities = []
                for p1, p2 in zip(parts1, parts2):
                    if p1 == p2:
                        similarities.append(1.0)
                    else:
                        similarities.append(0.0)
                
                return sum(similarities) / len(similarities)
        
        # Default: different fingerprints have 0 similarity
        return 0.0