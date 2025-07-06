"""Optimized serialization utilities for caching."""

import gzip
import json
import logging
import lz4.frame
import pickle
import snappy
from dataclasses import asdict, is_dataclass
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Type
from uuid import UUID

import msgpack
import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class SerializationError(Exception):
    """Error during serialization/deserialization."""
    pass


class CompressionType(Enum):
    """Compression types for serialized data."""
    
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    SNAPPY = "snappy"


class SerializationFormat(Enum):
    """Serialization formats."""
    
    JSON = "json"
    MSGPACK = "msgpack"
    PICKLE = "pickle"
    PROTOBUF = "protobuf"  # Placeholder for future implementation


class SerializerConfig:
    """Configuration for serialization."""
    
    def __init__(
        self,
        format: SerializationFormat = SerializationFormat.MSGPACK,
        compression: CompressionType = CompressionType.NONE,
        compression_threshold: int = 1024,  # bytes
        compression_level: int = 6,  # 1-9 for gzip
        max_size: Optional[int] = None,  # Maximum serialized size
        numpy_optimize: bool = True,
        pandas_optimize: bool = True
    ):
        """Initialize serializer configuration."""
        self.format = format
        self.compression = compression
        self.compression_threshold = compression_threshold
        self.compression_level = compression_level
        self.max_size = max_size
        self.numpy_optimize = numpy_optimize
        self.pandas_optimize = pandas_optimize


class OptimizedSerializer:
    """Optimized serializer for caching with multiple format support."""
    
    def __init__(self, config: Optional[SerializerConfig] = None):
        """Initialize serializer."""
        self.config = config or SerializerConfig()
        self._custom_encoders = {}
        self._custom_decoders = {}
        
        # Register default encoders
        self._register_default_encoders()
    
    def _register_default_encoders(self):
        """Register default type encoders."""
        # Datetime types
        self.register_encoder(datetime, lambda x: {"__datetime__": x.isoformat()})
        self.register_encoder(date, lambda x: {"__date__": x.isoformat()})
        self.register_encoder(time, lambda x: {"__time__": x.isoformat()})
        self.register_encoder(timedelta, lambda x: {"__timedelta__": x.total_seconds()})
        
        # Decimal
        self.register_encoder(Decimal, lambda x: {"__decimal__": str(x)})
        
        # UUID
        self.register_encoder(UUID, lambda x: {"__uuid__": str(x)})
        
        # Numpy arrays
        if self.config.numpy_optimize:
            self.register_encoder(np.ndarray, self._encode_numpy_array)
        
        # Pandas DataFrames and Series
        if self.config.pandas_optimize:
            self.register_encoder(pd.DataFrame, self._encode_pandas_dataframe)
            self.register_encoder(pd.Series, self._encode_pandas_series)
    
    def register_encoder(self, type_: Type, encoder: callable):
        """Register custom encoder for a type."""
        self._custom_encoders[type_] = encoder
    
    def register_decoder(self, key: str, decoder: callable):
        """Register custom decoder for a key."""
        self._custom_decoders[key] = decoder
    
    def _encode_numpy_array(self, arr: np.ndarray) -> Dict[str, Any]:
        """Encode numpy array efficiently."""
        return {
            "__numpy_array__": {
                "data": arr.tobytes(),
                "dtype": str(arr.dtype),
                "shape": arr.shape
            }
        }
    
    def _decode_numpy_array(self, data: Dict[str, Any]) -> np.ndarray:
        """Decode numpy array."""
        arr_data = data["__numpy_array__"]
        return np.frombuffer(
            arr_data["data"],
            dtype=arr_data["dtype"]
        ).reshape(arr_data["shape"])
    
    def _encode_pandas_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Encode pandas DataFrame efficiently."""
        return {
            "__pandas_dataframe__": {
                "data": df.to_json(orient="split", date_format="iso"),
                "index_name": df.index.name,
                "columns_name": df.columns.name
            }
        }
    
    def _decode_pandas_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Decode pandas DataFrame."""
        df_data = data["__pandas_dataframe__"]
        df = pd.read_json(df_data["data"], orient="split")
        df.index.name = df_data.get("index_name")
        df.columns.name = df_data.get("columns_name")
        return df
    
    def _encode_pandas_series(self, series: pd.Series) -> Dict[str, Any]:
        """Encode pandas Series efficiently."""
        return {
            "__pandas_series__": {
                "data": series.to_json(orient="split", date_format="iso"),
                "name": series.name
            }
        }
    
    def _decode_pandas_series(self, data: Dict[str, Any]) -> pd.Series:
        """Decode pandas Series."""
        series_data = data["__pandas_series__"]
        series = pd.read_json(series_data["data"], typ="series", orient="split")
        series.name = series_data.get("name")
        return series
    
    def _default_encoder(self, obj: Any) -> Any:
        """Default encoder for custom types."""
        # Check registered encoders
        for type_, encoder in self._custom_encoders.items():
            if isinstance(obj, type_):
                return encoder(obj)
        
        # Handle dataclasses
        if is_dataclass(obj):
            return {"__dataclass__": {
                "type": type(obj).__name__,
                "data": asdict(obj)
            }}
        
        # Handle Enums
        if isinstance(obj, Enum):
            return {"__enum__": {
                "type": type(obj).__name__,
                "value": obj.value
            }}
        
        # Handle sets
        if isinstance(obj, set):
            return {"__set__": list(obj)}
        
        # Handle bytes
        if isinstance(obj, bytes):
            return {"__bytes__": obj.hex()}
        
        # Fallback to string representation
        return {"__repr__": repr(obj)}
    
    def _object_hook(self, dct: Dict[str, Any]) -> Any:
        """Decode special objects."""
        # Datetime types
        if "__datetime__" in dct:
            return datetime.fromisoformat(dct["__datetime__"])
        elif "__date__" in dct:
            return date.fromisoformat(dct["__date__"])
        elif "__time__" in dct:
            return time.fromisoformat(dct["__time__"])
        elif "__timedelta__" in dct:
            return timedelta(seconds=dct["__timedelta__"])
        
        # Decimal
        elif "__decimal__" in dct:
            return Decimal(dct["__decimal__"])
        
        # UUID
        elif "__uuid__" in dct:
            return UUID(dct["__uuid__"])
        
        # Numpy array
        elif "__numpy_array__" in dct:
            return self._decode_numpy_array(dct)
        
        # Pandas DataFrame
        elif "__pandas_dataframe__" in dct:
            return self._decode_pandas_dataframe(dct)
        
        # Pandas Series
        elif "__pandas_series__" in dct:
            return self._decode_pandas_series(dct)
        
        # Set
        elif "__set__" in dct:
            return set(dct["__set__"])
        
        # Bytes
        elif "__bytes__" in dct:
            return bytes.fromhex(dct["__bytes__"])
        
        # Check custom decoders
        for key, decoder in self._custom_decoders.items():
            if key in dct:
                return decoder(dct)
        
        return dct
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize object to bytes."""
        try:
            # Choose serialization format
            if self.config.format == SerializationFormat.JSON:
                serialized = json.dumps(
                    obj,
                    default=self._default_encoder,
                    separators=(',', ':')
                ).encode('utf-8')
            
            elif self.config.format == SerializationFormat.MSGPACK:
                serialized = msgpack.packb(
                    obj,
                    default=self._default_encoder,
                    use_bin_type=True
                )
            
            elif self.config.format == SerializationFormat.PICKLE:
                serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            
            else:
                raise ValueError(f"Unsupported format: {self.config.format}")
            
            # Check size limit
            if self.config.max_size and len(serialized) > self.config.max_size:
                raise SerializationError(
                    f"Serialized size {len(serialized)} exceeds limit {self.config.max_size}"
                )
            
            # Apply compression if needed
            if (self.config.compression != CompressionType.NONE and
                len(serialized) >= self.config.compression_threshold):
                serialized = self._compress(serialized)
            
            return serialized
            
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise SerializationError(f"Failed to serialize object: {e}")
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to object."""
        try:
            # Check if data is compressed
            if self._is_compressed(data):
                data = self._decompress(data)
            
            # Choose deserialization format
            if self.config.format == SerializationFormat.JSON:
                return json.loads(
                    data.decode('utf-8'),
                    object_hook=self._object_hook
                )
            
            elif self.config.format == SerializationFormat.MSGPACK:
                return msgpack.unpackb(
                    data,
                    object_hook=self._object_hook,
                    raw=False
                )
            
            elif self.config.format == SerializationFormat.PICKLE:
                return pickle.loads(data)
            
            else:
                raise ValueError(f"Unsupported format: {self.config.format}")
                
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise SerializationError(f"Failed to deserialize data: {e}")
    
    def _compress(self, data: bytes) -> bytes:
        """Compress data."""
        if self.config.compression == CompressionType.GZIP:
            compressed = gzip.compress(data, compresslevel=self.config.compression_level)
            return b'GZ' + compressed
        
        elif self.config.compression == CompressionType.LZ4:
            compressed = lz4.frame.compress(data)
            return b'LZ' + compressed
        
        elif self.config.compression == CompressionType.SNAPPY:
            compressed = snappy.compress(data)
            return b'SN' + compressed
        
        return data
    
    def _decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        if data.startswith(b'GZ'):
            return gzip.decompress(data[2:])
        
        elif data.startswith(b'LZ'):
            return lz4.frame.decompress(data[2:])
        
        elif data.startswith(b'SN'):
            return snappy.decompress(data[2:])
        
        return data
    
    def _is_compressed(self, data: bytes) -> bool:
        """Check if data is compressed."""
        return data.startswith((b'GZ', b'LZ', b'SN'))
    
    def estimate_size(self, obj: Any) -> int:
        """Estimate serialized size of object."""
        try:
            return len(self.serialize(obj))
        except Exception:
            return 0


class SerializationBenchmark:
    """Benchmark different serialization methods."""
    
    @staticmethod
    def benchmark(obj: Any, iterations: int = 100) -> Dict[str, Dict[str, float]]:
        """Benchmark serialization methods."""
        import time
        
        results = {}
        
        # Test different formats
        for format_type in SerializationFormat:
            if format_type == SerializationFormat.PROTOBUF:
                continue  # Skip unimplemented
            
            # Test with and without compression
            for compression in [CompressionType.NONE, CompressionType.LZ4]:
                config = SerializerConfig(format=format_type, compression=compression)
                serializer = OptimizedSerializer(config)
                
                key = f"{format_type.value}_{compression.value}"
                
                # Serialize timing
                start = time.time()
                for _ in range(iterations):
                    data = serializer.serialize(obj)
                serialize_time = (time.time() - start) / iterations
                
                # Deserialize timing
                start = time.time()
                for _ in range(iterations):
                    _ = serializer.deserialize(data)
                deserialize_time = (time.time() - start) / iterations
                
                results[key] = {
                    "serialize_time": serialize_time,
                    "deserialize_time": deserialize_time,
                    "size": len(data),
                    "compression_ratio": len(serializer.serialize(obj)) / len(data) if compression != CompressionType.NONE else 1.0
                }
        
        return results


# Convenience functions
def create_serializer(
    format: str = "msgpack",
    compression: str = "none",
    compression_threshold: int = 1024
) -> OptimizedSerializer:
    """Create a serializer with string configuration."""
    config = SerializerConfig(
        format=SerializationFormat(format),
        compression=CompressionType(compression),
        compression_threshold=compression_threshold
    )
    return OptimizedSerializer(config)


def smart_serialize(obj: Any, size_threshold: int = 10000) -> bytes:
    """Smart serialization that chooses format based on object type and size."""
    # For small objects, use msgpack without compression
    if hasattr(obj, '__sizeof__') and obj.__sizeof__() < size_threshold:
        serializer = create_serializer("msgpack", "none")
    # For numpy/pandas, use msgpack with LZ4
    elif isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
        serializer = create_serializer("msgpack", "lz4", 512)
    # For large objects, use pickle with compression
    else:
        serializer = create_serializer("pickle", "lz4", 1024)
    
    return serializer.serialize(obj)