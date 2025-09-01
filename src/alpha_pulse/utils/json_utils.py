"""JSON utility functions for safe serialization and deserialization."""
import json
import numpy as np
from typing import Any, Dict, List, Union
from datetime import datetime, date


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def safe_json_dumps(data: Any, **kwargs) -> str:
    """Safely serialize data to JSON string."""
    try:
        converted_data = convert_numpy_types(data)
        return json.dumps(converted_data, **kwargs)
    except (TypeError, ValueError) as e:
        # Fallback to string representation
        return json.dumps({"error": str(e), "data": str(data)})


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely deserialize JSON string to Python object."""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else {}