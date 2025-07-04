"""Data models for online learning system."""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import numpy as np

Base = declarative_base()


class OnlineLearningSession(Base):
    """SQLAlchemy model for online learning sessions."""
    __tablename__ = 'online_learning_sessions'
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    agent_id = Column(String, index=True)
    strategy = Column(String)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    n_samples_processed = Column(Integer, default=0)
    n_updates = Column(Integer, default=0)
    n_drifts_detected = Column(Integer, default=0)
    final_performance = Column(Float, nullable=True)
    metadata = Column(JSON)
    
    __table_args__ = (
        Index('idx_session_agent_time', 'agent_id', 'start_time'),
    )


class DriftEvent(Base):
    """SQLAlchemy model for concept drift events."""
    __tablename__ = 'drift_events'
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    sample_index = Column(Integer)
    drift_type = Column(String)  # 'sudden', 'gradual', 'incremental', 'recurring'
    drift_level = Column(Float)
    confidence = Column(Float)
    detector_method = Column(String)
    action_taken = Column(String)
    metadata = Column(JSON)
    
    __table_args__ = (
        Index('idx_drift_session_time', 'session_id', 'timestamp'),
    )


class ModelCheckpoint(Base):
    """SQLAlchemy model for model checkpoints."""
    __tablename__ = 'model_checkpoints'
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    checkpoint_time = Column(DateTime, default=datetime.utcnow)
    model_version = Column(Integer)
    performance_metrics = Column(JSON)
    model_path = Column(String)
    is_best = Column(Boolean, default=False)
    metadata = Column(JSON)
    
    __table_args__ = (
        Index('idx_checkpoint_session_version', 'session_id', 'model_version'),
    )


class StreamingMetrics(Base):
    """SQLAlchemy model for streaming performance metrics."""
    __tablename__ = 'streaming_metrics'
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_type = Column(String)  # 'accuracy', 'loss', 'latency', 'memory'
    value = Column(Float)
    window_size = Column(Integer)
    metadata = Column(JSON)
    
    __table_args__ = (
        Index('idx_metrics_session_type', 'session_id', 'metric_type'),
        Index('idx_metrics_timestamp', 'timestamp'),
    )


# Pydantic Models for API

class OnlineDataPointModel(BaseModel):
    """API model for online data points."""
    timestamp: datetime
    features: List[float]
    label: Optional[float] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StreamingBatch(BaseModel):
    """API model for streaming data batches."""
    batch_id: str
    data_points: List[OnlineDataPointModel]
    source: str
    priority: int = 1
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LearningSessionRequest(BaseModel):
    """API model for starting a learning session."""
    agent_id: str
    strategy: str = "adaptive"
    config: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "agent_id": "tech_agent_001",
                "strategy": "adaptive",
                "config": {
                    "learning_rate": 0.01,
                    "batch_size": 32,
                    "drift_detection": {
                        "method": "adwin",
                        "sensitivity": 0.002
                    }
                }
            }
        }


class LearningSessionResponse(BaseModel):
    """API model for learning session response."""
    session_id: str
    status: str
    start_time: datetime
    config: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PredictionRequest(BaseModel):
    """API model for prediction requests."""
    session_id: str
    features: List[List[float]]
    return_probabilities: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_123",
                "features": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "return_probabilities": False
            }
        }


class PredictionResponse(BaseModel):
    """API model for prediction response."""
    predictions: List[float]
    probabilities: Optional[List[List[float]]] = None
    model_version: int
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DriftDetectionAlert(BaseModel):
    """API model for drift detection alerts."""
    session_id: str
    drift_type: str
    drift_level: float
    confidence: float
    timestamp: datetime
    recommended_action: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LearningMetrics(BaseModel):
    """API model for learning metrics."""
    session_id: str
    timestamp: datetime
    n_samples_seen: int
    n_updates: int
    current_accuracy: float
    memory_usage_mb: float
    processing_rate_hz: float
    drift_detected: bool
    model_version: int
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelUpdateNotification(BaseModel):
    """API model for model update notifications."""
    session_id: str
    update_type: str  # 'incremental', 'drift_adaptation', 'ensemble_update'
    timestamp: datetime
    old_version: int
    new_version: int
    performance_improvement: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StreamingValidationResult(BaseModel):
    """API model for streaming validation results."""
    session_id: str
    validation_window: int
    metrics: Dict[str, float]  # accuracy, precision, recall, f1, etc.
    timestamp: datetime
    is_stable: bool
    warnings: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AdaptiveStrategyUpdate(BaseModel):
    """API model for adaptive strategy updates."""
    session_id: str
    old_strategy: str
    new_strategy: str
    reason: str
    market_conditions: Dict[str, float]
    expected_improvement: float
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryManagementStatus(BaseModel):
    """API model for memory management status."""
    session_id: str
    managed_bytes: int
    managed_items: int
    utilization: float
    eviction_policy: str
    last_eviction: Optional[datetime] = None
    gc_count: int
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OnlineLearningConfig(BaseModel):
    """Configuration for online learning system."""
    learning_rate: float = 0.01
    batch_size: int = 32
    update_frequency: int = 1
    performance_window_size: int = 100
    
    drift_detection: Dict[str, Any] = Field(default_factory=lambda: {
        "method": "adwin",
        "check_frequency": 100,
        "ensemble_mode": False
    })
    
    memory_management: Dict[str, Any] = Field(default_factory=lambda: {
        "max_memory_mb": 1024,
        "eviction_policy": "adaptive",
        "gc_threshold": 0.9
    })
    
    adaptive_optimization: Dict[str, Any] = Field(default_factory=lambda: {
        "optimizer_type": "adam",
        "adapt_betas": True,
        "gradient_clipping": 1.0
    })
    
    ensemble_config: Dict[str, Any] = Field(default_factory=lambda: {
        "max_models": 5,
        "combination_method": "weighted_average",
        "diversity_threshold": 0.1
    })
    
    class Config:
        schema_extra = {
            "example": {
                "learning_rate": 0.01,
                "batch_size": 32,
                "drift_detection": {
                    "method": "ensemble",
                    "ensemble_mode": True
                }
            }
        }


class StreamingDataStats(BaseModel):
    """Statistics about streaming data."""
    total_samples: int
    samples_per_second: float
    feature_means: List[float]
    feature_stds: List[float]
    label_distribution: Dict[str, float]
    missing_rate: float
    anomaly_rate: float
    
    
class OnlineLearningStatus(BaseModel):
    """Overall status of online learning system."""
    active_sessions: int
    total_samples_processed: int
    total_drifts_detected: int
    average_accuracy: float
    system_memory_usage_mb: float
    system_cpu_usage_percent: float
    uptime_seconds: float
    warnings: List[str] = Field(default_factory=list)
    
    
# Helper functions for data conversion

def numpy_to_api(arr: np.ndarray) -> List[Union[float, List[float]]]:
    """Convert numpy array to API-compatible format."""
    if arr.ndim == 1:
        return arr.tolist()
    else:
        return [row.tolist() for row in arr]


def api_to_numpy(data: List[Union[float, List[float]]]) -> np.ndarray:
    """Convert API data to numpy array."""
    return np.array(data)


def create_data_point(features: np.ndarray, 
                     label: Optional[float] = None,
                     weight: float = 1.0,
                     metadata: Optional[Dict[str, Any]] = None) -> OnlineDataPointModel:
    """Create an online data point from numpy array."""
    return OnlineDataPointModel(
        timestamp=datetime.utcnow(),
        features=features.tolist(),
        label=label,
        weight=weight,
        metadata=metadata or {}
    )