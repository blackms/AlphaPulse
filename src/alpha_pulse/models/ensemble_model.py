"""Data models for ensemble methods."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import Column, String, Float, DateTime, Integer, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field, ConfigDict

Base = declarative_base()


# SQLAlchemy Models

class EnsembleConfig(Base):
    """Ensemble configuration database model."""
    __tablename__ = 'ensemble_configs'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    ensemble_type = Column(String, nullable=False)  # voting, stacking, boosting
    config = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    predictions = relationship("EnsemblePrediction", back_populates="ensemble_config")
    performance_metrics = relationship("EnsemblePerformance", back_populates="ensemble_config")


class TradingAgent(Base):
    """Trading agent database model."""
    __tablename__ = 'trading_agents'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    agent_type = Column(String, nullable=False)  # technical, fundamental, sentiment, ml
    config = Column(JSON, nullable=False)
    performance_score = Column(Float, default=0.0)
    status = Column(String, default='inactive')  # active, inactive, testing
    created_at = Column(DateTime, default=datetime.utcnow)
    last_signal_at = Column(DateTime)
    signal_count = Column(Integer, default=0)
    
    # Relationships
    signals = relationship("AgentSignalRecord", back_populates="agent")
    weights = relationship("AgentWeight", back_populates="agent")


class AgentSignalRecord(Base):
    """Agent signal history database model."""
    __tablename__ = 'agent_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String, ForeignKey('trading_agents.id'))
    timestamp = Column(DateTime, nullable=False)
    signal = Column(Float, nullable=False)  # -1 to 1
    confidence = Column(Float, nullable=False)  # 0 to 1
    metadata = Column(JSON)
    market_conditions = Column(JSON)
    
    # Relationships
    agent = relationship("TradingAgent", back_populates="signals")
    ensemble_predictions = relationship("EnsemblePrediction", secondary="ensemble_signal_mapping")


class EnsemblePrediction(Base):
    """Ensemble prediction database model."""
    __tablename__ = 'ensemble_predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ensemble_id = Column(String, ForeignKey('ensemble_configs.id'))
    timestamp = Column(DateTime, nullable=False)
    signal = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    contributing_agents = Column(JSON)  # List of agent IDs
    weights = Column(JSON)  # Dict of agent_id: weight
    metadata = Column(JSON)
    execution_time_ms = Column(Float)
    
    # Relationships
    ensemble_config = relationship("EnsembleConfig", back_populates="predictions")
    signals = relationship("AgentSignalRecord", secondary="ensemble_signal_mapping")


class EnsembleSignalMapping(Base):
    """Many-to-many mapping between ensemble predictions and agent signals."""
    __tablename__ = 'ensemble_signal_mapping'
    
    ensemble_prediction_id = Column(Integer, ForeignKey('ensemble_predictions.id'), primary_key=True)
    agent_signal_id = Column(Integer, ForeignKey('agent_signals.id'), primary_key=True)


class AgentWeight(Base):
    """Agent weight history database model."""
    __tablename__ = 'agent_weights'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String, ForeignKey('trading_agents.id'))
    ensemble_id = Column(String, ForeignKey('ensemble_configs.id'))
    weight = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    reason = Column(String)  # performance_update, manual_adjustment, rebalance
    
    # Relationships
    agent = relationship("TradingAgent", back_populates="weights")


class EnsemblePerformance(Base):
    """Ensemble performance metrics database model."""
    __tablename__ = 'ensemble_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ensemble_id = Column(String, ForeignKey('ensemble_configs.id'))
    timestamp = Column(DateTime, nullable=False)
    accuracy = Column(Float)
    sharpe_ratio = Column(Float)
    hit_rate = Column(Float)
    avg_confidence = Column(Float)
    signal_count = Column(Integer)
    profit_loss = Column(Float)
    max_drawdown = Column(Float)
    metrics = Column(JSON)  # Additional metrics
    
    # Relationships
    ensemble_config = relationship("EnsembleConfig", back_populates="performance_metrics")


# Pydantic Models for API

class AgentSignalCreate(BaseModel):
    """Model for creating agent signals."""
    agent_id: str
    signal: float = Field(..., ge=-1, le=1)
    confidence: float = Field(..., ge=0, le=1)
    metadata: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class AgentSignalResponse(AgentSignalCreate):
    """Model for agent signal responses."""
    timestamp: datetime
    id: Optional[int] = None


class EnsembleConfigCreate(BaseModel):
    """Model for creating ensemble configurations."""
    name: str
    ensemble_type: str = Field(..., pattern="^(voting|stacking|boosting)$")
    config: Dict[str, Any]
    
    model_config = ConfigDict(from_attributes=True)


class EnsembleConfigResponse(EnsembleConfigCreate):
    """Model for ensemble config responses."""
    id: str
    created_at: datetime
    updated_at: datetime
    is_active: bool


class EnsemblePredictionCreate(BaseModel):
    """Model for creating ensemble predictions."""
    ensemble_id: str
    agent_signals: List[AgentSignalCreate]
    
    model_config = ConfigDict(from_attributes=True)


class EnsemblePredictionResponse(BaseModel):
    """Model for ensemble prediction responses."""
    id: int
    ensemble_id: str
    timestamp: datetime
    signal: float
    confidence: float
    contributing_agents: List[str]
    weights: Dict[str, float]
    metadata: Dict[str, Any]
    execution_time_ms: Optional[float] = None
    
    model_config = ConfigDict(from_attributes=True)


class AgentPerformance(BaseModel):
    """Model for agent performance metrics."""
    agent_id: str
    performance_score: float
    signal_count: int
    last_signal_at: Optional[datetime] = None
    status: str
    recent_accuracy: Optional[float] = None
    recent_sharpe: Optional[float] = None
    
    model_config = ConfigDict(from_attributes=True)


class EnsemblePerformanceMetrics(BaseModel):
    """Model for ensemble performance metrics."""
    ensemble_id: str
    timestamp: datetime
    accuracy: float
    sharpe_ratio: float
    hit_rate: float
    avg_confidence: float
    signal_count: int
    profit_loss: float
    max_drawdown: float
    additional_metrics: Optional[Dict[str, float]] = None
    
    model_config = ConfigDict(from_attributes=True)


class WeightUpdate(BaseModel):
    """Model for weight updates."""
    agent_id: str
    weight: float = Field(..., ge=0)
    reason: str
    
    model_config = ConfigDict(from_attributes=True)


class EnsembleWeightUpdate(BaseModel):
    """Model for batch weight updates."""
    ensemble_id: str
    weight_updates: List[WeightUpdate]
    rebalance: bool = False
    
    model_config = ConfigDict(from_attributes=True)


class SignalAggregationRequest(BaseModel):
    """Model for signal aggregation requests."""
    signals: List[AgentSignalCreate]
    method: str = "weighted_average"
    config: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class SignalAggregationResponse(BaseModel):
    """Model for signal aggregation responses."""
    signal: float
    confidence: float
    diversity: float
    method: str
    signal_count: int
    metadata: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class EnsembleOptimizationRequest(BaseModel):
    """Model for ensemble optimization requests."""
    ensemble_id: str
    lookback_period: int = 100
    optimization_method: str = "sharpe"
    constraints: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class EnsembleOptimizationResponse(BaseModel):
    """Model for ensemble optimization responses."""
    ensemble_id: str
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    expected_improvement: float
    optimization_metrics: Dict[str, float]
    
    model_config = ConfigDict(from_attributes=True)


class AgentRegistration(BaseModel):
    """Model for agent registration."""
    name: str
    agent_type: str
    config: Dict[str, Any]
    initial_weight: float = 1.0
    
    model_config = ConfigDict(from_attributes=True)


class AgentRegistrationResponse(BaseModel):
    """Model for agent registration response."""
    agent_id: str
    name: str
    agent_type: str
    status: str
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class EnsembleDiversityMetrics(BaseModel):
    """Model for ensemble diversity metrics."""
    ensemble_id: str
    timestamp: datetime
    signal_diversity: float
    agent_correlation_matrix: Dict[str, Dict[str, float]]
    effective_agents: int
    diversity_score: float
    
    model_config = ConfigDict(from_attributes=True)


class BacktestRequest(BaseModel):
    """Model for backtesting requests."""
    ensemble_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000
    position_sizing: str = "fixed"
    risk_limits: Optional[Dict[str, float]] = None
    
    model_config = ConfigDict(from_attributes=True)


class BacktestResult(BaseModel):
    """Model for backtest results."""
    ensemble_id: str
    period: Dict[str, datetime]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    metrics: Dict[str, float]
    equity_curve: List[float]
    
    model_config = ConfigDict(from_attributes=True)