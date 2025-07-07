"""
Monitoring interfaces for AlphaPulse.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional


class MetricType(Enum):
    """Types of metrics to monitor."""
    SYSTEM_HEALTH = "system_health"
    PORTFOLIO_VALUE = "portfolio_value"
    RISK_METRICS = "risk_metrics"
    TRADE_PERFORMANCE = "trade_performance"
    API_LATENCY = "api_latency"
    ERROR_RATE = "error_rate"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthStatus:
    """System health status."""
    status: str
    uptime: int
    last_check: datetime
    components: Dict[str, str]


@dataclass
class Alert:
    """Alert notification."""
    level: AlertLevel
    metric_type: MetricType
    message: str
    timestamp: datetime
    value: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class IMonitor(ABC):
    """Base interface for monitoring components."""
    
    @abstractmethod
    async def collect_metrics(self) -> Dict[str, float]:
        """Collect current metrics."""
        pass
    
    @abstractmethod
    async def check_thresholds(self, metrics: Dict[str, float]) -> List[Alert]:
        """Check if any metrics exceed thresholds."""
        pass
    
    @abstractmethod
    async def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        pass