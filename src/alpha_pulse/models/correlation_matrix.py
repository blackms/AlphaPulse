"""
Data models for correlation analysis.

Defines structures for correlation matrices, regimes, and related metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import numpy as np


class CorrelationType(Enum):
    """Types of correlation matrices."""
    FULL = "full"
    ROLLING = "rolling"
    CONDITIONAL = "conditional"
    STRESSED = "stressed"
    SHRUNK = "shrunk"


class RegimeType(Enum):
    """Market regime types."""
    NORMAL = "normal"
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    CRISIS = "crisis"


@dataclass
class CorrelationMatrix:
    """Represents a correlation matrix with metadata."""
    matrix: np.ndarray
    assets: List[str]
    correlation_type: CorrelationType
    timestamp: datetime
    window_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate correlation matrix."""
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Correlation matrix must be square")
        if self.matrix.shape[0] != len(self.assets):
            raise ValueError("Matrix dimension must match number of assets")
        
        # Ensure diagonal is 1
        np.fill_diagonal(self.matrix, 1.0)
    
    def get_correlation(self, asset1: str, asset2: str) -> float:
        """Get correlation between two assets."""
        if asset1 not in self.assets or asset2 not in self.assets:
            raise ValueError(f"Asset not found in correlation matrix")
        
        idx1 = self.assets.index(asset1)
        idx2 = self.assets.index(asset2)
        
        return self.matrix[idx1, idx2]
    
    def get_average_correlation(self) -> float:
        """Calculate average pairwise correlation."""
        # Get upper triangle (excluding diagonal)
        upper_triangle = self.matrix[np.triu_indices_from(self.matrix, k=1)]
        return np.mean(upper_triangle)
    
    def get_max_correlation(self) -> Tuple[str, str, float]:
        """Find pair with maximum correlation."""
        # Mask diagonal
        masked = self.matrix.copy()
        np.fill_diagonal(masked, -np.inf)
        
        # Find max
        max_idx = np.unravel_index(np.argmax(masked), masked.shape)
        max_corr = masked[max_idx]
        
        return self.assets[max_idx[0]], self.assets[max_idx[1]], max_corr
    
    def get_min_correlation(self) -> Tuple[str, str, float]:
        """Find pair with minimum correlation."""
        # Get upper triangle
        n = len(self.assets)
        min_corr = np.inf
        min_pair = (None, None)
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.matrix[i, j] < min_corr:
                    min_corr = self.matrix[i, j]
                    min_pair = (self.assets[i], self.assets[j])
        
        return min_pair[0], min_pair[1], min_corr
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(
            self.matrix,
            index=self.assets,
            columns=self.assets
        )


@dataclass
class CorrelationRegime:
    """Represents a correlation regime period."""
    regime_id: str
    start_date: datetime
    end_date: datetime
    regime_type: str
    average_correlation: float
    volatility_regime: str
    n_observations: int = 0
    transition_probability: Dict[str, float] = field(default_factory=dict)
    characteristics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_days(self) -> int:
        """Calculate regime duration in days."""
        return (self.end_date - self.start_date).days
    
    def is_crisis_regime(self) -> bool:
        """Check if this is a crisis regime."""
        return (self.regime_type == "high_correlation" and 
                self.volatility_regime in ["high_volatility", "extreme_volatility"])


@dataclass
class CorrelationBreakTest:
    """Results of correlation structural break test."""
    test_statistic: float
    p_value: float
    break_dates: List[datetime]
    confidence_level: float
    method: str  # e.g., "Chow", "CUSUM", "Bai-Perron"
    significant: bool
    
    def get_break_summary(self) -> str:
        """Generate summary of breaks."""
        if not self.significant:
            return "No significant correlation breaks detected"
        
        breaks_str = ", ".join(d.strftime("%Y-%m-%d") for d in self.break_dates)
        return f"Significant breaks detected at: {breaks_str} (p-value: {self.p_value:.4f})"


@dataclass
class CorrelationStability:
    """Measures of correlation stability over time."""
    asset1: str
    asset2: str
    mean_correlation: float
    std_correlation: float
    min_correlation: float
    max_correlation: float
    coefficient_of_variation: float
    stability_score: float  # 0-1, higher is more stable
    n_regime_changes: int
    longest_stable_period_days: int
    
    def is_stable(self, threshold: float = 0.8) -> bool:
        """Check if correlation is stable."""
        return self.stability_score >= threshold
    
    def get_stability_classification(self) -> str:
        """Classify stability level."""
        if self.stability_score >= 0.8:
            return "Very Stable"
        elif self.stability_score >= 0.6:
            return "Stable"
        elif self.stability_score >= 0.4:
            return "Moderately Stable"
        elif self.stability_score >= 0.2:
            return "Unstable"
        else:
            return "Very Unstable"


@dataclass
class DynamicCorrelation:
    """Dynamic correlation model results."""
    asset1: str
    asset2: str
    time_series: List[Tuple[datetime, float]]
    model_type: str  # e.g., "DCC-GARCH", "EWMA", "Rolling"
    parameters: Dict[str, float]
    forecast: Optional[List[Tuple[datetime, float]]] = None
    confidence_bands: Optional[List[Tuple[datetime, float, float]]] = None
    
    def get_current_correlation(self) -> float:
        """Get most recent correlation value."""
        if not self.time_series:
            return np.nan
        return self.time_series[-1][1]
    
    def get_correlation_at_date(self, date: datetime) -> Optional[float]:
        """Get correlation at specific date."""
        for dt, corr in self.time_series:
            if dt.date() == date.date():
                return corr
        return None
    
    def get_trend(self, periods: int = 20) -> str:
        """Determine recent trend in correlation."""
        if len(self.time_series) < periods:
            return "Insufficient data"
        
        recent_corrs = [corr for _, corr in self.time_series[-periods:]]
        start_corr = np.mean(recent_corrs[:5])
        end_corr = np.mean(recent_corrs[-5:])
        
        diff = end_corr - start_corr
        if abs(diff) < 0.05:
            return "Stable"
        elif diff > 0:
            return "Increasing"
        else:
            return "Decreasing"


@dataclass
class CorrelationCluster:
    """Group of assets with high internal correlation."""
    cluster_id: str
    assets: List[str]
    average_internal_correlation: float
    average_external_correlation: float
    cohesion_score: float  # How tightly correlated the cluster is
    separation_score: float  # How separated from other clusters
    representative_asset: str  # Asset that best represents the cluster
    
    def get_cluster_size(self) -> int:
        """Get number of assets in cluster."""
        return len(self.assets)
    
    def is_well_defined(self, cohesion_threshold: float = 0.7,
                       separation_threshold: float = 0.3) -> bool:
        """Check if cluster is well-defined."""
        return (self.cohesion_score >= cohesion_threshold and
                self.average_external_correlation <= separation_threshold)