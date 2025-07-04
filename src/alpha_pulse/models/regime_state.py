"""
Market Regime State Representations and Management.

This module provides detailed state representations for market regimes,
including state persistence, serialization, and regime-specific configurations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RegimeCharacteristic(Enum):
    """Characteristics that define market regimes."""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    MOMENTUM = "momentum"
    VALUE = "value"
    DEFENSIVE = "defensive"
    GROWTH = "growth"


@dataclass
class RegimeState:
    """
    Comprehensive representation of a market regime state.
    
    This class encapsulates all information about a specific market regime,
    including its statistical properties, trading characteristics, and
    optimal strategies.
    """
    regime_id: int
    regime_name: str
    regime_type: str  # From RegimeType enum
    
    # Statistical properties
    mean_returns: float
    volatility: float
    skewness: float
    kurtosis: float
    sharpe_ratio: float
    
    # Market characteristics
    characteristics: List[RegimeCharacteristic] = field(default_factory=list)
    
    # Regime stability
    typical_duration: float = 20.0
    stability_score: float = 0.5
    transition_probabilities: Dict[int, float] = field(default_factory=dict)
    
    # Trading parameters
    optimal_leverage: float = 1.0
    recommended_strategies: List[str] = field(default_factory=list)
    risk_parameters: Dict[str, float] = field(default_factory=dict)
    
    # Feature importance
    key_features: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    last_occurrence: Optional[datetime] = None
    total_occurrences: int = 0
    total_duration: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert regime state to dictionary."""
        data = asdict(self)
        # Convert enums to strings
        data['characteristics'] = [c.value for c in self.characteristics]
        # Convert datetime
        if self.last_occurrence:
            data['last_occurrence'] = self.last_occurrence.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegimeState':
        """Create regime state from dictionary."""
        # Convert characteristics back to enums
        if 'characteristics' in data:
            data['characteristics'] = [
                RegimeCharacteristic(c) for c in data['characteristics']
            ]
        # Convert datetime
        if 'last_occurrence' in data and data['last_occurrence']:
            data['last_occurrence'] = datetime.fromisoformat(data['last_occurrence'])
        return cls(**data)
    
    def update_statistics(self, returns: np.ndarray, features: pd.DataFrame):
        """Update regime statistics with new observations."""
        if len(returns) == 0:
            return
        
        # Update return statistics
        self.mean_returns = np.mean(returns)
        self.volatility = np.std(returns) * np.sqrt(252)  # Annualized
        self.skewness = float(pd.Series(returns).skew())
        self.kurtosis = float(pd.Series(returns).kurt())
        
        # Update Sharpe ratio
        if self.volatility > 0:
            self.sharpe_ratio = self.mean_returns / self.volatility * np.sqrt(252)
        else:
            self.sharpe_ratio = 0.0
        
        # Update feature importance (simplified)
        if not features.empty:
            feature_means = features.mean()
            total_importance = feature_means.abs().sum()
            if total_importance > 0:
                self.key_features = (feature_means.abs() / total_importance).to_dict()
    
    def get_risk_adjusted_position_size(self, 
                                      base_position: float,
                                      current_volatility: float) -> float:
        """Calculate risk-adjusted position size for this regime."""
        # Adjust for regime volatility
        vol_adjustment = self.volatility / current_volatility if current_volatility > 0 else 1.0
        vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)
        
        # Apply optimal leverage
        position_size = base_position * self.optimal_leverage * vol_adjustment
        
        # Apply regime-specific limits
        max_position = self.risk_parameters.get('max_position_size', 1.0)
        min_position = self.risk_parameters.get('min_position_size', 0.1)
        
        return np.clip(position_size, min_position, max_position)
    
    def is_similar_to(self, other: 'RegimeState', threshold: float = 0.8) -> bool:
        """Check if two regime states are similar."""
        # Compare statistical properties
        return_similarity = 1 - abs(self.mean_returns - other.mean_returns) / 0.1
        vol_similarity = 1 - abs(self.volatility - other.volatility) / 0.5
        
        # Compare characteristics
        common_chars = set(self.characteristics) & set(other.characteristics)
        total_chars = set(self.characteristics) | set(other.characteristics)
        char_similarity = len(common_chars) / len(total_chars) if total_chars else 0
        
        # Weighted similarity
        similarity = (0.3 * return_similarity + 
                     0.3 * vol_similarity + 
                     0.4 * char_similarity)
        
        return similarity >= threshold


@dataclass
class RegimeStateHistory:
    """Track historical regime state information."""
    regime_states: List[RegimeState] = field(default_factory=list)
    state_transitions: List[Tuple[int, int, datetime]] = field(default_factory=list)
    state_durations: Dict[int, List[int]] = field(default_factory=dict)
    
    def add_state(self, state: RegimeState):
        """Add or update a regime state."""
        # Check if state already exists
        for i, existing_state in enumerate(self.regime_states):
            if existing_state.regime_id == state.regime_id:
                self.regime_states[i] = state
                return
        
        self.regime_states.append(state)
    
    def record_transition(self, from_state: int, to_state: int, timestamp: datetime):
        """Record a state transition."""
        self.state_transitions.append((from_state, to_state, timestamp))
        
        # Update duration tracking
        if len(self.state_transitions) > 1:
            prev_transition = self.state_transitions[-2]
            duration = (timestamp - prev_transition[2]).days
            
            if prev_transition[1] not in self.state_durations:
                self.state_durations[prev_transition[1]] = []
            self.state_durations[prev_transition[1]].append(duration)
    
    def get_state_by_id(self, regime_id: int) -> Optional[RegimeState]:
        """Get regime state by ID."""
        for state in self.regime_states:
            if state.regime_id == regime_id:
                return state
        return None
    
    def get_most_common_transitions(self, n: int = 5) -> List[Tuple[Tuple[int, int], int]]:
        """Get most common state transitions."""
        from collections import Counter
        
        transition_pairs = [(t[0], t[1]) for t in self.state_transitions]
        counter = Counter(transition_pairs)
        
        return counter.most_common(n)
    
    def get_regime_report(self) -> str:
        """Generate a report of regime history."""
        report = []
        report.append("=" * 50)
        report.append("REGIME STATE HISTORY REPORT")
        report.append("=" * 50)
        
        # State summary
        report.append("\nREGIME STATES:")
        for state in self.regime_states:
            report.append(f"\n{state.regime_name} (ID: {state.regime_id})")
            report.append(f"  Type: {state.regime_type}")
            report.append(f"  Mean Returns: {state.mean_returns:.2%}")
            report.append(f"  Volatility: {state.volatility:.2%}")
            report.append(f"  Sharpe Ratio: {state.sharpe_ratio:.2f}")
            report.append(f"  Typical Duration: {state.typical_duration:.1f} days")
            report.append(f"  Characteristics: {', '.join(c.value for c in state.characteristics)}")
        
        # Transition summary
        report.append("\nMOST COMMON TRANSITIONS:")
        for (from_id, to_id), count in self.get_most_common_transitions():
            from_state = self.get_state_by_id(from_id)
            to_state = self.get_state_by_id(to_id)
            if from_state and to_state:
                report.append(f"  {from_state.regime_name} -> {to_state.regime_name}: {count} times")
        
        # Duration statistics
        report.append("\nDURATION STATISTICS:")
        for regime_id, durations in self.state_durations.items():
            state = self.get_state_by_id(regime_id)
            if state and durations:
                report.append(f"  {state.regime_name}:")
                report.append(f"    Mean: {np.mean(durations):.1f} days")
                report.append(f"    Std: {np.std(durations):.1f} days")
                report.append(f"    Min: {np.min(durations)} days")
                report.append(f"    Max: {np.max(durations)} days")
        
        return "\n".join(report)


class RegimeStateFactory:
    """Factory for creating predefined regime states."""
    
    @staticmethod
    def create_bull_market() -> RegimeState:
        """Create a bull market regime state."""
        return RegimeState(
            regime_id=0,
            regime_name="Bull Market",
            regime_type="bull",
            mean_returns=0.12,  # 12% annualized
            volatility=0.15,    # 15% annualized
            skewness=-0.5,
            kurtosis=3.0,
            sharpe_ratio=0.8,
            characteristics=[
                RegimeCharacteristic.TRENDING,
                RegimeCharacteristic.LOW_VOLATILITY,
                RegimeCharacteristic.RISK_ON,
                RegimeCharacteristic.MOMENTUM
            ],
            typical_duration=250,
            stability_score=0.8,
            optimal_leverage=1.5,
            recommended_strategies=["trend_following", "momentum", "growth"],
            risk_parameters={
                "max_position_size": 1.5,
                "stop_loss": 0.05,
                "take_profit": 0.15
            }
        )
    
    @staticmethod
    def create_bear_market() -> RegimeState:
        """Create a bear market regime state."""
        return RegimeState(
            regime_id=1,
            regime_name="Bear Market",
            regime_type="bear",
            mean_returns=-0.15,
            volatility=0.25,
            skewness=0.5,
            kurtosis=4.0,
            sharpe_ratio=-0.6,
            characteristics=[
                RegimeCharacteristic.TRENDING,
                RegimeCharacteristic.HIGH_VOLATILITY,
                RegimeCharacteristic.RISK_OFF,
                RegimeCharacteristic.DEFENSIVE
            ],
            typical_duration=150,
            stability_score=0.6,
            optimal_leverage=0.5,
            recommended_strategies=["short_selling", "defensive", "volatility"],
            risk_parameters={
                "max_position_size": 0.5,
                "stop_loss": 0.03,
                "take_profit": 0.08
            }
        )
    
    @staticmethod
    def create_sideways_market() -> RegimeState:
        """Create a sideways/ranging market regime state."""
        return RegimeState(
            regime_id=2,
            regime_name="Sideways Market",
            regime_type="sideways",
            mean_returns=0.05,
            volatility=0.12,
            skewness=0.0,
            kurtosis=3.0,
            sharpe_ratio=0.4,
            characteristics=[
                RegimeCharacteristic.MEAN_REVERTING,
                RegimeCharacteristic.LOW_VOLATILITY,
                RegimeCharacteristic.VALUE
            ],
            typical_duration=100,
            stability_score=0.5,
            optimal_leverage=1.0,
            recommended_strategies=["mean_reversion", "range_trading", "arbitrage"],
            risk_parameters={
                "max_position_size": 1.0,
                "stop_loss": 0.03,
                "take_profit": 0.05
            }
        )
    
    @staticmethod
    def create_crisis_market() -> RegimeState:
        """Create a crisis/crash market regime state."""
        return RegimeState(
            regime_id=3,
            regime_name="Crisis Market",
            regime_type="crisis",
            mean_returns=-0.30,
            volatility=0.40,
            skewness=1.0,
            kurtosis=6.0,
            sharpe_ratio=-0.75,
            characteristics=[
                RegimeCharacteristic.HIGH_VOLATILITY,
                RegimeCharacteristic.RISK_OFF,
                RegimeCharacteristic.DEFENSIVE
            ],
            typical_duration=30,
            stability_score=0.3,
            optimal_leverage=0.2,
            recommended_strategies=["cash", "hedging", "safe_haven"],
            risk_parameters={
                "max_position_size": 0.2,
                "stop_loss": 0.02,
                "reduce_all_positions": True
            }
        )
    
    @staticmethod
    def create_recovery_market() -> RegimeState:
        """Create a recovery market regime state."""
        return RegimeState(
            regime_id=4,
            regime_name="Recovery Market",
            regime_type="recovery",
            mean_returns=0.18,
            volatility=0.20,
            skewness=-0.3,
            kurtosis=3.5,
            sharpe_ratio=0.9,
            characteristics=[
                RegimeCharacteristic.TRENDING,
                RegimeCharacteristic.RISK_ON,
                RegimeCharacteristic.VALUE,
                RegimeCharacteristic.GROWTH
            ],
            typical_duration=120,
            stability_score=0.6,
            optimal_leverage=1.2,
            recommended_strategies=["value_investing", "sector_rotation", "growth"],
            risk_parameters={
                "max_position_size": 1.2,
                "stop_loss": 0.04,
                "take_profit": 0.12,
                "scale_in": True
            }
        )
    
    @staticmethod
    def create_default_states() -> List[RegimeState]:
        """Create a default set of regime states."""
        return [
            RegimeStateFactory.create_bull_market(),
            RegimeStateFactory.create_bear_market(),
            RegimeStateFactory.create_sideways_market(),
            RegimeStateFactory.create_crisis_market(),
            RegimeStateFactory.create_recovery_market()
        ]


class RegimeStateManager:
    """Manage regime states and their configurations."""
    
    def __init__(self):
        self.states: Dict[int, RegimeState] = {}
        self.history = RegimeStateHistory()
        self.current_state_id: Optional[int] = None
        
        # Initialize with default states
        for state in RegimeStateFactory.create_default_states():
            self.add_state(state)
    
    def add_state(self, state: RegimeState):
        """Add a regime state to the manager."""
        self.states[state.regime_id] = state
        self.history.add_state(state)
    
    def update_state(self, regime_id: int, returns: np.ndarray, features: pd.DataFrame):
        """Update a regime state with new data."""
        if regime_id in self.states:
            self.states[regime_id].update_statistics(returns, features)
            self.states[regime_id].total_occurrences += 1
            self.states[regime_id].last_occurrence = datetime.now()
    
    def transition_to(self, new_state_id: int, timestamp: Optional[datetime] = None):
        """Record a transition to a new state."""
        timestamp = timestamp or datetime.now()
        
        if self.current_state_id is not None:
            self.history.record_transition(
                self.current_state_id, 
                new_state_id, 
                timestamp
            )
        
        self.current_state_id = new_state_id
    
    def get_current_state(self) -> Optional[RegimeState]:
        """Get the current regime state."""
        if self.current_state_id is not None:
            return self.states.get(self.current_state_id)
        return None
    
    def get_state_by_type(self, regime_type: str) -> Optional[RegimeState]:
        """Get regime state by type."""
        for state in self.states.values():
            if state.regime_type == regime_type:
                return state
        return None
    
    def save_states(self, filepath: str):
        """Save regime states to file."""
        data = {
            'states': [state.to_dict() for state in self.states.values()],
            'current_state_id': self.current_state_id,
            'history': {
                'transitions': [
                    (f, t, ts.isoformat()) 
                    for f, t, ts in self.history.state_transitions
                ],
                'durations': self.history.state_durations
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_states(self, filepath: str):
        """Load regime states from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load states
        self.states.clear()
        for state_data in data['states']:
            state = RegimeState.from_dict(state_data)
            self.add_state(state)
        
        # Load current state
        self.current_state_id = data.get('current_state_id')
        
        # Load history
        if 'history' in data:
            self.history.state_transitions = [
                (f, t, datetime.fromisoformat(ts))
                for f, t, ts in data['history']['transitions']
            ]
            self.history.state_durations = data['history']['durations']