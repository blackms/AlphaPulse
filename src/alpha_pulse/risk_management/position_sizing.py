"""
Position sizing implementations for AlphaPulse.
"""
from typing import Dict, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from loguru import logger

from .interfaces import IPositionSizer, PositionSizeResult


@dataclass
class KellyParams:
    """Parameters for Kelly Criterion calculation."""
    win_rate: float  # Historical win rate
    profit_ratio: float  # Average win/loss ratio
    fraction: float = 1.0  # Kelly fraction (1.0 = full Kelly)


class KellyCriterionSizer(IPositionSizer):
    """Position sizer using the Kelly Criterion."""

    def __init__(
        self,
        lookback_periods: int = 100,
        min_trades: int = 20,
        max_size_pct: float = 0.2,
    ):
        """
        Initialize Kelly Criterion position sizer.

        Args:
            lookback_periods: Number of periods to look back for win rate calculation
            min_trades: Minimum number of trades required for calculation
            max_size_pct: Maximum position size as percentage of portfolio
        """
        self.lookback_periods = lookback_periods
        self.min_trades = min_trades
        self.max_size_pct = max_size_pct
        self._trade_history: Dict[str, pd.DataFrame] = {}
        logger.info(
            f"Initialized KellyCriterionSizer (lookback={lookback_periods}, "
            f"min_trades={min_trades}, max_size={max_size_pct*100}%)"
        )

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        volatility: float,
        signal_strength: float,
        historical_returns: Optional[pd.Series] = None,
    ) -> PositionSizeResult:
        """Calculate position size using Kelly Criterion."""
        # Get Kelly parameters from historical data if available
        kelly_params = self._calculate_kelly_params(symbol, historical_returns)
        
        # Calculate Kelly fraction
        if kelly_params:
            kelly_fraction = (
                kelly_params.win_rate - 
                (1 - kelly_params.win_rate) / kelly_params.profit_ratio
            )
            
            # Adjust fraction by signal strength and volatility
            kelly_fraction *= signal_strength * (1 - min(volatility, 0.5))
            
            # Apply Kelly fraction limit
            kelly_fraction = min(kelly_fraction, self.max_size_pct)
            
            confidence = min(
                1.0,
                len(self._trade_history.get(symbol, pd.DataFrame())) / self.min_trades
            )
        else:
            # Use conservative estimate without historical data
            kelly_fraction = self.max_size_pct * signal_strength * (1 - min(volatility, 0.5))
            confidence = 0.5

        # Calculate final position size
        position_size = portfolio_value * max(0, kelly_fraction)
        
        return PositionSizeResult(
            size=position_size,
            confidence=confidence,
            metrics={
                "kelly_fraction": kelly_fraction,
                "win_rate": kelly_params.win_rate if kelly_params else None,
                "profit_ratio": kelly_params.profit_ratio if kelly_params else None,
            }
        )

    def _calculate_kelly_params(
        self,
        symbol: str,
        historical_returns: Optional[pd.Series],
    ) -> Optional[KellyParams]:
        """Calculate Kelly Criterion parameters from historical data."""
        if symbol not in self._trade_history or historical_returns is None:
            return None
            
        trades = self._trade_history[symbol].tail(self.lookback_periods)
        if len(trades) < self.min_trades:
            return None
            
        # Calculate win rate
        wins = trades[trades["pnl"] > 0]
        win_rate = len(wins) / len(trades)
        
        # Calculate profit ratio
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        losses = trades[trades["pnl"] < 0]
        avg_loss = abs(losses["pnl"].mean()) if len(losses) > 0 else 1
        profit_ratio = avg_win / avg_loss if avg_loss != 0 else 1
        
        return KellyParams(
            win_rate=win_rate,
            profit_ratio=profit_ratio,
            fraction=1.0,  # Using full Kelly
        )

    def update_trade_history(
        self,
        symbol: str,
        trade_data: Dict,
    ) -> None:
        """
        Update trade history for a symbol.

        Args:
            symbol: Trading symbol
            trade_data: Dictionary containing trade details
        """
        if symbol not in self._trade_history:
            self._trade_history[symbol] = pd.DataFrame()
        
        self._trade_history[symbol] = pd.concat([
            self._trade_history[symbol],
            pd.DataFrame([trade_data])
        ]).tail(self.lookback_periods)


class VolatilityBasedSizer(IPositionSizer):
    """Position sizer using volatility-based sizing."""

    def __init__(
        self,
        target_volatility: float = 0.01,  # Target daily volatility
        max_size_pct: float = 0.2,
        volatility_lookback: int = 20,
        min_size_pct: float = 0.01,
    ):
        """
        Initialize volatility-based position sizer.

        Args:
            target_volatility: Target portfolio volatility (default: 1% daily)
            max_size_pct: Maximum position size as percentage of portfolio
            volatility_lookback: Periods for volatility calculation
            min_size_pct: Minimum position size as percentage of portfolio
        """
        self.target_volatility = target_volatility
        self.max_size_pct = max_size_pct
        self.volatility_lookback = volatility_lookback
        self.min_size_pct = min_size_pct
        logger.info(
            f"Initialized VolatilityBasedSizer (target_vol={target_volatility*100}%, "
            f"max_size={max_size_pct*100}%, lookback={volatility_lookback})"
        )

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        volatility: float,
        signal_strength: float,
        historical_returns: Optional[pd.Series] = None,
    ) -> PositionSizeResult:
        """Calculate position size based on volatility targeting."""
        if historical_returns is not None and len(historical_returns) >= self.volatility_lookback:
            # Calculate rolling volatility
            rolling_vol = (
                historical_returns
                .rolling(self.volatility_lookback)
                .std()
                .iloc[-1]
            )
            # Use max of current and historical volatility
            volatility = max(volatility, rolling_vol)
            confidence = 0.8
        else:
            confidence = 0.5

        # Calculate position size to target portfolio volatility
        if volatility > 0:
            position_pct = self.target_volatility / volatility
        else:
            position_pct = self.max_size_pct

        # Apply limits
        position_pct = min(
            self.max_size_pct,
            max(self.min_size_pct, position_pct)
        )
        
        # Adjust by signal strength
        position_pct *= signal_strength
        
        # Calculate final position size
        position_size = portfolio_value * position_pct
        
        return PositionSizeResult(
            size=position_size,
            confidence=confidence,
            metrics={
                "position_pct": position_pct,
                "volatility": volatility,
                "target_volatility": self.target_volatility,
            }
        )


class AdaptivePositionSizer(IPositionSizer):
    """
    Adaptive position sizer that combines multiple strategies and adapts based on market conditions.
    """

    def __init__(
        self,
        kelly_sizer: Optional[KellyCriterionSizer] = None,
        vol_sizer: Optional[VolatilityBasedSizer] = None,
        max_size_pct: float = 0.2,
    ):
        """
        Initialize adaptive position sizer.

        Args:
            kelly_sizer: Kelly Criterion sizer instance
            vol_sizer: Volatility-based sizer instance
            max_size_pct: Maximum position size as percentage of portfolio
        """
        self.kelly_sizer = kelly_sizer or KellyCriterionSizer()
        self.vol_sizer = vol_sizer or VolatilityBasedSizer()
        self.max_size_pct = max_size_pct
        logger.info("Initialized AdaptivePositionSizer")

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        volatility: float,
        signal_strength: float,
        historical_returns: Optional[pd.Series] = None,
    ) -> PositionSizeResult:
        """Calculate position size using adaptive strategy selection."""
        # Get recommendations from both strategies
        kelly_result = self.kelly_sizer.calculate_position_size(
            symbol=symbol,
            current_price=current_price,
            portfolio_value=portfolio_value,
            volatility=volatility,
            signal_strength=signal_strength,
            historical_returns=historical_returns,
        )
        
        vol_result = self.vol_sizer.calculate_position_size(
            symbol=symbol,
            current_price=current_price,
            portfolio_value=portfolio_value,
            volatility=volatility,
            signal_strength=signal_strength,
            historical_returns=historical_returns,
        )
        
        # Weight results by their confidence
        total_confidence = kelly_result.confidence + vol_result.confidence
        if total_confidence > 0:
            kelly_weight = kelly_result.confidence / total_confidence
            vol_weight = vol_result.confidence / total_confidence
        else:
            kelly_weight = vol_weight = 0.5

        # Calculate weighted position size
        position_size = (
            kelly_result.size * kelly_weight +
            vol_result.size * vol_weight
        )
        
        # Apply maximum size limit
        position_size = min(
            position_size,
            portfolio_value * self.max_size_pct
        )
        
        return PositionSizeResult(
            size=position_size,
            confidence=max(kelly_result.confidence, vol_result.confidence),
            metrics={
                "kelly_weight": kelly_weight,
                "vol_weight": vol_weight,
                "kelly_size": kelly_result.size,
                "vol_size": vol_result.size,
                **kelly_result.metrics,
                **{f"vol_{k}": v for k, v in vol_result.metrics.items()},
            }
        )