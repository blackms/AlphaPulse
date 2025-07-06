"""Portfolio and risk management regime integration examples."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..models.market_regime import MarketRegime
from ..models.portfolio import Portfolio, Position
from ..models.signals import Signal
from ..integration.regime_integration import (
    RegimeAwareRiskManager, 
    RegimeAwarePortfolioOptimizer
)
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class RegimeIntegratedRiskManager(RegimeAwareRiskManager):
    """Risk manager that fully integrates regime detection."""
    
    def __init__(self, config: Dict, regime_hub=None):
        """Initialize regime-integrated risk manager."""
        super().__init__(regime_hub)
        self.config = config
        self.base_risk_limit = config.get('base_risk_limit', 0.02)  # 2% base risk
        
    def calculate_position_size(
        self, 
        signal: Signal, 
        portfolio: Portfolio,
        market_data: pd.DataFrame
    ) -> float:
        """Calculate position size with full regime integration."""
        # Base position size from Kelly criterion or fixed fractional
        base_size = self._calculate_base_position_size(signal, portfolio)
        
        # Apply regime adjustment
        regime_adjusted_size = self.calculate_position_size_with_regime(base_size)
        
        # Apply regime-specific constraints
        max_position = self._get_regime_max_position(portfolio)
        regime_adjusted_size = min(regime_adjusted_size, max_position)
        
        # Adjust for regime volatility
        volatility_multiplier = self._get_regime_volatility_adjustment(market_data)
        final_size = regime_adjusted_size * volatility_multiplier
        
        # Log the adjustments
        logger.info(
            f"Position sizing - Base: {base_size:.4f}, "
            f"Regime adjusted: {regime_adjusted_size:.4f}, "
            f"Final: {final_size:.4f} "
            f"(Regime: {self._current_regime}, Confidence: {self._regime_confidence:.2%})"
        )
        
        return final_size
    
    def _calculate_base_position_size(
        self, 
        signal: Signal, 
        portfolio: Portfolio
    ) -> float:
        """Calculate base position size before regime adjustments."""
        # Simple fixed fractional for example
        return portfolio.total_value * self.base_risk_limit / signal.expected_risk
    
    def _get_regime_max_position(self, portfolio: Portfolio) -> float:
        """Get maximum position size for current regime."""
        regime_limits = {
            MarketRegime.BULL: 0.15,      # 15% max position
            MarketRegime.BEAR: 0.08,      # 8% max position
            MarketRegime.SIDEWAYS: 0.10,  # 10% max position
            MarketRegime.CRISIS: 0.05,    # 5% max position
            MarketRegime.RECOVERY: 0.12   # 12% max position
        }
        
        base_limit = regime_limits.get(self._current_regime, 0.10)
        
        # Further reduce if confidence is low
        if self._regime_confidence < 0.7:
            base_limit *= 0.8
        
        return portfolio.total_value * base_limit
    
    def _get_regime_volatility_adjustment(self, market_data: pd.DataFrame) -> float:
        """Adjust position size based on regime-specific volatility."""
        if not self.regime_hub:
            return 1.0
        
        params = self.regime_hub.get_regime_params()
        volatility_threshold = params.get('volatility_threshold', 0.2)
        
        # Calculate current volatility
        returns = market_data['close'].pct_change()
        current_vol = returns.std() * np.sqrt(252)  # Annualized
        
        # Reduce position size if volatility exceeds regime threshold
        if current_vol > volatility_threshold:
            adjustment = volatility_threshold / current_vol
            return max(adjustment, 0.5)  # At least 50% of position
        
        return 1.0
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        position_type: str,
        volatility: float
    ) -> float:
        """Calculate stop loss with regime adjustment."""
        # Base stop loss (2 ATR)
        base_stop_distance = 2 * volatility
        
        # Apply regime multiplier
        stop_multiplier = self.get_stop_loss_multiplier()
        adjusted_stop_distance = base_stop_distance * stop_multiplier
        
        # Calculate stop price
        if position_type == 'long':
            stop_price = entry_price - adjusted_stop_distance
        else:
            stop_price = entry_price + adjusted_stop_distance
        
        # Tighter stops in crisis
        if self._current_regime == MarketRegime.CRISIS:
            # Maximum 2% loss in crisis
            max_loss = entry_price * 0.02
            if position_type == 'long':
                stop_price = max(stop_price, entry_price - max_loss)
            else:
                stop_price = min(stop_price, entry_price + max_loss)
        
        return stop_price
    
    def get_portfolio_leverage_limit(self) -> float:
        """Get overall portfolio leverage limit for current regime."""
        return self.get_leverage_limit()
    
    def should_reduce_risk(self) -> bool:
        """Determine if risk should be reduced based on regime."""
        # Always reduce risk in crisis
        if self._current_regime == MarketRegime.CRISIS:
            return True
        
        # Reduce risk if transitioning to worse regime
        if self._current_regime == MarketRegime.BEAR and self._regime_confidence > 0.8:
            return True
        
        # Reduce risk if confidence is very low
        if self._regime_confidence < 0.4:
            return True
        
        return False
    
    def get_risk_budget(self, strategy_type: str) -> float:
        """Get risk budget for strategy type in current regime."""
        base_budgets = {
            'trend_following': 0.03,
            'mean_reversion': 0.02,
            'arbitrage': 0.01,
            'market_making': 0.015
        }
        
        base_budget = base_budgets.get(strategy_type, 0.02)
        
        # Adjust based on regime
        regime_multipliers = {
            MarketRegime.BULL: 1.2,
            MarketRegime.BEAR: 0.7,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.CRISIS: 0.3,
            MarketRegime.RECOVERY: 1.1
        }
        
        multiplier = regime_multipliers.get(self._current_regime, 1.0)
        
        return base_budget * multiplier


class RegimeIntegratedPortfolioOptimizer(RegimeAwarePortfolioOptimizer):
    """Portfolio optimizer with full regime integration."""
    
    def __init__(self, config: Dict, regime_hub=None):
        """Initialize regime-integrated portfolio optimizer."""
        super().__init__(regime_hub)
        self.config = config
        
    async def optimize_portfolio(
        self,
        current_portfolio: Portfolio,
        signals: List[Signal],
        market_data: Dict[str, pd.DataFrame],
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Optimize portfolio with regime awareness."""
        # Get regime-specific weights
        regime_weights = self.get_regime_weights()
        
        # Adjust optimization parameters based on regime
        optimization_params = self._get_regime_optimization_params()
        
        # Filter signals based on regime appropriateness
        filtered_signals = self._filter_signals_for_regime(signals)
        
        # Calculate optimal weights
        weights = await self._optimize_with_regime_constraints(
            current_portfolio,
            filtered_signals,
            market_data,
            regime_weights,
            optimization_params,
            constraints
        )
        
        # Apply regime-specific position limits
        weights = self._apply_regime_position_limits(weights)
        
        # Log regime impact
        logger.info(
            f"Portfolio optimization complete - Regime: {self._current_regime}, "
            f"Signals: {len(signals)} -> {len(filtered_signals)} filtered, "
            f"Risk aversion: {optimization_params['risk_aversion']}"
        )
        
        return weights
    
    def _get_regime_optimization_params(self) -> Dict:
        """Get optimization parameters for current regime."""
        regime_params = {
            MarketRegime.BULL: {
                'risk_aversion': 0.5,        # Lower risk aversion
                'return_weight': 0.7,        # Focus on returns
                'risk_weight': 0.3,
                'rebalance_threshold': 0.10, # 10% threshold
                'turnover_penalty': 0.001
            },
            MarketRegime.BEAR: {
                'risk_aversion': 2.0,        # Higher risk aversion
                'return_weight': 0.3,        # Focus on risk
                'risk_weight': 0.7,
                'rebalance_threshold': 0.05, # 5% threshold
                'turnover_penalty': 0.005
            },
            MarketRegime.SIDEWAYS: {
                'risk_aversion': 1.0,        # Neutral
                'return_weight': 0.5,
                'risk_weight': 0.5,
                'rebalance_threshold': 0.08,
                'turnover_penalty': 0.003
            },
            MarketRegime.CRISIS: {
                'risk_aversion': 3.0,        # Maximum risk aversion
                'return_weight': 0.1,        # Survival mode
                'risk_weight': 0.9,
                'rebalance_threshold': 0.03, # 3% threshold
                'turnover_penalty': 0.01
            },
            MarketRegime.RECOVERY: {
                'risk_aversion': 0.8,
                'return_weight': 0.6,
                'risk_weight': 0.4,
                'rebalance_threshold': 0.08,
                'turnover_penalty': 0.002
            }
        }
        
        return regime_params.get(self._current_regime, regime_params[MarketRegime.SIDEWAYS])
    
    def _filter_signals_for_regime(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals based on regime appropriateness."""
        if not self._current_regime:
            return signals
        
        filtered = []
        
        for signal in signals:
            # Check if signal source is appropriate for regime
            if self._is_signal_appropriate_for_regime(signal):
                # Adjust signal strength based on regime
                regime_strength_multiplier = self._get_signal_strength_multiplier(signal)
                signal.strength *= regime_strength_multiplier
                filtered.append(signal)
        
        return filtered
    
    def _is_signal_appropriate_for_regime(self, signal: Signal) -> bool:
        """Check if signal is appropriate for current regime."""
        # In crisis, only accept high-confidence defensive signals
        if self._current_regime == MarketRegime.CRISIS:
            return (signal.strength > 0.8 and 
                    signal.metadata.get('strategy') in ['defensive', 'arbitrage'])
        
        # In sideways market, prefer mean reversion
        if self._current_regime == MarketRegime.SIDEWAYS:
            return signal.metadata.get('strategy') in ['mean_reversion', 'range_trading']
        
        # In trending markets, prefer trend following
        if self._current_regime in [MarketRegime.BULL, MarketRegime.RECOVERY]:
            return signal.metadata.get('strategy') in ['trend_following', 'momentum']
        
        return True
    
    def _get_signal_strength_multiplier(self, signal: Signal) -> float:
        """Get strength multiplier for signal based on regime."""
        # Boost signals that match regime preference
        strategy = signal.metadata.get('strategy', 'unknown')
        
        regime_strategy_boost = {
            MarketRegime.BULL: {
                'trend_following': 1.3,
                'momentum': 1.2,
                'mean_reversion': 0.8
            },
            MarketRegime.BEAR: {
                'defensive': 1.5,
                'mean_reversion': 1.2,
                'trend_following': 0.7
            },
            MarketRegime.SIDEWAYS: {
                'mean_reversion': 1.4,
                'range_trading': 1.3,
                'trend_following': 0.6
            },
            MarketRegime.CRISIS: {
                'defensive': 2.0,
                'arbitrage': 1.5,
                'trend_following': 0.3
            },
            MarketRegime.RECOVERY: {
                'momentum': 1.4,
                'trend_following': 1.2,
                'defensive': 0.8
            }
        }
        
        regime_boosts = regime_strategy_boost.get(self._current_regime, {})
        return regime_boosts.get(strategy, 1.0)
    
    def _apply_regime_position_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply regime-specific position limits."""
        max_positions = {
            MarketRegime.BULL: 20,      # More positions ok
            MarketRegime.BEAR: 10,      # Concentrate
            MarketRegime.SIDEWAYS: 15,
            MarketRegime.CRISIS: 5,     # Very concentrated
            MarketRegime.RECOVERY: 15
        }
        
        max_pos = max_positions.get(self._current_regime, 15)
        
        # If too many positions, keep only the strongest
        if len(weights) > max_pos:
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            weights = dict(sorted_weights[:max_pos])
            
            # Renormalize
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    async def _optimize_with_regime_constraints(
        self,
        portfolio: Portfolio,
        signals: List[Signal],
        market_data: Dict[str, pd.DataFrame],
        regime_weights: Dict[str, float],
        optimization_params: Dict,
        constraints: Optional[Dict]
    ) -> Dict[str, float]:
        """Run optimization with regime-specific constraints."""
        # This would implement the actual optimization algorithm
        # For example, using cvxpy or scipy.optimize
        # Here's a simplified version
        
        weights = {}
        
        # Start with regime-based asset class weights
        for asset_class, target_weight in regime_weights.items():
            # Find signals for this asset class
            class_signals = [s for s in signals if s.asset_class == asset_class]
            
            if class_signals:
                # Distribute weight among signals
                signal_weight = target_weight / len(class_signals)
                for signal in class_signals:
                    weights[signal.symbol] = signal_weight * signal.strength
        
        # Apply risk parity adjustment in volatile regimes
        if self._current_regime in [MarketRegime.BEAR, MarketRegime.CRISIS]:
            weights = self._apply_risk_parity(weights, market_data)
        
        return weights
    
    def _apply_risk_parity(
        self, 
        weights: Dict[str, float], 
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Apply risk parity to weights."""
        # Calculate volatilities
        volatilities = {}
        for symbol in weights:
            if symbol in market_data:
                returns = market_data[symbol]['close'].pct_change()
                vol = returns.std() * np.sqrt(252)
                volatilities[symbol] = vol
        
        # Adjust weights inversely to volatility
        if volatilities:
            total_inv_vol = sum(1/v for v in volatilities.values())
            for symbol in weights:
                if symbol in volatilities:
                    weights[symbol] = (1/volatilities[symbol]) / total_inv_vol
        
        return weights