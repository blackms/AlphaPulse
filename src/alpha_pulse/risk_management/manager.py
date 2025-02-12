"""
Risk management system for AlphaPulse.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass, field

from loguru import logger

from .interfaces import IRiskManager, RiskMetrics
from .position_sizing import (
    AdaptivePositionSizer,
    IPositionSizer,
    PositionSizeResult,
)
from .analysis import RiskAnalyzer
from .portfolio import (
    AdaptivePortfolioOptimizer,
    IPortfolioOptimizer,
    PortfolioConstraints,
)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size: float = 0.2  # Maximum position size as fraction of portfolio
    max_portfolio_leverage: float = 1.5  # Maximum total portfolio leverage
    max_drawdown: float = 0.25  # Maximum allowed drawdown
    stop_loss: float = 0.1  # Stop loss level per position
    var_confidence: float = 0.95  # VaR confidence level
    risk_free_rate: float = 0.0  # Risk-free rate for calculations
    target_volatility: float = 0.15  # Target annual portfolio volatility
    rebalance_threshold: float = 0.1  # Threshold for portfolio rebalancing
    initial_portfolio_value: float = 100000.0  # Initial portfolio value


@dataclass
class PortfolioState:
    """Current portfolio state."""
    positions: Dict[str, Dict] = field(default_factory=dict)
    cash: float = 0.0
    portfolio_value: float = 0.0
    current_weights: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Optional[RiskMetrics] = None
    last_rebalance: Optional[pd.Timestamp] = None


class RiskManager(IRiskManager):
    """Comprehensive risk management system."""

    def __init__(
        self,
        exchange,
        config: Optional[RiskConfig] = None,
        position_sizer: Optional[IPositionSizer] = None,
        risk_analyzer: Optional[RiskAnalyzer] = None,
        portfolio_optimizer: Optional[IPortfolioOptimizer] = None,
    ):
        """
        Initialize risk management system.

        Args:
            exchange: Exchange interface instance
            config: Risk management configuration
            position_sizer: Position sizing strategy
            risk_analyzer: Risk analysis component
            portfolio_optimizer: Portfolio optimization strategy
        """
        self.exchange = exchange
        self.config = config or RiskConfig()
        self.position_sizer = position_sizer or AdaptivePositionSizer()
        self.risk_analyzer = risk_analyzer or RiskAnalyzer()
        self.portfolio_optimizer = portfolio_optimizer or AdaptivePortfolioOptimizer()
        
        self.state = PortfolioState(
            portfolio_value=self.config.initial_portfolio_value,
            cash=self.config.initial_portfolio_value
        )
        self._historical_metrics: List[RiskMetrics] = []
        
        logger.info("Initialized RiskManager")

    async def calculate_risk_exposure(self) -> Dict[str, float]:
        """Calculate current risk exposure for all positions."""
        try:
            # Get current positions and portfolio value
            portfolio_value = await self.exchange.get_portfolio_value()
            self.state.portfolio_value = portfolio_value

            # Get spot and futures positions
            spot_positions = {}
            futures_positions = {}
            
            balances = await self.exchange.get_balances()
            for asset, balance in balances.items():
                if balance.total > 0:
                    if asset == 'USDT':
                        value = balance.total
                    else:
                        price = await self.exchange.get_ticker_price(f"{asset}/USDT")
                        if price:
                            value = balance.total * price
                        else:
                            continue
                    spot_positions[asset] = value

            # Calculate exposure metrics
            exposure = {}
            
            # Net exposure
            for asset, value in spot_positions.items():
                exposure[f"{asset}_net_exposure"] = float(value)
                
                # Calculate exposure as percentage of portfolio
                if portfolio_value > 0:
                    exposure[f"{asset}_exposure_pct"] = float(value / portfolio_value)
                else:
                    exposure[f"{asset}_exposure_pct"] = 0.0

            # Add portfolio-level metrics
            exposure["total_exposure"] = float(sum(spot_positions.values()))
            exposure["exposure_ratio"] = float(
                sum(spot_positions.values()) / portfolio_value if portfolio_value > 0 else 0
            )

            return exposure
            
        except Exception as e:
            logger.error(f"Error calculating risk exposure: {str(e)}")
            raise

    async def evaluate_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        portfolio_value: float,
        current_positions: Dict[str, float],
    ) -> bool:
        """Evaluate if a trade meets risk management criteria."""
        logger.debug(f"Evaluating trade: {symbol} {side} {quantity} @ {current_price}")
        logger.debug(f"Portfolio value: {portfolio_value}, Current positions: {current_positions}")
        
        # Update portfolio state
        self.state.portfolio_value = portfolio_value
        self.state.positions = current_positions

        # Calculate position value
        position_value = float(quantity) * float(current_price)
        position_size = position_value / float(portfolio_value) if float(portfolio_value) > 0 else 0
        logger.debug(f"Position value: {position_value}, Position size: {position_size:.2%}")

        # Check position size limits
        if position_size > self.config.max_position_size:
            logger.warning(
                f"Trade rejected: Position size ({position_size:.2%}) "
                f"exceeds limit ({self.config.max_position_size:.2%})"
            )
            return False

        # Calculate total exposure
        try:
            total_exposure = sum(
                abs(pos['quantity'] * pos['current_price'])
                for pos in current_positions.values()
            ) + position_value
            leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
            logger.debug(f"Total exposure: {total_exposure}, Leverage: {leverage:.2f}")
            
            if leverage > self.config.max_portfolio_leverage:
                logger.warning(
                    f"Trade rejected: Portfolio leverage "
                    f"({leverage:.2f}) exceeds limit "
                    f"({self.config.max_portfolio_leverage:.2f})"
                )
                return False
        except Exception as e:
            logger.error(f"Error calculating exposure: {str(e)}")
            return False

        # Check drawdown limit if we have risk metrics
        if self.state.risk_metrics:
            drawdown = abs(self.state.risk_metrics.max_drawdown)
            logger.debug(f"Current drawdown: {drawdown:.2%}")
            if drawdown > self.config.max_drawdown:
                logger.warning(
                    f"Trade rejected: Maximum drawdown "
                    f"({drawdown:.2%}) exceeded limit "
                    f"({self.config.max_drawdown:.2%})"
                )
                return False

        logger.debug("Trade passed all risk checks")
        return True

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        signal_strength: float,
        historical_returns: Optional[pd.Series] = None,
    ) -> PositionSizeResult:
        """Calculate recommended position size."""
        volatility = (
            self.state.risk_metrics.volatility
            if self.state.risk_metrics
            else 0.15  # Default assumption
        )
        
        return self.position_sizer.calculate_position_size(
            symbol=symbol,
            current_price=current_price,
            portfolio_value=self.state.portfolio_value,
            volatility=volatility,
            signal_strength=signal_strength,
            historical_returns=historical_returns,
        )

    def update_risk_metrics(
        self,
        portfolio_returns: pd.Series,
        asset_returns: Dict[str, pd.Series],
    ) -> None:
        """Update risk metrics with new data."""
        # Calculate portfolio risk metrics
        risk_metrics = self.risk_analyzer.calculate_metrics(
            portfolio_returns,
            self.config.risk_free_rate
        )
        
        self.state.risk_metrics = risk_metrics
        self._historical_metrics.append(risk_metrics)
        
        # Check if rebalancing is needed
        if self._should_rebalance(asset_returns):
            self._rebalance_portfolio(asset_returns)

        logger.info(
            f"Updated risk metrics - Volatility: {risk_metrics.volatility:.2%}, "
            f"VaR: {risk_metrics.var_95:.2%}, MaxDD: {risk_metrics.max_drawdown:.2%}"
        )

    def get_stop_loss_prices(
        self,
        positions: Dict[str, Dict],
    ) -> Dict[str, float]:
        """Calculate stop-loss prices for current positions."""
        stop_losses = {}
        for symbol, position in positions.items():
            entry_price = position['avg_entry_price']
            current_price = position['current_price']
            quantity = position['quantity']
            
            # Calculate stop loss based on entry price
            if quantity > 0:  # Long position
                stop_price = entry_price * (1 - self.config.stop_loss)
            else:  # Short position
                stop_price = entry_price * (1 + self.config.stop_loss)
            
            stop_losses[symbol] = stop_price
            
        return stop_losses

    def get_position_limits(
        self,
        portfolio_value: float,
    ) -> Dict[str, float]:
        """Get maximum position sizes for risk management."""
        base_limit = portfolio_value * self.config.max_position_size
        
        # Adjust limits based on current portfolio state
        if self.state.risk_metrics:
            # Reduce limits in high volatility environments
            vol_adjustment = max(
                0.5,
                1 - (self.state.risk_metrics.volatility / self.config.target_volatility)
            )
            adjusted_limit = base_limit * vol_adjustment
        else:
            adjusted_limit = base_limit
        
        # Return same limit for all assets (can be extended to be asset-specific)
        return {'default': adjusted_limit}

    def _should_rebalance(self, asset_returns: Dict[str, pd.Series]) -> bool:
        """Determine if portfolio rebalancing is needed."""
        if not self.state.last_rebalance:
            return True
            
        # Check if enough time has passed since last rebalance
        time_since_rebalance = pd.Timestamp.now() - self.state.last_rebalance
        if time_since_rebalance.days < 1:  # Minimum 1 day between rebalances
            return False
            
        # Check if weights have deviated significantly
        if not self.state.current_weights:
            return True
            
        returns_df = pd.DataFrame(asset_returns)
        optimal_weights = self.portfolio_optimizer.optimize(
            returns_df,
            self.config.risk_free_rate,
            PortfolioConstraints(
                max_total_weight=self.config.max_portfolio_leverage
            )
        )
        
        # Calculate maximum weight deviation
        max_deviation = max(
            abs(self.state.current_weights.get(symbol, 0) - weight)
            for symbol, weight in optimal_weights.items()
        )
        
        return max_deviation > self.config.rebalance_threshold

    def _rebalance_portfolio(self, asset_returns: Dict[str, pd.Series]) -> None:
        """Calculate and store new optimal portfolio weights."""
        returns_df = pd.DataFrame(asset_returns)
        
        # Optimize portfolio weights
        new_weights = self.portfolio_optimizer.optimize(
            returns_df,
            self.config.risk_free_rate,
            PortfolioConstraints(
                max_total_weight=self.config.max_portfolio_leverage
            )
        )
        
        self.state.current_weights = new_weights
        self.state.last_rebalance = pd.Timestamp.now()
        
        logger.info(f"Portfolio rebalanced - New weights: {new_weights}")

    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk management report."""
        if not self.state.risk_metrics:
            return {}
            
        return {
            'portfolio_value': self.state.portfolio_value,
            'current_leverage': (
                sum(
                    abs(pos['quantity'] * pos['current_price'])
                    for pos in self.state.positions.values()
                ) / max(self.state.portfolio_value, 1e-10)  # Avoid division by zero
            ),
            'risk_metrics': {
                'volatility': self.state.risk_metrics.volatility,
                'var_95': self.state.risk_metrics.var_95,
                'cvar_95': self.state.risk_metrics.cvar_95,
                'max_drawdown': self.state.risk_metrics.max_drawdown,
                'sharpe_ratio': self.state.risk_metrics.sharpe_ratio,
                'sortino_ratio': self.state.risk_metrics.sortino_ratio,
                'calmar_ratio': self.state.risk_metrics.calmar_ratio,
            },
            'current_weights': self.state.current_weights,
            'last_rebalance': self.state.last_rebalance,
        }
