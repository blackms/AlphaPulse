"""
Risk management system for AlphaPulse.
"""
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from dataclasses import dataclass, field
import asyncio

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
from alpha_pulse.decorators.audit_decorators import (
    audit_risk_check,
    audit_trade_decision,
    audit_portfolio_action
)
from alpha_pulse.risk.correlation_analyzer import (
    CorrelationAnalyzer,
    CorrelationAnalysisConfig,
    CorrelationMethod
)
from alpha_pulse.services.monte_carlo_integration_service import MonteCarloIntegrationService


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
        correlation_analyzer: Optional[CorrelationAnalyzer] = None,
        risk_budgeting_service: Optional[Any] = None,
        monte_carlo_service: Optional[MonteCarloIntegrationService] = None,
    ):
        """
        Initialize risk management system.

        Args:
            exchange: Exchange interface instance
            config: Risk management configuration
            position_sizer: Position sizing strategy
            risk_analyzer: Risk analysis component
            portfolio_optimizer: Portfolio optimization strategy
            correlation_analyzer: Correlation analysis component
            risk_budgeting_service: Dynamic risk budgeting service
            monte_carlo_service: Monte Carlo integration service
        """
        self.exchange = exchange
        self.config = config or RiskConfig()
        self.position_sizer = position_sizer or AdaptivePositionSizer()
        self.risk_analyzer = risk_analyzer or RiskAnalyzer()
        self.portfolio_optimizer = portfolio_optimizer or AdaptivePortfolioOptimizer()
        self.correlation_analyzer = correlation_analyzer or CorrelationAnalyzer()
        self.risk_budgeting_service = risk_budgeting_service
        self.monte_carlo_service = monte_carlo_service or MonteCarloIntegrationService()
        
        self.state = PortfolioState(
            portfolio_value=self.config.initial_portfolio_value,
            cash=self.config.initial_portfolio_value
        )
        self._historical_metrics: List[RiskMetrics] = []
        self._historical_returns: Dict[str, pd.Series] = {}
        
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

    @audit_risk_check(risk_type='trade_evaluation', threshold_param='max_position_size', value_param='position_size')
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

        # Get dynamic limits if available
        max_position_size = self.config.max_position_size
        max_leverage = self.config.max_portfolio_leverage
        
        if self.risk_budgeting_service:
            try:
                # Get current budget
                portfolio_budget = await self.risk_budgeting_service.get_portfolio_budget()
                if portfolio_budget:
                    # Use dynamic leverage limit
                    max_leverage = portfolio_budget.leverage_limit
                    
                    # Check symbol-specific limits
                    for allocation in portfolio_budget.allocations:
                        if allocation.entity_type == 'asset' and allocation.entity_id == symbol:
                            # Use remaining budget as position limit
                            remaining_pct = (allocation.allocated_amount - allocation.utilized_amount) / portfolio_value
                            max_position_size = min(max_position_size, max(0, remaining_pct))
                            logger.debug(f"Using dynamic position limit for {symbol}: {max_position_size:.2%}")
                            break
            except Exception as e:
                logger.warning(f"Failed to get dynamic limits: {e}")
        
        # Check position size limits with tolerance for floating-point rounding
        TOLERANCE = 1e-10  # Small tolerance for floating-point comparison
        if position_size > (max_position_size + TOLERANCE):
            logger.warning(
                f"Trade rejected: Position size ({position_size:.2%}) "
                f"exceeds limit ({max_position_size:.2%})"
            )
            return False

        # Calculate total exposure
        try:
            total_exposure = sum(
                abs(float(pos['quantity']) * float(pos['current_price']))
                for pos in current_positions.values()
            ) + position_value
            leverage = total_exposure / float(portfolio_value) if float(portfolio_value) > 0 else 0
            logger.debug(f"Total exposure: {total_exposure}, Leverage: {leverage:.2f}")
            
            if leverage > max_leverage:
                logger.warning(
                    f"Trade rejected: Portfolio leverage "
                    f"({leverage:.2f}) exceeds limit "
                    f"({max_leverage:.2f})"
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

    @audit_trade_decision(extract_reasoning=True, include_market_data=False)
    async def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        signal_strength: float,
        historical_returns: Optional[pd.Series] = None,
        strategy_name: Optional[str] = None,
    ) -> PositionSizeResult:
        """Calculate recommended position size with dynamic risk budgets."""
        volatility = (
            self.state.risk_metrics.volatility
            if self.state.risk_metrics
            else 0.15  # Default assumption
        )
        
        # Get risk budget constraints if service is available
        risk_budget = None
        if self.risk_budgeting_service:
            try:
                # Get current budget allocations
                portfolio_budget = await self.risk_budgeting_service.get_portfolio_budget()
                
                if portfolio_budget:
                    # Extract relevant budget constraints
                    risk_budget = {
                        'total_budget_limit': portfolio_budget.leverage_limit,
                        'symbol_budget': {},
                        'strategy_budget': None
                    }
                    
                    # Find allocation for this symbol or strategy
                    for allocation in portfolio_budget.allocations:
                        if allocation.entity_type == 'asset' and allocation.entity_id == symbol:
                            # Calculate remaining budget for this symbol
                            remaining_budget = (allocation.allocated_amount - allocation.utilized_amount) / self.state.portfolio_value
                            risk_budget['symbol_budget'][symbol] = max(0, remaining_budget)
                        elif allocation.entity_type == 'strategy' and allocation.entity_id == strategy_name:
                            remaining_budget = (allocation.allocated_amount - allocation.utilized_amount) / self.state.portfolio_value
                            risk_budget['strategy_budget'] = max(0, remaining_budget)
                    
                    logger.debug(f"Applied risk budget constraints for {symbol}: {risk_budget}")
                    
            except Exception as e:
                logger.warning(f"Failed to get risk budget: {e}")
        
        return self.position_sizer.calculate_position_size(
            symbol=symbol,
            current_price=current_price,
            portfolio_value=self.state.portfolio_value,
            volatility=volatility,
            signal_strength=signal_strength,
            historical_returns=historical_returns,
            risk_budget=risk_budget,
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
        
        # Store historical returns for correlation analysis
        self._historical_returns = asset_returns
        
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
        
        report = {
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
        
        # Add correlation analysis if we have historical returns
        if self._historical_returns:
            try:
                # Convert dict of Series to DataFrame
                returns_df = pd.DataFrame(self._historical_returns)
                
                if not returns_df.empty and len(returns_df) >= 30:  # Need minimum data
                    # Get comprehensive correlation summary
                    correlation_summary = self.correlation_analyzer.get_correlation_summary(returns_df)
                    
                    # Calculate correlation matrices
                    pearson_matrix = self.correlation_analyzer.calculate_correlation_matrix(
                        returns_df, CorrelationMethod.PEARSON
                    )
                    
                    # Calculate tail dependencies if we have enough data
                    tail_deps = None
                    if len(returns_df) >= 100:
                        tail_deps = self.correlation_analyzer.calculate_tail_dependencies(returns_df)
                    
                    # Detect correlation regimes
                    regimes = None
                    if len(returns_df) >= 252:  # Need at least 1 year of data
                        regimes = self.correlation_analyzer.detect_correlation_regimes(returns_df)
                    
                    report['correlation_analysis'] = {
                        'summary': correlation_summary,
                        'correlation_matrix': {
                            'pearson': pearson_matrix.matrix.tolist(),
                            'assets': pearson_matrix.assets,
                            'average_correlation': pearson_matrix.get_average_correlation()
                        },
                        'tail_dependencies': [
                            {
                                'pair': f"{td.asset1}_{td.asset2}",
                                'lower_tail': td.lower_tail,
                                'upper_tail': td.upper_tail,
                                'asymmetry': td.asymmetry
                            } for td in tail_deps
                        ] if tail_deps else None,
                        'correlation_regimes': [
                            {
                                'regime_id': r.regime_id,
                                'type': r.regime_type,
                                'start_date': r.start_date.isoformat() if hasattr(r.start_date, 'isoformat') else str(r.start_date),
                                'end_date': r.end_date.isoformat() if hasattr(r.end_date, 'isoformat') else str(r.end_date),
                                'avg_correlation': r.average_correlation
                            } for r in regimes
                        ] if regimes else None,
                        'rolling_correlations': self._calculate_recent_rolling_correlations(returns_df)
                    }
            except Exception as e:
                logger.warning(f"Failed to generate correlation analysis: {str(e)}")
                report['correlation_analysis'] = {'error': str(e)}
        
        # Add Monte Carlo simulation results
        if self.monte_carlo_service and self._historical_returns:
            try:
                # Get portfolio returns for Monte Carlo
                portfolio_returns = pd.Series(
                    [sum(r.values()) for r in self._historical_returns.values()]
                    if isinstance(self._historical_returns, dict) else self._historical_returns
                )
                
                # Calculate Monte Carlo VaR
                mc_var_results = asyncio.run(
                    self.monte_carlo_service.calculate_monte_carlo_var(
                        portfolio_returns,
                        confidence_levels=[0.95, 0.99]
                    )
                )
                
                report['monte_carlo_analysis'] = {
                    'var_results': mc_var_results,
                    'simulation_params': {
                        'n_simulations': self.monte_carlo_service.mc_engine.n_simulations,
                        'time_horizon': self.monte_carlo_service.mc_engine.time_horizon,
                        'confidence_levels': [0.95, 0.99]
                    },
                    'comparison': {
                        'historical_var_95': self.state.risk_metrics.var_95,
                        'monte_carlo_var_95': mc_var_results.get('var_95', 0),
                        'difference': abs(
                            self.state.risk_metrics.var_95 - 
                            mc_var_results.get('var_95', 0)
                        )
                    }
                }
                
                logger.info("Added Monte Carlo analysis to risk report")
            except Exception as e:
                logger.warning(f"Failed to generate Monte Carlo analysis: {str(e)}")
                report['monte_carlo_analysis'] = {'error': str(e)}
        
        return report
    
    def _calculate_recent_rolling_correlations(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate recent rolling correlations for the report."""
        try:
            # Get rolling correlations for the last 30 days
            rolling_corrs = self.correlation_analyzer.calculate_rolling_correlations(
                returns_df.tail(90),  # Use last 90 days
                window=30
            )
            
            # Return the most recent correlation for each pair
            recent_corrs = {}
            for pair, corr_series in rolling_corrs.items():
                clean_series = corr_series.dropna()
                if len(clean_series) > 0:
                    recent_corrs[pair] = float(clean_series.iloc[-1])
            
            return recent_corrs
        except Exception as e:
            logger.warning(f"Failed to calculate recent rolling correlations: {str(e)}")
            return {}
