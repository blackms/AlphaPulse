"""
Risk budgeting service for portfolio management.

Provides high-level interface for dynamic risk budgeting, regime detection,
and portfolio optimization.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from alpha_pulse.models.portfolio import Portfolio
from alpha_pulse.models.market_regime import (
    RegimeDetectionResult, MarketRegime, RegimeType
)
from alpha_pulse.models.risk_budget import (
    RiskBudget, RiskBudgetRebalancing, RiskBudgetSnapshot
)
from alpha_pulse.risk.regime_detector import MarketRegimeDetector
from alpha_pulse.risk.dynamic_budgeting import DynamicRiskBudgetManager
from alpha_pulse.data_pipeline.data_fetcher import DataFetcher
from alpha_pulse.monitoring.alerting import AlertingSystem

logger = logging.getLogger(__name__)


@dataclass
class RiskBudgetingConfig:
    """Configuration for risk budgeting service."""
    base_volatility_target: float = 0.15
    max_leverage: float = 2.0
    rebalancing_frequency: str = "daily"
    
    # Regime detection
    regime_lookback_days: int = 252
    regime_update_frequency: str = "hourly"
    
    # Risk limits
    max_position_size: float = 0.15
    min_positions: int = 5
    max_sector_concentration: float = 0.40
    
    # Monitoring
    enable_alerts: bool = True
    auto_rebalance: bool = False
    
    # Performance tracking
    track_performance: bool = True
    snapshot_frequency: str = "hourly"


class RiskBudgetingService:
    """Service for dynamic risk budgeting and portfolio management."""
    
    def __init__(
        self,
        config: Optional[RiskBudgetingConfig] = None,
        data_fetcher: Optional[DataFetcher] = None,
        alerting_system: Optional[AlertingSystem] = None
    ):
        """Initialize risk budgeting service."""
        self.config = config or RiskBudgetingConfig()
        self.data_fetcher = data_fetcher
        self.alerting = alerting_system
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector()
        self.budget_manager = DynamicRiskBudgetManager(
            base_volatility_target=self.config.base_volatility_target,
            max_leverage=self.config.max_leverage,
            rebalancing_frequency=self.config.rebalancing_frequency
        )
        
        # State tracking
        self.current_regime: Optional[MarketRegime] = None
        self.current_budget: Optional[RiskBudget] = None
        self.last_regime_update: Optional[datetime] = None
        self.last_rebalance: Optional[datetime] = None
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
    async def start(self):
        """Start risk budgeting service."""
        logger.info("Starting risk budgeting service")
        self._running = True
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._regime_monitoring_loop()),
            asyncio.create_task(self._rebalancing_loop()),
            asyncio.create_task(self._performance_tracking_loop())
        ]
        
    async def stop(self):
        """Stop risk budgeting service."""
        logger.info("Stopping risk budgeting service")
        self._running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
    
    async def initialize_portfolio_budgets(
        self,
        portfolio: Portfolio
    ) -> RiskBudget:
        """Initialize risk budgets for portfolio."""
        logger.info(f"Initializing risk budgets for portfolio {portfolio.portfolio_id}")
        
        # Fetch market data
        market_data = await self._fetch_market_data(portfolio)
        
        # Detect current regime
        regime_result = await self.detect_market_regime(market_data)
        self.current_regime = regime_result.current_regime
        
        # Create initial budget
        budget = self.budget_manager.create_regime_based_budget(
            portfolio, regime_result, market_data
        )
        
        self.current_budget = budget
        
        # Send alert
        if self.alerting:
            await self.alerting.send_alert(
                level="info",
                title="Risk Budget Initialized",
                message=f"Created {budget.regime_type} regime budget with "
                       f"{budget.target_volatility:.1%} volatility target"
            )
        
        return budget
    
    async def detect_market_regime(
        self,
        market_data: Optional[pd.DataFrame] = None,
        additional_indicators: Optional[Dict[str, float]] = None
    ) -> RegimeDetectionResult:
        """Detect current market regime."""
        if market_data is None:
            # Fetch default market data
            market_data = await self._fetch_default_market_data()
        
        # Run regime detection
        regime_result = self.regime_detector.detect_regime(
            market_data,
            additional_indicators
        )
        
        # Check for regime change
        if self.current_regime and \
           regime_result.current_regime.regime_type != self.current_regime.regime_type:
            await self._handle_regime_change(
                self.current_regime,
                regime_result.current_regime
            )
        
        self.current_regime = regime_result.current_regime
        self.last_regime_update = datetime.utcnow()
        
        return regime_result
    
    async def check_rebalancing_needs(
        self,
        portfolio: Portfolio
    ) -> Optional[RiskBudgetRebalancing]:
        """Check if portfolio needs rebalancing."""
        if not self.current_budget or not self.current_regime:
            return None
        
        # Fetch latest market data
        market_data = await self._fetch_market_data(portfolio)
        
        # Check triggers
        rebalancing = self.budget_manager.check_rebalancing_triggers(
            portfolio,
            self.current_regime,
            market_data
        )
        
        if rebalancing:
            # Add performance estimates
            await self._estimate_rebalancing_impact(
                portfolio,
                rebalancing,
                market_data
            )
            
            # Send alert
            if self.alerting:
                await self.alerting.send_alert(
                    level="warning",
                    title="Rebalancing Recommended",
                    message=f"Trigger: {rebalancing.trigger_type}, "
                           f"Turnover: {rebalancing.get_total_turnover():.1%}"
                )
        
        return rebalancing
    
    async def execute_rebalancing(
        self,
        portfolio: Portfolio,
        rebalancing: RiskBudgetRebalancing,
        dry_run: bool = False
    ) -> Dict[str, float]:
        """Execute portfolio rebalancing."""
        logger.info(f"Executing rebalancing {rebalancing.rebalancing_id} "
                   f"(dry_run={dry_run})")
        
        if dry_run:
            # Return proposed changes without execution
            return rebalancing.allocation_changes
        
        # Execute rebalancing
        position_adjustments = self.budget_manager.execute_rebalancing(
            portfolio,
            rebalancing
        )
        
        self.last_rebalance = datetime.utcnow()
        
        # Track performance
        await self._track_rebalancing_performance(
            portfolio,
            rebalancing,
            position_adjustments
        )
        
        # Send confirmation
        if self.alerting:
            await self.alerting.send_alert(
                level="info",
                title="Rebalancing Executed",
                message=f"Adjusted {len(position_adjustments)} positions, "
                       f"Turnover: {rebalancing.get_total_turnover():.1%}"
            )
        
        return position_adjustments
    
    async def update_volatility_target(
        self,
        portfolio: Portfolio
    ) -> float:
        """Update volatility target based on current conditions."""
        if not self.current_regime:
            return self.config.base_volatility_target
        
        # Fetch market data
        market_data = await self._fetch_market_data(portfolio)
        returns = market_data.pct_change().dropna()
        
        # Calculate current and forecast volatility
        current_vol = returns.std().mean() * np.sqrt(252)
        
        # Simple EWMA forecast
        vol_series = returns.rolling(20).std() * np.sqrt(252)
        forecast_vol = vol_series.ewm(span=10).mean().iloc[-1].mean()
        
        # Update target
        new_leverage = self.budget_manager.update_volatility_target(
            current_vol,
            forecast_vol,
            self.current_regime
        )
        
        logger.info(f"Updated leverage target to {new_leverage:.2f}")
        
        return new_leverage
    
    async def get_risk_analytics(
        self,
        portfolio: Portfolio
    ) -> Dict[str, Any]:
        """Get comprehensive risk analytics."""
        # Fetch market data
        market_data = await self._fetch_market_data(portfolio)
        
        # Get budget analytics
        budget_analytics = self.budget_manager.get_risk_budget_analytics(
            portfolio,
            market_data
        )
        
        # Get regime analytics
        regime_analytics = self.regime_detector.get_regime_analytics()
        
        # Combine analytics
        analytics = {
            "budget_metrics": budget_analytics,
            "regime_metrics": regime_analytics,
            "current_regime": {
                "type": self.current_regime.regime_type.value if self.current_regime else "unknown",
                "confidence": self.current_regime.confidence if self.current_regime else 0,
                "duration_days": self.current_regime.duration_days if self.current_regime else 0
            },
            "performance": self._calculate_performance_metrics()
        }
        
        return analytics
    
    async def optimize_portfolio_allocation(
        self,
        portfolio: Portfolio,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Optimize portfolio allocation for current regime."""
        if not self.current_regime:
            # Detect regime first
            regime_result = await self.detect_market_regime()
            self.current_regime = regime_result.current_regime
        
        # Add default constraints
        if constraints is None:
            constraints = {}
        
        constraints.update({
            "max_positions": max(self.config.min_positions, len(portfolio.positions)),
            "max_position_size": self.config.max_position_size,
            "sector_limits": self._get_sector_limits()
        })
        
        # Run optimization
        optimal_allocation = self.budget_manager.optimize_risk_allocation(
            portfolio,
            self.current_regime,
            constraints
        )
        
        return optimal_allocation
    
    async def _regime_monitoring_loop(self):
        """Background task for regime monitoring."""
        while self._running:
            try:
                # Check update frequency
                if self._should_update_regime():
                    # Detect regime
                    await self.detect_market_regime()
                    
                    # Log regime probabilities
                    if hasattr(self.regime_detector, 'current_regime'):
                        logger.info(f"Current regime: {self.current_regime.regime_type}, "
                                   f"Confidence: {self.current_regime.confidence:.2f}")
                
                # Sleep based on frequency
                await asyncio.sleep(self._get_update_interval())
                
            except Exception as e:
                logger.error(f"Error in regime monitoring: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _rebalancing_loop(self):
        """Background task for rebalancing checks."""
        while self._running:
            try:
                if self.config.auto_rebalance and self._should_check_rebalancing():
                    # Get active portfolios (simplified)
                    # In practice, would get from portfolio service
                    portfolio = None  # Placeholder
                    
                    if portfolio:
                        rebalancing = await self.check_rebalancing_needs(portfolio)
                        
                        if rebalancing and self._should_auto_rebalance(rebalancing):
                            await self.execute_rebalancing(portfolio, rebalancing)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in rebalancing loop: {e}")
                await asyncio.sleep(60)
    
    async def _performance_tracking_loop(self):
        """Background task for performance tracking."""
        while self._running:
            try:
                if self.config.track_performance:
                    await self._update_performance_metrics()
                
                # Sleep based on snapshot frequency
                interval = self._get_snapshot_interval()
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in performance tracking: {e}")
                await asyncio.sleep(300)
    
    async def _fetch_market_data(
        self,
        portfolio: Portfolio
    ) -> pd.DataFrame:
        """Fetch market data for portfolio assets."""
        if not self.data_fetcher:
            # Return dummy data for testing
            return self._generate_dummy_market_data(portfolio)
        
        # Get unique symbols
        symbols = list(set(pos.symbol for pos in portfolio.positions.values()))
        
        # Add market indices
        symbols.extend(['SPY', 'VIX', 'TLT', 'GLD'])
        
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.regime_lookback_days)
        
        market_data = await self.data_fetcher.fetch_historical_data(
            symbols,
            start_date,
            end_date
        )
        
        return market_data
    
    async def _fetch_default_market_data(self) -> pd.DataFrame:
        """Fetch default market data for regime detection."""
        if not self.data_fetcher:
            return self._generate_dummy_market_data(None)
        
        # Default symbols for regime detection
        symbols = ['SPY', 'VIX', 'TLT', 'GLD', 'DXY', 'HYG', 'IWM']
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.regime_lookback_days)
        
        market_data = await self.data_fetcher.fetch_historical_data(
            symbols,
            start_date,
            end_date
        )
        
        return market_data
    
    def _generate_dummy_market_data(
        self,
        portfolio: Optional[Portfolio]
    ) -> pd.DataFrame:
        """Generate dummy market data for testing."""
        dates = pd.date_range(
            end=datetime.now(),
            periods=self.config.regime_lookback_days,
            freq='D'
        )
        
        # Generate synthetic data
        np.random.seed(42)
        
        symbols = ['SPY', 'VIX', 'TLT', 'GLD']
        if portfolio:
            symbols.extend([pos.symbol for pos in portfolio.positions.values()])
        
        data = {}
        for symbol in symbols:
            if symbol == 'VIX':
                # VIX with mean reversion
                vix = [20]
                for _ in range(len(dates) - 1):
                    change = np.random.normal(-0.1 * (vix[-1] - 20) / 20, 2)
                    vix.append(max(10, min(80, vix[-1] + change)))
                data[symbol] = vix
            else:
                # Price series with trend and volatility
                returns = np.random.normal(0.0005, 0.02, len(dates))
                prices = 100 * np.cumprod(1 + returns)
                data[symbol] = prices
        
        return pd.DataFrame(data, index=dates)
    
    async def _handle_regime_change(
        self,
        old_regime: MarketRegime,
        new_regime: MarketRegime
    ):
        """Handle regime transition."""
        logger.info(f"Regime change: {old_regime.regime_type} -> {new_regime.regime_type}")
        
        # Send alert
        if self.alerting:
            severity = "critical" if new_regime.regime_type == RegimeType.CRISIS else "warning"
            
            await self.alerting.send_alert(
                level=severity,
                title="Market Regime Change Detected",
                message=f"Regime changed from {old_regime.regime_type.value} to "
                       f"{new_regime.regime_type.value}. "
                       f"Confidence: {new_regime.confidence:.2f}"
            )
        
        # Update risk parameters
        if self.current_budget:
            self.current_budget.regime_type = new_regime.regime_type.value
            self.current_budget.regime_multiplier = new_regime.suggested_leverage
    
    async def _estimate_rebalancing_impact(
        self,
        portfolio: Portfolio,
        rebalancing: RiskBudgetRebalancing,
        market_data: pd.DataFrame
    ):
        """Estimate impact of rebalancing."""
        # Calculate expected volatility change
        current_vol = self.budget_manager._calculate_portfolio_volatility(
            portfolio,
            market_data.pct_change().dropna()
        )
        
        # Estimate new volatility (simplified)
        vol_change = 0.0
        for asset, change in rebalancing.allocation_changes.items():
            if asset in market_data.columns:
                asset_vol = market_data[asset].pct_change().std() * np.sqrt(252)
                vol_change += abs(change) * (asset_vol - current_vol)
        
        rebalancing.expected_volatility_change = vol_change
        
        # Estimate return impact (simplified)
        rebalancing.expected_return_impact = -rebalancing.transaction_cost_estimate
    
    async def _track_rebalancing_performance(
        self,
        portfolio: Portfolio,
        rebalancing: RiskBudgetRebalancing,
        adjustments: Dict[str, float]
    ):
        """Track rebalancing performance."""
        performance_record = {
            "timestamp": datetime.utcnow(),
            "rebalancing_id": rebalancing.rebalancing_id,
            "trigger": rebalancing.trigger_type,
            "n_adjustments": len(adjustments),
            "turnover": rebalancing.get_total_turnover(),
            "portfolio_value": portfolio.total_value
        }
        
        self.performance_history.append(performance_record)
        
        # Keep limited history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    async def _update_performance_metrics(self):
        """Update performance metrics."""
        # This would calculate and store performance metrics
        # For now, just log
        logger.debug("Updating performance metrics")
    
    def _should_update_regime(self) -> bool:
        """Check if regime should be updated."""
        if not self.last_regime_update:
            return True
        
        interval = self._get_update_interval()
        return (datetime.utcnow() - self.last_regime_update).seconds >= interval
    
    def _should_check_rebalancing(self) -> bool:
        """Check if rebalancing check is due."""
        if not self.last_rebalance:
            return True
        
        if self.config.rebalancing_frequency == "daily":
            return (datetime.utcnow() - self.last_rebalance).days >= 1
        elif self.config.rebalancing_frequency == "weekly":
            return (datetime.utcnow() - self.last_rebalance).days >= 7
        else:
            return True
    
    def _should_auto_rebalance(
        self,
        rebalancing: RiskBudgetRebalancing
    ) -> bool:
        """Check if auto-rebalancing should proceed."""
        # Check urgency
        if rebalancing.trigger_type == "risk_breach":
            return True
        
        # Check turnover threshold
        if rebalancing.get_total_turnover() > 0.30:
            return False  # Too much turnover for auto
        
        # Check execution risk
        if rebalancing.execution_risk == "high":
            return False
        
        return True
    
    def _get_update_interval(self) -> int:
        """Get regime update interval in seconds."""
        intervals = {
            "real-time": 60,
            "minute": 60,
            "hourly": 3600,
            "daily": 86400
        }
        
        return intervals.get(self.config.regime_update_frequency, 3600)
    
    def _get_snapshot_interval(self) -> int:
        """Get snapshot interval in seconds."""
        intervals = {
            "minute": 60,
            "hourly": 3600,
            "daily": 86400
        }
        
        return intervals.get(self.config.snapshot_frequency, 3600)
    
    def _get_sector_limits(self) -> Dict[str, float]:
        """Get sector concentration limits."""
        # Default sector limits
        return {
            "technology": self.config.max_sector_concentration,
            "financials": self.config.max_sector_concentration,
            "healthcare": self.config.max_sector_concentration,
            "energy": self.config.max_sector_concentration * 0.7,  # Lower for volatile sectors
            "utilities": self.config.max_sector_concentration * 1.2,  # Higher for defensive
            "default": self.config.max_sector_concentration
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from history."""
        if not self.performance_history:
            return {}
        
        # Calculate rebalancing statistics
        n_rebalances = len(self.performance_history)
        
        if n_rebalances > 0:
            avg_turnover = np.mean([r["turnover"] for r in self.performance_history])
            
            # Time between rebalances
            if n_rebalances > 1:
                intervals = []
                for i in range(1, n_rebalances):
                    interval = (self.performance_history[i]["timestamp"] - 
                               self.performance_history[i-1]["timestamp"]).days
                    intervals.append(interval)
                avg_interval = np.mean(intervals)
            else:
                avg_interval = 0
            
            return {
                "n_rebalances": n_rebalances,
                "avg_turnover": avg_turnover,
                "avg_interval_days": avg_interval,
                "last_rebalance": self.performance_history[-1]["timestamp"]
            }
        
        return {}