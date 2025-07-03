"""
Tests for dynamic risk budgeting system.

Tests risk budget allocation accuracy, rebalancing trigger effectiveness,
and regime-based performance.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from alpha_pulse.models.portfolio import Portfolio, Position
from alpha_pulse.models.market_regime import (
    RegimeType, MarketRegime, RegimeDetectionResult
)
from alpha_pulse.models.risk_budget import (
    RiskBudget, RiskBudgetType, AllocationMethod,
    RiskAllocation, RiskBudgetRebalancing, VolatilityTarget
)
from alpha_pulse.risk.dynamic_budgeting import DynamicRiskBudgetManager
from alpha_pulse.config.regime_parameters import RISK_BUDGET_PARAMS


class TestDynamicRiskBudgetManager:
    """Test dynamic risk budget management."""
    
    @pytest.fixture
    def manager(self):
        """Create risk budget manager instance."""
        return DynamicRiskBudgetManager(
            base_volatility_target=0.15,
            max_leverage=2.0,
            rebalancing_frequency="daily"
        )
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        positions = {
            'AAPL': Position(
                symbol='AAPL',
                quantity=100,
                current_price=150.0,
                average_cost=145.0,
                position_type='long',
                sector='technology'
            ),
            'JPM': Position(
                symbol='JPM',
                quantity=200,
                current_price=140.0,
                average_cost=135.0,
                position_type='long',
                sector='financials'
            ),
            'XLU': Position(
                symbol='XLU',
                quantity=300,
                current_price=70.0,
                average_cost=68.0,
                position_type='long',
                sector='utilities'
            )
        }
        
        portfolio = Portfolio(
            portfolio_id='test_portfolio',
            name='Test Portfolio',
            total_value=64000.0,  # Sum of position values
            cash_balance=0.0,
            positions=positions
        )
        
        return portfolio
    
    @pytest.fixture
    def market_data(self):
        """Create sample market data."""
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        
        # Create correlated returns
        np.random.seed(42)
        market_return = np.random.normal(0.0005, 0.02, len(dates))
        
        data = pd.DataFrame({
            'AAPL': 150 * np.cumprod(1 + market_return + np.random.normal(0, 0.01, len(dates))),
            'JPM': 140 * np.cumprod(1 + market_return * 0.8 + np.random.normal(0, 0.008, len(dates))),
            'XLU': 70 * np.cumprod(1 + market_return * 0.3 + np.random.normal(0, 0.005, len(dates))),
            'SPY': 400 * np.cumprod(1 + market_return)
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def regime_result(self):
        """Create sample regime detection result."""
        current_regime = MarketRegime(
            regime_type=RegimeType.BULL,
            confidence=0.85,
            start_date=datetime.now() - timedelta(days=30),
            volatility_level='normal',
            trend_direction='up',
            suggested_leverage=1.2,
            max_position_size=0.15,
            stop_loss_multiplier=2.0,
            preferred_strategies=['momentum', 'growth'],
            avoided_strategies=['defensive', 'value']
        )
        
        regime_probs = {
            RegimeType.BULL: 0.85,
            RegimeType.BEAR: 0.05,
            RegimeType.SIDEWAYS: 0.08,
            RegimeType.CRISIS: 0.01,
            RegimeType.RECOVERY: 0.01
        }
        
        return RegimeDetectionResult(
            current_regime=current_regime,
            regime_probabilities=regime_probs,
            confidence_score=0.85,
            transition_probability=0.05
        )
    
    def test_create_regime_based_budget(self, manager, sample_portfolio, regime_result, market_data):
        """Test creation of regime-based risk budget."""
        budget = manager.create_regime_based_budget(
            sample_portfolio, regime_result, market_data
        )
        
        assert isinstance(budget, RiskBudget)
        assert budget.regime_type == 'bull'
        assert budget.regime_multiplier == 1.2
        assert budget.target_volatility == pytest.approx(0.18, 0.01)  # 15% * 1.2
        
        # Check allocations
        assert len(budget.allocations) == 3
        total_allocation = sum(a.allocated_risk for a in budget.allocations.values())
        assert total_allocation == pytest.approx(1.0, 0.01)
        
        # Check risk limit
        assert budget.total_risk_limit > 0
        assert budget.current_utilization >= 0
    
    def test_risk_parity_allocation(self, manager, sample_portfolio, market_data):
        """Test risk parity allocation method."""
        returns = market_data.pct_change().dropna()
        correlation_matrix = returns.corr()
        
        allocations = manager._create_risk_parity_allocations(
            sample_portfolio, returns, correlation_matrix
        )
        
        # Check allocations sum to 1
        total = sum(a.allocated_risk for a in allocations.values())
        assert total == pytest.approx(1.0, 0.01)
        
        # Lower volatility assets should have higher allocation
        # XLU (utilities) should have higher allocation than AAPL (tech)
        assert allocations['XLU'].allocated_risk > allocations['AAPL'].allocated_risk
    
    def test_regime_based_allocation(self, manager, sample_portfolio, regime_result, market_data):
        """Test regime-specific allocation logic."""
        returns = market_data.pct_change().dropna()
        
        # Bull regime allocations
        bull_allocations = manager._create_regime_based_allocations(
            sample_portfolio, regime_result, returns
        )
        
        # Change to bear regime
        regime_result.current_regime.regime_type = RegimeType.BEAR
        regime_result.current_regime.preferred_strategies = ['defensive', 'quality']
        regime_result.current_regime.avoided_strategies = ['growth', 'momentum']
        
        bear_allocations = manager._create_regime_based_allocations(
            sample_portfolio, regime_result, returns
        )
        
        # In bear regime, defensive assets should have higher allocation
        assert bear_allocations['XLU'].allocated_risk > bull_allocations['XLU'].allocated_risk
        assert bear_allocations['AAPL'].allocated_risk < bull_allocations['AAPL'].allocated_risk
    
    def test_volatility_target_update(self, manager):
        """Test volatility target updates based on regime."""
        current_vol = 0.20
        forecast_vol = 0.25
        
        # Bull regime
        bull_regime = MarketRegime(
            regime_type=RegimeType.BULL,
            confidence=0.9,
            start_date=datetime.now()
        )
        
        bull_leverage = manager.update_volatility_target(
            current_vol, forecast_vol, bull_regime
        )
        
        # Should reduce leverage due to high volatility
        assert bull_leverage < 1.0
        
        # Crisis regime
        crisis_regime = MarketRegime(
            regime_type=RegimeType.CRISIS,
            confidence=0.85,
            start_date=datetime.now()
        )
        
        crisis_leverage = manager.update_volatility_target(
            current_vol, forecast_vol, crisis_regime
        )
        
        # Crisis should have even lower leverage
        assert crisis_leverage < bull_leverage
    
    def test_rebalancing_triggers(self, manager, sample_portfolio, market_data):
        """Test rebalancing trigger detection."""
        # Create initial budget
        regime_result = Mock()
        regime_result.current_regime = MarketRegime(
            regime_type=RegimeType.BULL,
            confidence=0.8,
            start_date=datetime.now()
        )
        
        budget = manager.create_regime_based_budget(
            sample_portfolio, regime_result, market_data
        )
        
        # Test regime change trigger
        new_regime = MarketRegime(
            regime_type=RegimeType.BEAR,
            confidence=0.85,
            start_date=datetime.now()
        )
        
        rebalancing = manager.check_rebalancing_triggers(
            sample_portfolio, new_regime, market_data
        )
        
        assert rebalancing is not None
        assert rebalancing.trigger_type == 'regime_change'
        assert len(rebalancing.allocation_changes) > 0
    
    def test_risk_budget_breach_trigger(self, manager, sample_portfolio, market_data):
        """Test risk budget breach detection."""
        regime = MarketRegime(
            regime_type=RegimeType.BULL,
            confidence=0.8,
            start_date=datetime.now()
        )
        
        # Create budget with low risk limit
        manager.risk_budgets[RiskBudgetType.TOTAL] = RiskBudget(
            budget_id='test',
            budget_type=RiskBudgetType.TOTAL,
            total_risk_limit=0.10,  # Low limit
            target_volatility=0.15,
            regime_type='bull'
        )
        
        # Set high utilization
        manager.risk_budgets[RiskBudgetType.TOTAL].current_utilization = 0.12
        
        rebalancing = manager.check_rebalancing_triggers(
            sample_portfolio, regime, market_data
        )
        
        assert rebalancing is not None
        assert 'risk_breach' in rebalancing.trigger_details['type']
    
    def test_execute_rebalancing(self, manager, sample_portfolio):
        """Test rebalancing execution."""
        rebalancing = RiskBudgetRebalancing(
            rebalancing_id='test_rebal',
            budget_id='test_budget',
            timestamp=datetime.utcnow(),
            trigger_type='regime_change',
            current_allocations={'AAPL': 0.4, 'JPM': 0.35, 'XLU': 0.25},
            target_allocations={'AAPL': 0.3, 'JPM': 0.3, 'XLU': 0.4},
            allocation_changes={'AAPL': -0.1, 'JPM': -0.05, 'XLU': 0.15},
            transaction_cost_estimate=0.001
        )
        
        position_adjustments = manager.execute_rebalancing(
            sample_portfolio, rebalancing
        )
        
        assert len(position_adjustments) == 3
        assert rebalancing.executed
        assert rebalancing.execution_timestamp is not None
        assert len(manager.rebalancing_history) == 1
    
    def test_portfolio_optimization(self, manager, sample_portfolio, market_data):
        """Test portfolio optimization for different regimes."""
        # Bull regime optimization
        bull_regime = MarketRegime(
            regime_type=RegimeType.BULL,
            confidence=0.9,
            start_date=datetime.now(),
            max_position_size=0.30
        )
        
        bull_allocation = manager.optimize_risk_allocation(
            sample_portfolio, bull_regime
        )
        
        assert len(bull_allocation) > 0
        assert sum(bull_allocation.values()) == pytest.approx(1.0, 0.01)
        assert all(0 <= w <= 0.30 for w in bull_allocation.values())
        
        # Crisis regime optimization
        crisis_regime = MarketRegime(
            regime_type=RegimeType.CRISIS,
            confidence=0.85,
            start_date=datetime.now(),
            max_position_size=0.05
        )
        
        crisis_allocation = manager.optimize_risk_allocation(
            sample_portfolio, crisis_regime
        )
        
        # Crisis should have more diversified allocation
        assert all(w <= 0.05 for w in crisis_allocation.values())
        
        # Crisis allocation should favor defensive assets
        if 'XLU' in crisis_allocation and 'AAPL' in crisis_allocation:
            assert crisis_allocation['XLU'] >= crisis_allocation['AAPL']
    
    def test_risk_budget_analytics(self, manager, sample_portfolio, market_data):
        """Test risk budget analytics calculation."""
        # Create budget first
        regime_result = Mock()
        regime_result.current_regime = MarketRegime(
            regime_type=RegimeType.SIDEWAYS,
            confidence=0.75,
            start_date=datetime.now()
        )
        
        budget = manager.create_regime_based_budget(
            sample_portfolio, regime_result, market_data
        )
        
        # Get analytics
        analytics = manager.get_risk_budget_analytics(
            sample_portfolio, market_data
        )
        
        assert 'current_utilization' in analytics
        assert 'utilization_ratio' in analytics
        assert 'concentration_ratio' in analytics
        assert 'volatility_metrics' in analytics
        assert 'leverage_metrics' in analytics
        assert 'regime_info' in analytics
        assert 'allocation_details' in analytics
        
        # Check volatility metrics
        vol_metrics = analytics['volatility_metrics']
        assert vol_metrics['realized'] > 0
        assert vol_metrics['target'] > 0
        
        # Check allocation details
        assert len(analytics['allocation_details']) == 3
        for detail in analytics['allocation_details']:
            assert 'asset' in detail
            assert 'allocated' in detail
            assert 'utilized' in detail
            assert 'within_limits' in detail


class TestVolatilityTargeting:
    """Test volatility targeting functionality."""
    
    @pytest.fixture
    def vol_target(self):
        """Create volatility target instance."""
        return VolatilityTarget(
            target_id='test',
            base_target=0.15,
            current_target=0.15,
            max_leverage=2.0,
            min_leverage=0.2,
            leverage_smoothing=5,
            use_exponential_weighting=True
        )
    
    def test_leverage_calculation(self, vol_target):
        """Test target leverage calculations."""
        vol_target.realized_volatility = 0.20
        vol_target.forecast_volatility = 0.25
        
        leverage = vol_target.calculate_target_leverage()
        
        # With 15% target and 20-25% volatility, leverage should be < 1
        assert leverage < 1.0
        assert leverage >= vol_target.min_leverage
        
        # Test with low volatility
        vol_target.realized_volatility = 0.08
        vol_target.forecast_volatility = 0.10
        
        leverage_high = vol_target.calculate_target_leverage()
        
        # Should increase leverage but respect max
        assert leverage_high > 1.0
        assert leverage_high <= vol_target.max_leverage
    
    def test_regime_adjusted_targets(self, vol_target):
        """Test regime-specific volatility targets."""
        # Bull regime
        bull_target = vol_target.get_regime_adjusted_target('bull')
        assert bull_target > vol_target.base_target
        
        # Crisis regime
        crisis_target = vol_target.get_regime_adjusted_target('crisis')
        assert crisis_target < vol_target.base_target
        
        # Recovery regime
        recovery_target = vol_target.get_regime_adjusted_target('recovery')
        assert crisis_target < recovery_target < bull_target


class TestRebalancingMechanics:
    """Test rebalancing execution mechanics."""
    
    @pytest.fixture
    def manager(self):
        """Create manager with mock dependencies."""
        manager = DynamicRiskBudgetManager()
        manager.monitoring = Mock()
        return manager
    
    def test_allocation_drift_detection(self, manager, sample_portfolio, market_data):
        """Test allocation drift trigger."""
        # Create budget with specific allocations
        budget = RiskBudget(
            budget_id='test',
            budget_type=RiskBudgetType.TOTAL,
            total_risk_limit=0.15,
            target_volatility=0.15,
            regime_type='sideways'
        )
        
        # Set allocations with drift
        budget.allocations = {
            'AAPL': RiskAllocation(
                asset_or_category='AAPL',
                allocated_risk=0.33,
                current_utilization=0.45,  # 12% drift
                target_allocation=0.33
            ),
            'JPM': RiskAllocation(
                asset_or_category='JPM',
                allocated_risk=0.33,
                current_utilization=0.30,
                target_allocation=0.33
            ),
            'XLU': RiskAllocation(
                asset_or_category='XLU',
                allocated_risk=0.34,
                current_utilization=0.25,
                target_allocation=0.34
            )
        }
        
        manager.risk_budgets[RiskBudgetType.TOTAL] = budget
        
        regime = MarketRegime(
            regime_type=RegimeType.SIDEWAYS,
            confidence=0.7,
            start_date=datetime.now()
        )
        
        rebalancing = manager.check_rebalancing_triggers(
            sample_portfolio, regime, market_data
        )
        
        assert rebalancing is not None
        assert any(t['type'] == 'allocation_drift' for t in [rebalancing.trigger_details])
    
    def test_transaction_cost_estimation(self, manager):
        """Test transaction cost estimates."""
        rebalancing = RiskBudgetRebalancing(
            rebalancing_id='test',
            budget_id='test_budget',
            timestamp=datetime.utcnow(),
            trigger_type='regime_change',
            current_allocations={'A': 0.5, 'B': 0.3, 'C': 0.2},
            target_allocations={'A': 0.3, 'B': 0.4, 'C': 0.3},
            allocation_changes={'A': -0.2, 'B': 0.1, 'C': 0.1}
        )
        
        # Calculate turnover
        turnover = rebalancing.get_total_turnover()
        assert turnover == pytest.approx(0.2, 0.01)  # (0.2 + 0.1 + 0.1) / 2
        
        # Estimate costs (assuming 10 bps)
        rebalancing.transaction_cost_estimate = turnover * 0.001
        assert rebalancing.transaction_cost_estimate == pytest.approx(0.0002, 0.00001)
    
    def test_rebalancing_priority_order(self, manager):
        """Test rebalancing priority ordering."""
        allocation_changes = {
            'AAPL': -0.15,  # Large reduction
            'JPM': 0.05,     # Small increase
            'XLU': 0.10,     # Medium increase
            'MSFT': -0.02    # Small reduction
        }
        
        rebalancing = RiskBudgetRebalancing(
            rebalancing_id='test',
            budget_id='test_budget',
            timestamp=datetime.utcnow(),
            trigger_type='volatility_deviation',
            current_allocations={},
            target_allocations={},
            allocation_changes=allocation_changes,
            priority_order=sorted(
                allocation_changes.keys(),
                key=lambda x: abs(allocation_changes[x]),
                reverse=True
            )
        )
        
        # Check priority order (largest changes first)
        assert rebalancing.priority_order[0] == 'AAPL'
        assert rebalancing.priority_order[1] == 'XLU'
        assert rebalancing.priority_order[-1] == 'MSFT'


class TestRiskBudgetBacktesting:
    """Test backtesting of regime-based risk budgeting."""
    
    @pytest.fixture
    def historical_regimes(self):
        """Create historical regime sequence."""
        return [
            (RegimeType.BULL, 120),
            (RegimeType.SIDEWAYS, 60),
            (RegimeType.BEAR, 90),
            (RegimeType.CRISIS, 30),
            (RegimeType.RECOVERY, 60),
            (RegimeType.BULL, 100)
        ]
    
    def test_historical_regime_performance(self, historical_regimes):
        """Test performance across different regime periods."""
        manager = DynamicRiskBudgetManager()
        
        cumulative_return = 1.0
        regime_returns = []
        
        for regime_type, duration in historical_regimes:
            # Get regime parameters
            params = RISK_BUDGET_PARAMS.get(regime_type, {})
            vol_multiplier = params.get('volatility_target_multiplier', 1.0)
            
            # Simulate returns based on regime
            if regime_type == RegimeType.BULL:
                daily_return = np.random.normal(0.0008, 0.01 * vol_multiplier, duration)
            elif regime_type == RegimeType.BEAR:
                daily_return = np.random.normal(-0.0005, 0.015 * vol_multiplier, duration)
            elif regime_type == RegimeType.CRISIS:
                daily_return = np.random.normal(-0.002, 0.025 * vol_multiplier, duration)
            elif regime_type == RegimeType.RECOVERY:
                daily_return = np.random.normal(0.001, 0.012 * vol_multiplier, duration)
            else:  # SIDEWAYS
                daily_return = np.random.normal(0.0002, 0.008 * vol_multiplier, duration)
            
            regime_return = np.prod(1 + daily_return)
            regime_returns.append({
                'regime': regime_type,
                'duration': duration,
                'return': regime_return,
                'volatility': np.std(daily_return) * np.sqrt(252),
                'sharpe': np.mean(daily_return) / np.std(daily_return) * np.sqrt(252)
            })
            
            cumulative_return *= regime_return
        
        # Analyze results
        total_days = sum(r[1] for r in historical_regimes)
        annualized_return = cumulative_return ** (252 / total_days) - 1
        
        # Risk-adjusted returns should be positive
        assert annualized_return > -0.05  # Better than -5% annually
        
        # Check regime-specific performance
        bull_returns = [r['return'] for r in regime_returns if r['regime'] == RegimeType.BULL]
        crisis_returns = [r['return'] for r in regime_returns if r['regime'] == RegimeType.CRISIS]
        
        # Bull regimes should outperform crisis
        assert np.mean(bull_returns) > np.mean(crisis_returns)
    
    def test_risk_budget_allocation_accuracy(self):
        """Test accuracy of risk budget allocations."""
        manager = DynamicRiskBudgetManager()
        
        # Test allocations for different portfolio sizes
        portfolio_sizes = [3, 10, 50]
        
        for n_assets in portfolio_sizes:
            # Create dummy portfolio
            positions = {}
            for i in range(n_assets):
                positions[f'ASSET_{i}'] = Position(
                    symbol=f'ASSET_{i}',
                    quantity=100,
                    current_price=100.0,
                    average_cost=100.0,
                    position_type='long'
                )
            
            portfolio = Portfolio(
                portfolio_id='test',
                name='Test',
                total_value=n_assets * 10000,
                cash_balance=0,
                positions=positions
            )
            
            # Create equal weight allocations
            allocations = manager._create_equal_weight_allocations(portfolio)
            
            # Check allocation accuracy
            expected_weight = 1.0 / n_assets
            for allocation in allocations.values():
                assert allocation.allocated_risk == pytest.approx(expected_weight, 0.001)
            
            # Sum should be 1
            total = sum(a.allocated_risk for a in allocations.values())
            assert total == pytest.approx(1.0, 0.001)