"""
Tests for risk budgeting service.

Tests service integration, real-time updates, and end-to-end workflows.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from alpha_pulse.models.portfolio import Portfolio, Position
from alpha_pulse.models.market_regime import (
    RegimeType, MarketRegime, RegimeDetectionResult
)
from alpha_pulse.models.risk_budget import (
    RiskBudget, RiskBudgetRebalancing, RiskBudgetSnapshot
)
from alpha_pulse.services.risk_budgeting_service import (
    RiskBudgetingService, RiskBudgetingConfig
)
from alpha_pulse.data_pipeline.data_fetcher import DataFetcher
from alpha_pulse.monitoring.alerting import AlertingSystem


class TestRiskBudgetingService:
    """Test risk budgeting service functionality."""
    
    @pytest.fixture
    def config(self):
        """Create service configuration."""
        return RiskBudgetingConfig(
            base_volatility_target=0.15,
            max_leverage=2.0,
            rebalancing_frequency="daily",
            regime_lookback_days=252,
            regime_update_frequency="hourly",
            max_position_size=0.15,
            min_positions=5,
            max_sector_concentration=0.40,
            enable_alerts=True,
            auto_rebalance=False,
            track_performance=True,
            snapshot_frequency="hourly"
        )
    
    @pytest.fixture
    def mock_data_fetcher(self):
        """Create mock data fetcher."""
        fetcher = Mock(spec=DataFetcher)
        
        # Create sample market data
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        market_data = pd.DataFrame({
            'SPY': 400 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates))),
            'VIX': np.random.lognormal(2.9, 0.5, len(dates)),
            'TLT': 100 * np.cumprod(1 + np.random.normal(0.0002, 0.01, len(dates))),
            'GLD': 180 * np.cumprod(1 + np.random.normal(0.0003, 0.015, len(dates)))
        }, index=dates)
        
        async def fetch_historical_data(*args, **kwargs):
            return market_data
        
        fetcher.fetch_historical_data = AsyncMock(side_effect=fetch_historical_data)
        return fetcher
    
    @pytest.fixture
    def mock_alerting(self):
        """Create mock alerting system."""
        alerting = Mock(spec=AlertingSystem)
        alerting.send_alert = AsyncMock()
        return alerting
    
    @pytest.fixture
    async def service(self, config, mock_data_fetcher, mock_alerting):
        """Create risk budgeting service."""
        service = RiskBudgetingService(
            config=config,
            data_fetcher=mock_data_fetcher,
            alerting_system=mock_alerting
        )
        yield service
        # Cleanup
        if service._running:
            await service.stop()
    
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
            ),
            'GLD': Position(
                symbol='GLD',
                quantity=150,
                current_price=180.0,
                average_cost=175.0,
                position_type='long',
                sector='commodities'
            ),
            'MSFT': Position(
                symbol='MSFT',
                quantity=80,
                current_price=350.0,
                average_cost=340.0,
                position_type='long',
                sector='technology'
            )
        }
        
        total_value = sum(
            pos.quantity * pos.current_price 
            for pos in positions.values()
        )
        
        return Portfolio(
            portfolio_id='test_portfolio',
            name='Test Portfolio',
            total_value=total_value,
            cash_balance=5000.0,
            positions=positions
        )
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, service):
        """Test service initialization."""
        assert service.config is not None
        assert service.regime_detector is not None
        assert service.budget_manager is not None
        assert not service._running
        assert len(service._tasks) == 0
    
    @pytest.mark.asyncio
    async def test_start_stop_service(self, service):
        """Test starting and stopping service."""
        # Start service
        await service.start()
        assert service._running
        assert len(service._tasks) == 3  # 3 background tasks
        
        # Give tasks time to start
        await asyncio.sleep(0.1)
        
        # Stop service
        await service.stop()
        assert not service._running
        assert len(service._tasks) == 0
    
    @pytest.mark.asyncio
    async def test_initialize_portfolio_budgets(self, service, sample_portfolio):
        """Test portfolio budget initialization."""
        budget = await service.initialize_portfolio_budgets(sample_portfolio)
        
        assert isinstance(budget, RiskBudget)
        assert budget.regime_type in ['bull', 'bear', 'sideways', 'crisis', 'recovery']
        assert budget.target_volatility > 0
        assert len(budget.allocations) > 0
        
        # Check service state
        assert service.current_budget == budget
        assert service.current_regime is not None
        
        # Check alert was sent
        service.alerting.send_alert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_detect_market_regime(self, service):
        """Test market regime detection."""
        result = await service.detect_market_regime()
        
        assert isinstance(result, RegimeDetectionResult)
        assert result.current_regime is not None
        assert isinstance(result.current_regime.regime_type, RegimeType)
        assert service.current_regime == result.current_regime
        assert service.last_regime_update is not None
    
    @pytest.mark.asyncio
    async def test_regime_change_handling(self, service):
        """Test handling of regime changes."""
        # Set initial regime
        service.current_regime = MarketRegime(
            regime_type=RegimeType.BULL,
            confidence=0.8,
            start_date=datetime.now()
        )
        
        # Mock regime detector to return different regime
        new_regime_result = RegimeDetectionResult(
            current_regime=MarketRegime(
                regime_type=RegimeType.BEAR,
                confidence=0.85,
                start_date=datetime.now()
            ),
            regime_probabilities={},
            confidence_score=0.85
        )
        
        service.regime_detector.detect_regime = Mock(return_value=new_regime_result)
        
        # Detect regime (should trigger change handling)
        await service.detect_market_regime()
        
        # Check alert was sent for regime change
        service.alerting.send_alert.assert_called()
        alert_call = service.alerting.send_alert.call_args
        assert alert_call[1]['level'] == 'warning'
        assert 'Regime Change' in alert_call[1]['title']
    
    @pytest.mark.asyncio
    async def test_check_rebalancing_needs(self, service, sample_portfolio):
        """Test rebalancing needs checking."""
        # Initialize budget first
        await service.initialize_portfolio_budgets(sample_portfolio)
        
        # Check rebalancing
        rebalancing = await service.check_rebalancing_needs(sample_portfolio)
        
        # May or may not need rebalancing depending on market data
        if rebalancing:
            assert isinstance(rebalancing, RiskBudgetRebalancing)
            assert rebalancing.trigger_type in [
                'regime_change', 'risk_breach', 
                'volatility_deviation', 'allocation_drift'
            ]
            assert len(rebalancing.allocation_changes) > 0
    
    @pytest.mark.asyncio
    async def test_execute_rebalancing(self, service, sample_portfolio):
        """Test rebalancing execution."""
        # Create rebalancing recommendation
        rebalancing = RiskBudgetRebalancing(
            rebalancing_id='test_rebal',
            budget_id='test_budget',
            timestamp=datetime.utcnow(),
            trigger_type='regime_change',
            current_allocations={
                'AAPL': 0.25, 'JPM': 0.20, 'XLU': 0.15, 'GLD': 0.20, 'MSFT': 0.20
            },
            target_allocations={
                'AAPL': 0.20, 'JPM': 0.20, 'XLU': 0.25, 'GLD': 0.20, 'MSFT': 0.15
            },
            allocation_changes={
                'AAPL': -0.05, 'JPM': 0.0, 'XLU': 0.10, 'GLD': 0.0, 'MSFT': -0.05
            }
        )
        
        # Execute rebalancing
        adjustments = await service.execute_rebalancing(
            sample_portfolio, rebalancing, dry_run=False
        )
        
        assert len(adjustments) > 0
        assert service.last_rebalance is not None
        
        # Check alert was sent
        service.alerting.send_alert.assert_called()
        alert_call = service.alerting.send_alert.call_args
        assert 'Rebalancing Executed' in alert_call[1]['title']
    
    @pytest.mark.asyncio
    async def test_dry_run_rebalancing(self, service, sample_portfolio):
        """Test dry run rebalancing."""
        rebalancing = RiskBudgetRebalancing(
            rebalancing_id='test_rebal',
            budget_id='test_budget',
            timestamp=datetime.utcnow(),
            trigger_type='volatility_deviation',
            current_allocations={'AAPL': 0.5, 'JPM': 0.5},
            target_allocations={'AAPL': 0.4, 'JPM': 0.6},
            allocation_changes={'AAPL': -0.1, 'JPM': 0.1}
        )
        
        # Execute dry run
        adjustments = await service.execute_rebalancing(
            sample_portfolio, rebalancing, dry_run=True
        )
        
        # Should return changes without execution
        assert adjustments == rebalancing.allocation_changes
        assert service.last_rebalance is None  # Not updated for dry run
    
    @pytest.mark.asyncio
    async def test_update_volatility_target(self, service, sample_portfolio):
        """Test volatility target updates."""
        # Set current regime
        service.current_regime = MarketRegime(
            regime_type=RegimeType.SIDEWAYS,
            confidence=0.75,
            start_date=datetime.now()
        )
        
        # Update volatility target
        new_leverage = await service.update_volatility_target(sample_portfolio)
        
        assert new_leverage > 0
        assert new_leverage <= service.config.max_leverage
    
    @pytest.mark.asyncio
    async def test_get_risk_analytics(self, service, sample_portfolio):
        """Test risk analytics generation."""
        # Initialize budget
        await service.initialize_portfolio_budgets(sample_portfolio)
        
        # Get analytics
        analytics = await service.get_risk_analytics(sample_portfolio)
        
        assert 'budget_metrics' in analytics
        assert 'regime_metrics' in analytics
        assert 'current_regime' in analytics
        assert 'performance' in analytics
        
        # Check current regime info
        regime_info = analytics['current_regime']
        assert 'type' in regime_info
        assert 'confidence' in regime_info
        assert 'duration_days' in regime_info
    
    @pytest.mark.asyncio
    async def test_optimize_portfolio_allocation(self, service, sample_portfolio):
        """Test portfolio optimization."""
        # Optimize allocation
        optimal_allocation = await service.optimize_portfolio_allocation(
            sample_portfolio,
            constraints={
                'max_positions': 10,
                'max_position_size': 0.20,
                'sector_limits': {
                    'technology': 0.35,
                    'financials': 0.30
                }
            }
        )
        
        assert len(optimal_allocation) > 0
        assert sum(optimal_allocation.values()) == pytest.approx(1.0, 0.01)
        
        # Check constraints
        for symbol, weight in optimal_allocation.items():
            assert weight <= 0.20  # Max position size
    
    @pytest.mark.asyncio
    async def test_background_regime_monitoring(self, service):
        """Test background regime monitoring task."""
        # Mock should_update_regime to return True
        service._should_update_regime = Mock(return_value=True)
        
        # Start monitoring task directly
        task = asyncio.create_task(service._regime_monitoring_loop())
        
        # Let it run briefly
        await asyncio.sleep(0.2)
        
        # Stop the task
        service._running = False
        await task
        
        # Check regime was detected
        assert service.regime_detector.detect_regime.called
    
    @pytest.mark.asyncio
    async def test_auto_rebalancing_logic(self, service, sample_portfolio):
        """Test auto-rebalancing decision logic."""
        # Enable auto-rebalancing
        service.config.auto_rebalance = True
        
        # Create urgent rebalancing (risk breach)
        urgent_rebalancing = RiskBudgetRebalancing(
            rebalancing_id='urgent',
            budget_id='test',
            timestamp=datetime.utcnow(),
            trigger_type='risk_breach',
            current_allocations={},
            target_allocations={},
            allocation_changes={'A': 0.1, 'B': -0.1},
            execution_risk='low'
        )
        
        should_auto = service._should_auto_rebalance(urgent_rebalancing)
        assert should_auto  # Risk breach should auto-rebalance
        
        # Create high turnover rebalancing
        high_turnover = RiskBudgetRebalancing(
            rebalancing_id='high_turn',
            budget_id='test',
            timestamp=datetime.utcnow(),
            trigger_type='allocation_drift',
            current_allocations={},
            target_allocations={},
            allocation_changes={'A': 0.4, 'B': -0.4},  # 40% turnover
            execution_risk='low'
        )
        
        should_not_auto = service._should_auto_rebalance(high_turnover)
        assert not should_not_auto  # High turnover should not auto-rebalance
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, service, sample_portfolio):
        """Test performance tracking functionality."""
        # Enable tracking
        service.config.track_performance = True
        
        # Initialize budget
        await service.initialize_portfolio_budgets(sample_portfolio)
        
        # Execute some rebalancings
        for i in range(3):
            rebalancing = RiskBudgetRebalancing(
                rebalancing_id=f'rebal_{i}',
                budget_id='test',
                timestamp=datetime.utcnow(),
                trigger_type='regime_change',
                current_allocations={},
                target_allocations={},
                allocation_changes={'A': 0.05, 'B': -0.05}
            )
            
            await service._track_rebalancing_performance(
                sample_portfolio, rebalancing, {'A': 50, 'B': -50}
            )
        
        # Check performance history
        assert len(service.performance_history) == 3
        
        # Calculate performance metrics
        metrics = service._calculate_performance_metrics()
        assert 'n_rebalances' in metrics
        assert metrics['n_rebalances'] == 3
        assert 'avg_turnover' in metrics
        assert metrics['avg_turnover'] == pytest.approx(0.05, 0.01)


class TestRiskBudgetingServiceIntegration:
    """Integration tests for risk budgeting service."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, service, sample_portfolio):
        """Test complete workflow from initialization to rebalancing."""
        # 1. Start service
        await service.start()
        
        # 2. Initialize portfolio budgets
        initial_budget = await service.initialize_portfolio_budgets(sample_portfolio)
        assert initial_budget is not None
        
        # 3. Wait for regime update
        await asyncio.sleep(0.1)
        
        # 4. Check for rebalancing needs
        rebalancing = await service.check_rebalancing_needs(sample_portfolio)
        
        # 5. Execute rebalancing if needed
        if rebalancing:
            adjustments = await service.execute_rebalancing(
                sample_portfolio, rebalancing
            )
            assert len(adjustments) > 0
        
        # 6. Get analytics
        analytics = await service.get_risk_analytics(sample_portfolio)
        assert analytics is not None
        
        # 7. Stop service
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_crisis_response(self, service, sample_portfolio):
        """Test service response to crisis regime."""
        # Initialize in normal conditions
        await service.initialize_portfolio_budgets(sample_portfolio)
        
        # Simulate crisis detection
        crisis_regime = MarketRegime(
            regime_type=RegimeType.CRISIS,
            confidence=0.9,
            start_date=datetime.now(),
            volatility_level='extreme',
            suggested_leverage=0.2,
            max_position_size=0.05
        )
        
        service.regime_detector.detect_regime = Mock(
            return_value=RegimeDetectionResult(
                current_regime=crisis_regime,
                regime_probabilities={RegimeType.CRISIS: 0.9},
                confidence_score=0.9
            )
        )
        
        # Trigger regime detection
        await service.detect_market_regime()
        
        # Check crisis alert was sent
        alert_calls = service.alerting.send_alert.call_args_list
        crisis_alert = next(
            (call for call in alert_calls if call[1]['level'] == 'critical'),
            None
        )
        assert crisis_alert is not None
        
        # Check rebalancing recommendation
        rebalancing = await service.check_rebalancing_needs(sample_portfolio)
        assert rebalancing is not None
        assert rebalancing.trigger_type == 'regime_change'
        
        # Verify reduced allocations
        if service.current_budget:
            assert service.current_budget.regime_multiplier < 0.5
    
    @pytest.mark.asyncio
    async def test_multi_portfolio_management(self):
        """Test managing multiple portfolios."""
        # Create service
        service = RiskBudgetingService()
        
        # Create multiple portfolios
        portfolios = []
        for i in range(3):
            portfolio = Portfolio(
                portfolio_id=f'portfolio_{i}',
                name=f'Portfolio {i}',
                total_value=100000 * (i + 1),
                cash_balance=5000,
                positions={
                    f'STOCK_{j}': Position(
                        symbol=f'STOCK_{j}',
                        quantity=100,
                        current_price=100.0,
                        average_cost=95.0,
                        position_type='long'
                    )
                    for j in range(5)
                }
            )
            portfolios.append(portfolio)
        
        # Initialize budgets for each portfolio
        budgets = []
        for portfolio in portfolios:
            budget = await service.initialize_portfolio_budgets(portfolio)
            budgets.append(budget)
        
        # Each portfolio should have its own budget
        assert len(budgets) == 3
        assert all(b.budget_id != budgets[0].budget_id for b in budgets[1:])
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, service, sample_portfolio):
        """Test service error handling and recovery."""
        # Initialize service
        await service.start()
        
        # Simulate data fetcher error
        service.data_fetcher.fetch_historical_data.side_effect = Exception(
            "Network error"
        )
        
        # Should handle error gracefully
        result = await service.detect_market_regime()
        assert result is not None  # Should use fallback data
        
        # Restore data fetcher
        service.data_fetcher.fetch_historical_data.side_effect = None
        
        # Service should continue functioning
        analytics = await service.get_risk_analytics(sample_portfolio)
        assert analytics is not None
        
        await service.stop()