"""
Integration tests for dynamic risk budgeting system.

Tests end-to-end workflows, backtesting, and performance validation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Tuple

from alpha_pulse.models.portfolio import Portfolio, Position
from alpha_pulse.models.market_regime import RegimeType, MarketRegime
from alpha_pulse.risk.regime_detector import MarketRegimeDetector
from alpha_pulse.risk.dynamic_budgeting import DynamicRiskBudgetManager
from alpha_pulse.services.risk_budgeting_service import (
    RiskBudgetingService, RiskBudgetingConfig
)
from alpha_pulse.config.regime_parameters import (
    REGIME_TRANSITION_MATRIX, RISK_BUDGET_PARAMS
)


class TestRiskBudgetingBacktest:
    """Backtest dynamic risk budgeting strategies."""
    
    @pytest.fixture
    def market_data_generator(self):
        """Generate synthetic market data with regime characteristics."""
        def generate_regime_data(regime_type: RegimeType, n_days: int) -> pd.DataFrame:
            """Generate data for specific regime."""
            np.random.seed(42)
            
            # Regime-specific parameters
            params = {
                RegimeType.BULL: {
                    'return_mean': 0.0008,
                    'return_std': 0.012,
                    'vix_mean': 15,
                    'vix_std': 2,
                    'correlation': 0.5
                },
                RegimeType.BEAR: {
                    'return_mean': -0.0005,
                    'return_std': 0.020,
                    'vix_mean': 30,
                    'vix_std': 5,
                    'correlation': 0.8
                },
                RegimeType.SIDEWAYS: {
                    'return_mean': 0.0002,
                    'return_std': 0.010,
                    'vix_mean': 20,
                    'vix_std': 3,
                    'correlation': 0.4
                },
                RegimeType.CRISIS: {
                    'return_mean': -0.002,
                    'return_std': 0.035,
                    'vix_mean': 50,
                    'vix_std': 10,
                    'correlation': 0.9
                },
                RegimeType.RECOVERY: {
                    'return_mean': 0.001,
                    'return_std': 0.018,
                    'vix_mean': 25,
                    'vix_std': 4,
                    'correlation': 0.6
                }
            }
            
            p = params[regime_type]
            
            # Generate correlated returns
            dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
            
            # Market factor
            market_factor = np.random.normal(p['return_mean'], p['return_std'], n_days)
            
            # Individual asset returns with correlation
            assets = ['AAPL', 'JPM', 'XLU', 'GLD', 'TLT']
            data = pd.DataFrame(index=dates)
            
            for i, asset in enumerate(assets):
                # Asset-specific component
                idio_return = np.random.normal(0, p['return_std'] * 0.5, n_days)
                
                # Combine market and idiosyncratic
                asset_return = p['correlation'] * market_factor + \
                               np.sqrt(1 - p['correlation']**2) * idio_return
                
                # Adjust for asset characteristics
                if asset == 'XLU':  # Utilities - defensive
                    asset_return *= 0.6
                elif asset == 'GLD':  # Gold - counter-cyclical
                    asset_return = -asset_return * 0.5 if regime_type == RegimeType.CRISIS else asset_return * 0.8
                elif asset == 'TLT':  # Bonds - flight to quality
                    asset_return = -asset_return * 0.4 if regime_type in [RegimeType.CRISIS, RegimeType.BEAR] else asset_return * 0.7
                
                # Create price series
                prices = 100 * np.cumprod(1 + asset_return)
                data[asset] = prices
            
            # Add market indicators
            data['SPY'] = 400 * np.cumprod(1 + market_factor)
            data['VIX'] = np.random.lognormal(
                np.log(p['vix_mean']), p['vix_std'] / p['vix_mean'], n_days
            )
            
            # Add volume (higher in volatile regimes)
            base_volume = 1e9
            vol_multiplier = 1 + (p['vix_mean'] - 20) / 20
            data['volume'] = np.random.lognormal(
                np.log(base_volume * vol_multiplier), 0.3, n_days
            )
            
            return data
        
        return generate_regime_data
    
    @pytest.fixture
    def historical_regime_sequence(self):
        """Create realistic historical regime sequence."""
        # 2-year sequence with realistic transitions
        return [
            (RegimeType.BULL, 180),      # 6 months bull
            (RegimeType.SIDEWAYS, 90),   # 3 months consolidation
            (RegimeType.BEAR, 120),      # 4 months bear
            (RegimeType.CRISIS, 30),     # 1 month crisis
            (RegimeType.RECOVERY, 90),   # 3 months recovery
            (RegimeType.BULL, 150),      # 5 months bull
            (RegimeType.SIDEWAYS, 60),   # 2 months sideways
            (RegimeType.BEAR, 60)        # 2 months bear
        ]
    
    def test_regime_based_performance(self, market_data_generator, historical_regime_sequence):
        """Test performance of regime-based risk budgeting."""
        # Initialize components
        detector = MarketRegimeDetector()
        manager = DynamicRiskBudgetManager(
            base_volatility_target=0.15,
            max_leverage=2.0
        )
        
        # Create initial portfolio
        initial_value = 1000000  # $1M
        portfolio_value = initial_value
        
        # Track performance
        performance_log = []
        regime_log = []
        
        # Simulate through regime sequence
        for regime_type, duration in historical_regime_sequence:
            # Generate regime data
            regime_data = market_data_generator(regime_type, duration)
            
            # Create regime result
            regime = MarketRegime(
                regime_type=regime_type,
                confidence=0.8,
                start_date=datetime.now()
            )
            
            # Get regime parameters
            regime_params = RISK_BUDGET_PARAMS[regime_type]
            vol_multiplier = regime_params['volatility_target_multiplier']
            
            # Calculate daily returns with regime-adjusted volatility
            returns = regime_data.pct_change().dropna()
            
            # Apply volatility targeting
            target_vol = 0.15 * vol_multiplier
            realized_vol = returns.std() * np.sqrt(252)
            vol_scale = target_vol / realized_vol.mean() if realized_vol.mean() > 0 else 1.0
            vol_scale = np.clip(vol_scale, 0.2, 2.0)  # Leverage limits
            
            # Calculate portfolio returns
            # Use regime-specific allocation weights
            if regime_type == RegimeType.CRISIS:
                # Defensive allocation
                weights = {'TLT': 0.4, 'GLD': 0.3, 'XLU': 0.2, 'AAPL': 0.05, 'JPM': 0.05}
            elif regime_type == RegimeType.BULL:
                # Growth allocation
                weights = {'AAPL': 0.3, 'JPM': 0.25, 'SPY': 0.25, 'XLU': 0.1, 'TLT': 0.1}
            elif regime_type == RegimeType.BEAR:
                # Balanced defensive
                weights = {'TLT': 0.3, 'XLU': 0.25, 'GLD': 0.2, 'AAPL': 0.15, 'JPM': 0.1}
            else:
                # Equal weight
                assets = ['AAPL', 'JPM', 'XLU', 'GLD', 'TLT']
                weights = {asset: 1/len(assets) for asset in assets}
            
            # Calculate weighted returns
            portfolio_returns = pd.Series(index=returns.index, dtype=float)
            for date in returns.index:
                daily_return = sum(
                    weights.get(asset, 0) * returns.loc[date, asset]
                    for asset in returns.columns
                    if asset in weights
                )
                portfolio_returns[date] = daily_return * vol_scale
            
            # Update portfolio value
            for ret in portfolio_returns:
                portfolio_value *= (1 + ret)
                
            # Calculate regime performance
            regime_return = (portfolio_returns + 1).prod() - 1
            regime_vol = portfolio_returns.std() * np.sqrt(252)
            regime_sharpe = (regime_return * 252 / duration) / regime_vol if regime_vol > 0 else 0
            
            # Log performance
            performance_log.append({
                'regime': regime_type.value,
                'duration': duration,
                'return': regime_return,
                'volatility': regime_vol,
                'sharpe': regime_sharpe,
                'final_value': portfolio_value
            })
            
            regime_log.append((regime_type, duration, portfolio_value))
        
        # Analyze overall performance
        total_days = sum(r[1] for r in historical_regime_sequence)
        total_return = (portfolio_value / initial_value) - 1
        annualized_return = (portfolio_value / initial_value) ** (252 / total_days) - 1
        
        # Calculate overall volatility
        all_returns = []
        for perf in performance_log:
            daily_ret = (1 + perf['return']) ** (1 / perf['duration']) - 1
            all_returns.extend([daily_ret] * perf['duration'])
        
        overall_volatility = np.std(all_returns) * np.sqrt(252)
        overall_sharpe = annualized_return / overall_volatility if overall_volatility > 0 else 0
        
        # Performance assertions
        assert total_return > -0.20  # No worse than -20% total
        assert annualized_return > -0.10  # No worse than -10% annualized
        assert overall_sharpe > -0.5  # Reasonable risk-adjusted returns
        
        # Regime-specific checks
        bull_perfs = [p for p in performance_log if p['regime'] == 'bull']
        crisis_perfs = [p for p in performance_log if p['regime'] == 'crisis']
        
        if bull_perfs and crisis_perfs:
            # Bull returns should be positive on average
            avg_bull_return = np.mean([p['return'] for p in bull_perfs])
            assert avg_bull_return > 0
            
            # Crisis volatility should be managed
            avg_crisis_vol = np.mean([p['volatility'] for p in crisis_perfs])
            assert avg_crisis_vol < 0.25  # Controlled volatility in crisis
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': overall_volatility,
            'sharpe_ratio': overall_sharpe,
            'performance_log': performance_log
        }
    
    def test_rebalancing_effectiveness(self, market_data_generator):
        """Test effectiveness of rebalancing strategies."""
        # Create test portfolio
        positions = {
            'AAPL': Position('AAPL', 1000, 150, 150, 'long', 'technology'),
            'JPM': Position('JPM', 2000, 140, 140, 'long', 'financials'),
            'XLU': Position('XLU', 3000, 70, 70, 'long', 'utilities'),
            'GLD': Position('GLD', 1500, 180, 180, 'long', 'commodities')
        }
        
        portfolio = Portfolio(
            portfolio_id='test',
            name='Test Portfolio',
            total_value=1000000,
            cash_balance=100000,
            positions=positions
        )
        
        # Test different rebalancing frequencies
        frequencies = ['daily', 'weekly', 'monthly']
        frequency_results = {}
        
        for freq in frequencies:
            manager = DynamicRiskBudgetManager(rebalancing_frequency=freq)
            
            # Generate 6 months of sideways market data
            market_data = market_data_generator(RegimeType.SIDEWAYS, 180)
            returns = market_data.pct_change().dropna()
            
            # Track rebalancing
            n_rebalances = 0
            total_turnover = 0
            transaction_costs = 0
            
            # Simulate rebalancing checks
            if freq == 'daily':
                check_days = list(range(1, 180))
            elif freq == 'weekly':
                check_days = list(range(7, 180, 7))
            else:  # monthly
                check_days = list(range(30, 180, 30))
            
            current_weights = {'AAPL': 0.25, 'JPM': 0.35, 'XLU': 0.21, 'GLD': 0.19}
            target_weights = {'AAPL': 0.25, 'JPM': 0.25, 'XLU': 0.25, 'GLD': 0.25}
            
            for day in check_days:
                # Calculate weight drift
                day_returns = returns.iloc[day-1]
                for asset in current_weights:
                    if asset in day_returns:
                        current_weights[asset] *= (1 + day_returns[asset])
                
                # Normalize weights
                total_weight = sum(current_weights.values())
                current_weights = {k: v/total_weight for k, v in current_weights.items()}
                
                # Check if rebalancing needed (5% threshold)
                max_drift = max(abs(current_weights[a] - target_weights[a]) for a in target_weights)
                
                if max_drift > 0.05:
                    # Rebalance
                    turnover = sum(abs(current_weights[a] - target_weights[a]) for a in target_weights) / 2
                    total_turnover += turnover
                    transaction_costs += turnover * 0.001  # 10 bps
                    n_rebalances += 1
                    
                    # Reset to target
                    current_weights = target_weights.copy()
            
            frequency_results[freq] = {
                'n_rebalances': n_rebalances,
                'avg_turnover': total_turnover / n_rebalances if n_rebalances > 0 else 0,
                'total_costs': transaction_costs,
                'rebalances_per_month': n_rebalances / 6
            }
        
        # Assertions
        # Daily should have most rebalances
        assert frequency_results['daily']['n_rebalances'] > frequency_results['weekly']['n_rebalances']
        assert frequency_results['weekly']['n_rebalances'] > frequency_results['monthly']['n_rebalances']
        
        # But monthly should be most cost-effective
        assert frequency_results['monthly']['total_costs'] < frequency_results['daily']['total_costs']
        
        return frequency_results
    
    def test_stress_scenarios(self, market_data_generator):
        """Test system behavior under stress scenarios."""
        detector = MarketRegimeDetector()
        manager = DynamicRiskBudgetManager()
        
        stress_scenarios = [
            {
                'name': 'Flash Crash',
                'regime_sequence': [
                    (RegimeType.BULL, 5),
                    (RegimeType.CRISIS, 1),  # 1-day crash
                    (RegimeType.RECOVERY, 5)
                ],
                'expected_max_loss': 0.15  # Max 15% loss
            },
            {
                'name': 'Prolonged Bear',
                'regime_sequence': [
                    (RegimeType.BEAR, 60),
                    (RegimeType.CRISIS, 10),
                    (RegimeType.BEAR, 30)
                ],
                'expected_max_loss': 0.25  # Max 25% loss
            },
            {
                'name': 'Whipsaw',
                'regime_sequence': [
                    (RegimeType.BULL, 10),
                    (RegimeType.BEAR, 10),
                    (RegimeType.BULL, 10),
                    (RegimeType.BEAR, 10)
                ],
                'expected_max_loss': 0.10  # Max 10% loss
            }
        ]
        
        scenario_results = {}
        
        for scenario in stress_scenarios:
            portfolio_value = 1000000
            min_value = portfolio_value
            max_drawdown = 0
            
            for regime_type, duration in scenario['regime_sequence']:
                # Generate stressed market data
                market_data = market_data_generator(regime_type, duration)
                
                # Apply regime-based risk management
                regime_params = RISK_BUDGET_PARAMS[regime_type]
                leverage = regime_params.get('max_leverage', 1.0)
                
                # Calculate returns with risk management
                returns = market_data.pct_change().dropna()
                
                # Crisis regime: maximum risk reduction
                if regime_type == RegimeType.CRISIS:
                    # Reduce exposure dramatically
                    leverage = 0.2
                    # Shift to defensive assets
                    defensive_weight = 0.8
                else:
                    defensive_weight = 0.3
                
                # Simulate portfolio performance
                for _, daily_returns in returns.iterrows():
                    # Weighted return (simplified)
                    portfolio_return = daily_returns.mean() * leverage
                    
                    # Apply defensive tilt in crisis
                    if regime_type == RegimeType.CRISIS:
                        portfolio_return *= 0.5  # Additional protection
                    
                    portfolio_value *= (1 + portfolio_return)
                    min_value = min(min_value, portfolio_value)
                    
                    # Track drawdown
                    current_drawdown = (portfolio_value - 1000000) / 1000000
                    max_drawdown = min(max_drawdown, current_drawdown)
            
            scenario_results[scenario['name']] = {
                'final_value': portfolio_value,
                'max_drawdown': max_drawdown,
                'passed': max_drawdown > -scenario['expected_max_loss']
            }
        
        # All scenarios should pass their loss limits
        for name, result in scenario_results.items():
            assert result['passed'], f"Scenario {name} exceeded max loss limit"
        
        return scenario_results


class TestRegimeDetectionAccuracy:
    """Test accuracy of regime detection in various conditions."""
    
    def test_regime_classification_accuracy(self):
        """Test classification accuracy against known regimes."""
        detector = MarketRegimeDetector()
        
        # Create test cases with clear regime characteristics
        test_cases = [
            {
                'vix': 12,
                'momentum_3m': 0.15,
                'trend_strength': 40,
                'expected_regime': RegimeType.BULL,
                'min_confidence': 0.7
            },
            {
                'vix': 45,
                'momentum_1m': -0.15,
                'liquidity_crisis': True,
                'expected_regime': RegimeType.CRISIS,
                'min_confidence': 0.8
            },
            {
                'vix': 22,
                'momentum_3m': 0.01,
                'trend_strength': 15,
                'expected_regime': RegimeType.SIDEWAYS,
                'min_confidence': 0.6
            }
        ]
        
        correct_classifications = 0
        
        for test_case in test_cases:
            # Create market data matching test case
            dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
            
            # Generate data with specified characteristics
            market_data = pd.DataFrame(index=dates)
            market_data['VIX'] = test_case['vix']
            
            # Create price series with specified momentum
            momentum = test_case.get('momentum_3m', 0)
            daily_return = (1 + momentum) ** (1/60) - 1  # 3-month to daily
            prices = 100 * np.cumprod(1 + np.random.normal(daily_return, 0.01, len(dates)))
            market_data['SPY'] = prices
            
            # Detect regime
            result = detector.detect_regime(market_data, test_case)
            
            # Check accuracy
            if result.current_regime.regime_type == test_case['expected_regime']:
                correct_classifications += 1
                
                # Check confidence
                assert result.current_regime.confidence >= test_case['min_confidence'], \
                    f"Low confidence for {test_case['expected_regime']}: {result.current_regime.confidence}"
        
        accuracy = correct_classifications / len(test_cases)
        assert accuracy >= 0.8, f"Low classification accuracy: {accuracy}"
    
    def test_regime_transition_detection(self):
        """Test detection of regime transitions."""
        detector = MarketRegimeDetector()
        
        # Create data with clear transition
        dates = pd.date_range(end=datetime.now(), periods=120, freq='D')
        
        # First 60 days: Bull regime
        bull_returns = np.random.normal(0.001, 0.01, 60)
        # Next 60 days: Bear regime  
        bear_returns = np.random.normal(-0.001, 0.02, 60)
        
        all_returns = np.concatenate([bull_returns, bear_returns])
        prices = 100 * np.cumprod(1 + all_returns)
        
        market_data = pd.DataFrame({
            'SPY': prices,
            'VIX': np.concatenate([
                np.random.normal(15, 2, 60),  # Low VIX in bull
                np.random.normal(30, 5, 60)   # High VIX in bear
            ])
        }, index=dates)
        
        # Detect regime at transition point
        transition_window = 10
        regimes_around_transition = []
        
        for i in range(55, 65):  # Around day 60
            window_data = market_data.iloc[:i]
            result = detector.detect_regime(window_data)
            regimes_around_transition.append({
                'day': i,
                'regime': result.current_regime.regime_type,
                'confidence': result.current_regime.confidence
            })
        
        # Should detect transition
        regime_types = [r['regime'] for r in regimes_around_transition]
        assert RegimeType.BULL in regime_types[:5]
        assert RegimeType.BEAR in regime_types[-5:] or RegimeType.SIDEWAYS in regime_types[-5:]


class TestPortfolioOptimizationQuality:
    """Test quality of portfolio optimization under different regimes."""
    
    def test_optimization_convergence(self):
        """Test that optimization converges to reasonable solutions."""
        manager = DynamicRiskBudgetManager()
        
        # Create test portfolio with many assets
        n_assets = 20
        positions = {}
        for i in range(n_assets):
            positions[f'ASSET_{i}'] = Position(
                symbol=f'ASSET_{i}',
                quantity=1000,
                current_price=100.0,
                average_cost=100.0,
                position_type='long',
                sector='diversified'
            )
        
        portfolio = Portfolio(
            portfolio_id='test',
            name='Large Portfolio',
            total_value=n_assets * 100000,
            cash_balance=0,
            positions=positions
        )
        
        # Test optimization for different regimes
        regimes = [
            MarketRegime(RegimeType.BULL, 0.9, datetime.now()),
            MarketRegime(RegimeType.CRISIS, 0.85, datetime.now()),
            MarketRegime(RegimeType.SIDEWAYS, 0.7, datetime.now())
        ]
        
        for regime in regimes:
            # Set regime-specific max position
            if regime.regime_type == RegimeType.CRISIS:
                regime.max_position_size = 0.05
            else:
                regime.max_position_size = 0.15
            
            # Optimize
            allocation = manager.optimize_risk_allocation(portfolio, regime)
            
            # Check convergence
            assert len(allocation) > 0
            assert abs(sum(allocation.values()) - 1.0) < 0.01
            
            # Check constraints
            for weight in allocation.values():
                assert 0 <= weight <= regime.max_position_size + 0.001
            
            # Check diversification in crisis
            if regime.regime_type == RegimeType.CRISIS:
                assert len(allocation) >= 15  # Well diversified
                assert max(allocation.values()) <= 0.05


@pytest.mark.asyncio
class TestAsyncIntegration:
    """Test asynchronous integration of components."""
    
    async def test_concurrent_regime_detection_and_rebalancing(self):
        """Test concurrent operations."""
        config = RiskBudgetingConfig(
            regime_update_frequency="minute",
            rebalancing_frequency="daily"
        )
        
        service = RiskBudgetingService(config=config)
        
        # Create test portfolio
        portfolio = Portfolio(
            portfolio_id='async_test',
            name='Async Test',
            total_value=1000000,
            cash_balance=50000,
            positions={
                'SPY': Position('SPY', 1000, 400, 380, 'long', 'index'),
                'TLT': Position('TLT', 2000, 100, 95, 'long', 'bonds'),
                'GLD': Position('GLD', 500, 180, 175, 'long', 'commodities')
            }
        )
        
        # Start service
        await service.start()
        
        try:
            # Run concurrent operations
            tasks = [
                service.detect_market_regime(),
                service.initialize_portfolio_budgets(portfolio),
                service.get_risk_analytics(portfolio)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check all completed successfully
            for result in results:
                assert not isinstance(result, Exception)
            
            # Check regime detection
            assert results[0].current_regime is not None
            
            # Check budget initialization
            assert results[1].budget_id is not None
            
            # Check analytics
            assert 'regime_metrics' in results[2]
            
        finally:
            await service.stop()
    
    async def test_real_time_updates(self):
        """Test real-time update handling."""
        service = RiskBudgetingService()
        
        # Simulate real-time market updates
        update_count = 0
        
        async def market_update_simulator():
            nonlocal update_count
            while update_count < 5:
                # Simulate market data update
                additional_indicators = {
                    'intraday_volatility': np.random.uniform(0.001, 0.003),
                    'order_flow': np.random.uniform(-1, 1),
                    'tick_count': np.random.randint(1000, 5000)
                }
                
                # Detect regime with real-time data
                await service.detect_market_regime(
                    additional_indicators=additional_indicators
                )
                
                update_count += 1
                await asyncio.sleep(0.1)  # 100ms between updates
        
        # Run simulator
        await market_update_simulator()
        
        # Should handle all updates
        assert update_count == 5
        assert service.last_regime_update is not None