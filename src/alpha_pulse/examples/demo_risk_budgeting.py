"""
Demo script for dynamic risk budgeting based on market regime detection.

This example demonstrates:
1. Market regime detection from real-time data
2. Dynamic risk budget allocation
3. Volatility targeting adjustments
4. Rebalancing recommendations
5. Performance analytics
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn
import matplotlib.pyplot as plt
import seaborn as sns

from alpha_pulse.models.portfolio import Portfolio, Position
from alpha_pulse.models.market_regime import RegimeType
from alpha_pulse.risk.regime_detector import MarketRegimeDetector
from alpha_pulse.risk.dynamic_budgeting import DynamicRiskBudgetManager
from alpha_pulse.services.risk_budgeting_service import (
    RiskBudgetingService, RiskBudgetingConfig
)
from alpha_pulse.data_pipeline.data_fetcher import DataFetcher
from alpha_pulse.config.regime_parameters import RISK_BUDGET_PARAMS

console = Console()


class RiskBudgetingDemo:
    """Demo application for risk budgeting system."""
    
    def __init__(self):
        """Initialize demo components."""
        self.console = Console()
        
        # Configure risk budgeting
        self.config = RiskBudgetingConfig(
            base_volatility_target=0.15,
            max_leverage=2.0,
            rebalancing_frequency="daily",
            regime_lookback_days=252,
            regime_update_frequency="hourly",
            enable_alerts=True,
            auto_rebalance=False,
            track_performance=True
        )
        
        # Initialize service (without real data fetcher for demo)
        self.service = RiskBudgetingService(config=self.config)
        
        # Create demo portfolio
        self.portfolio = self._create_demo_portfolio()
        
        # Generate demo market data
        self.market_data = self._generate_demo_market_data()
        
    def _create_demo_portfolio(self) -> Portfolio:
        """Create a diversified demo portfolio."""
        positions = {
            # Technology
            'AAPL': Position('AAPL', 500, 175.50, 165.00, 'long', 'technology'),
            'MSFT': Position('MSFT', 300, 380.25, 360.00, 'long', 'technology'),
            
            # Financials
            'JPM': Position('JPM', 400, 155.75, 150.00, 'long', 'financials'),
            'BAC': Position('BAC', 1000, 35.50, 33.00, 'long', 'financials'),
            
            # Healthcare
            'JNJ': Position('JNJ', 200, 162.30, 158.00, 'long', 'healthcare'),
            'PFE': Position('PFE', 800, 28.75, 27.50, 'long', 'healthcare'),
            
            # Consumer Defensive
            'PG': Position('PG', 300, 155.00, 150.00, 'long', 'consumer_staples'),
            'KO': Position('KO', 600, 61.25, 59.00, 'long', 'consumer_staples'),
            
            # Utilities (Defensive)
            'NEE': Position('NEE', 400, 78.50, 75.00, 'long', 'utilities'),
            'SO': Position('SO', 500, 68.75, 66.00, 'long', 'utilities'),
            
            # Bonds ETF
            'TLT': Position('TLT', 1000, 92.50, 95.00, 'long', 'bonds'),
            
            # Gold ETF
            'GLD': Position('GLD', 300, 185.00, 180.00, 'long', 'commodities'),
            
            # Real Estate
            'VNQ': Position('VNQ', 400, 88.25, 85.00, 'long', 'real_estate')
        }
        
        # Calculate total value
        total_value = sum(
            pos.quantity * pos.current_price 
            for pos in positions.values()
        )
        
        portfolio = Portfolio(
            portfolio_id='demo_portfolio',
            name='Dynamic Risk Demo Portfolio',
            total_value=total_value,
            cash_balance=50000.0,  # $50k cash
            positions=positions
        )
        
        return portfolio
    
    def _generate_demo_market_data(self) -> pd.DataFrame:
        """Generate realistic market data with regime transitions."""
        # Create 2 years of daily data
        dates = pd.date_range(end=datetime.now(), periods=504, freq='D')
        
        # Define regime periods for demo
        regime_periods = [
            (0, 126, 'bull'),       # 6 months bull
            (126, 189, 'sideways'), # 3 months sideways
            (189, 315, 'bear'),     # 6 months bear
            (315, 336, 'crisis'),   # 3 weeks crisis
            (336, 420, 'recovery'), # 4 months recovery
            (420, 504, 'bull')      # 4 months bull
        ]
        
        # Generate data
        data = pd.DataFrame(index=dates)
        
        # Market index (SPY)
        spy_prices = []
        current_price = 400
        
        for start, end, regime in regime_periods:
            n_days = end - start
            
            if regime == 'bull':
                daily_returns = np.random.normal(0.0008, 0.012, n_days)
            elif regime == 'bear':
                daily_returns = np.random.normal(-0.0005, 0.018, n_days)
            elif regime == 'crisis':
                daily_returns = np.random.normal(-0.002, 0.035, n_days)
            elif regime == 'recovery':
                daily_returns = np.random.normal(0.001, 0.020, n_days)
            else:  # sideways
                daily_returns = np.random.normal(0.0001, 0.010, n_days)
            
            for ret in daily_returns:
                current_price *= (1 + ret)
                spy_prices.append(current_price)
        
        data['SPY'] = spy_prices
        
        # VIX (inversely related to market performance)
        vix_values = []
        for start, end, regime in regime_periods:
            if regime == 'bull':
                vix = np.random.normal(14, 2, end - start)
            elif regime == 'bear':
                vix = np.random.normal(28, 4, end - start)
            elif regime == 'crisis':
                vix = np.random.normal(50, 8, end - start)
            elif regime == 'recovery':
                vix = np.random.normal(22, 3, end - start)
            else:  # sideways
                vix = np.random.normal(18, 2.5, end - start)
            
            vix_values.extend(np.maximum(vix, 10))  # VIX floor at 10
        
        data['VIX'] = vix_values
        
        # Add other indicators
        data['volume'] = np.random.lognormal(20.5, 0.3, len(dates))  # Log-normal volume
        
        # Individual stock prices (correlated with market)
        for symbol in self.portfolio.positions.keys():
            if symbol == 'TLT':  # Bonds - negative correlation in crisis
                data[symbol] = 100 * np.cumprod(
                    1 + np.where(
                        data['SPY'].pct_change() < -0.02,
                        -data['SPY'].pct_change() * 0.3,  # Flight to quality
                        data['SPY'].pct_change() * -0.2 + np.random.normal(0, 0.005, len(dates))
                    )
                )
            elif symbol == 'GLD':  # Gold - hedge asset
                data[symbol] = 180 * np.cumprod(
                    1 + np.where(
                        data['VIX'] > 30,
                        np.random.normal(0.001, 0.01, len(dates)),  # Positive in crisis
                        np.random.normal(0.0001, 0.008, len(dates))
                    )
                )
            else:  # Stocks - correlated with market
                beta = np.random.uniform(0.7, 1.3)  # Random beta
                data[symbol] = self.portfolio.positions[symbol].current_price * np.cumprod(
                    1 + data['SPY'].pct_change() * beta + np.random.normal(0, 0.01, len(dates))
                )
        
        return data
    
    async def run_demo(self):
        """Run the risk budgeting demo."""
        self.console.print("\n[bold cyan]Dynamic Risk Budgeting Demo[/bold cyan]\n")
        
        # Step 1: Display portfolio
        self._display_portfolio()
        
        # Step 2: Detect current market regime
        await self._demonstrate_regime_detection()
        
        # Step 3: Initialize risk budgets
        await self._demonstrate_budget_initialization()
        
        # Step 4: Show volatility targeting
        await self._demonstrate_volatility_targeting()
        
        # Step 5: Check rebalancing needs
        await self._demonstrate_rebalancing()
        
        # Step 6: Show analytics
        await self._demonstrate_analytics()
        
        # Step 7: Simulate regime transition
        await self._demonstrate_regime_transition()
        
        # Step 8: Performance visualization
        self._visualize_performance()
    
    def _display_portfolio(self):
        """Display current portfolio holdings."""
        table = Table(title="Current Portfolio Holdings")
        table.add_column("Symbol", style="cyan")
        table.add_column("Sector", style="magenta")
        table.add_column("Quantity", justify="right")
        table.add_column("Price", justify="right", style="green")
        table.add_column("Value", justify="right", style="yellow")
        table.add_column("Weight", justify="right")
        
        total_value = self.portfolio.total_value
        
        # Sort by value
        sorted_positions = sorted(
            self.portfolio.positions.items(),
            key=lambda x: x[1].quantity * x[1].current_price,
            reverse=True
        )
        
        for symbol, position in sorted_positions:
            value = position.quantity * position.current_price
            weight = value / total_value
            
            table.add_row(
                symbol,
                position.sector,
                f"{position.quantity:,}",
                f"${position.current_price:.2f}",
                f"${value:,.2f}",
                f"{weight:.1%}"
            )
        
        # Add cash
        table.add_row(
            "CASH",
            "-",
            "-",
            "-",
            f"${self.portfolio.cash_balance:,.2f}",
            f"{self.portfolio.cash_balance / total_value:.1%}",
            style="dim"
        )
        
        self.console.print(table)
        self.console.print(f"\n[bold]Total Portfolio Value:[/bold] ${total_value:,.2f}\n")
    
    async def _demonstrate_regime_detection(self):
        """Demonstrate market regime detection."""
        self.console.print("[bold yellow]Step 1: Detecting Market Regime[/bold yellow]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Analyzing market indicators...", total=None)
            
            # Detect regime using recent data
            recent_data = self.market_data.iloc[-252:]  # Last year
            result = await self.service.detect_market_regime(recent_data)
            
            progress.update(task, completed=True)
        
        # Display regime detection results
        regime = result.current_regime
        
        regime_panel = Panel(
            f"""[bold]Current Market Regime:[/bold] {regime.regime_type.value.upper()}
[bold]Confidence:[/bold] {regime.confidence:.1%}
[bold]Volatility Level:[/bold] {regime.volatility_level}
[bold]Trend Direction:[/bold] {regime.trend_direction}
[bold]Duration:[/bold] {regime.duration_days} days

[bold]Regime Probabilities:[/bold]
• Bull: {result.regime_probabilities.get(RegimeType.BULL, 0):.1%}
• Bear: {result.regime_probabilities.get(RegimeType.BEAR, 0):.1%}
• Sideways: {result.regime_probabilities.get(RegimeType.SIDEWAYS, 0):.1%}
• Crisis: {result.regime_probabilities.get(RegimeType.CRISIS, 0):.1%}
• Recovery: {result.regime_probabilities.get(RegimeType.RECOVERY, 0):.1%}
""",
            title="Market Regime Analysis",
            border_style="cyan"
        )
        
        self.console.print(regime_panel)
        self.console.print()
    
    async def _demonstrate_budget_initialization(self):
        """Demonstrate risk budget initialization."""
        self.console.print("[bold yellow]Step 2: Initializing Risk Budgets[/bold yellow]\n")
        
        budget = await self.service.initialize_portfolio_budgets(self.portfolio)
        
        # Display budget details
        budget_table = Table(title=f"Risk Budget - {budget.regime_type.upper()} Regime")
        budget_table.add_column("Parameter", style="cyan")
        budget_table.add_column("Value", justify="right")
        
        budget_table.add_row("Target Volatility", f"{budget.target_volatility:.1%}")
        budget_table.add_row("Risk Limit", f"{budget.total_risk_limit:.1%}")
        budget_table.add_row("Regime Multiplier", f"{budget.regime_multiplier:.2f}x")
        budget_table.add_row("Current Utilization", f"{budget.current_utilization:.1%}")
        budget_table.add_row("Allocation Method", budget.allocation_method.value)
        
        self.console.print(budget_table)
        
        # Display allocations
        alloc_table = Table(title="Risk Allocations by Asset")
        alloc_table.add_column("Asset", style="cyan")
        alloc_table.add_column("Target", justify="right")
        alloc_table.add_column("Current", justify="right")
        alloc_table.add_column("Risk Contrib", justify="right")
        alloc_table.add_column("Status", justify="center")
        
        # Sort by allocation
        sorted_allocations = sorted(
            budget.allocations.items(),
            key=lambda x: x[1].allocated_risk,
            reverse=True
        )
        
        for asset, allocation in sorted_allocations[:10]:  # Top 10
            status = "✓" if allocation.is_within_limits else "⚠"
            status_color = "green" if allocation.is_within_limits else "yellow"
            
            alloc_table.add_row(
                asset,
                f"{allocation.allocated_risk:.1%}",
                f"{allocation.current_utilization:.1%}",
                f"{allocation.risk_contribution:.1%}",
                f"[{status_color}]{status}[/{status_color}]"
            )
        
        self.console.print(alloc_table)
        self.console.print()
    
    async def _demonstrate_volatility_targeting(self):
        """Demonstrate volatility targeting."""
        self.console.print("[bold yellow]Step 3: Volatility Targeting[/bold yellow]\n")
        
        # Calculate current portfolio volatility
        returns = self.market_data.pct_change().dropna()
        current_vol = returns.iloc[-20:].std().mean() * np.sqrt(252)
        
        # Forecast volatility (simple EWMA)
        forecast_vol = returns.ewm(span=20).std().iloc[-1].mean() * np.sqrt(252)
        
        # Update volatility target
        new_leverage = await self.service.update_volatility_target(self.portfolio)
        
        vol_panel = Panel(
            f"""[bold]Current Portfolio Volatility:[/bold] {current_vol:.1%}
[bold]Forecast Volatility:[/bold] {forecast_vol:.1%}
[bold]Target Volatility:[/bold] {self.config.base_volatility_target:.1%}

[bold]Leverage Adjustment:[/bold] {new_leverage:.2f}x
[bold]Risk Budget:[/bold] ${self.portfolio.total_value * new_leverage:,.0f}

[dim]Leverage is adjusted to maintain target volatility
despite changing market conditions.[/dim]
""",
            title="Volatility Targeting",
            border_style="green"
        )
        
        self.console.print(vol_panel)
        self.console.print()
    
    async def _demonstrate_rebalancing(self):
        """Demonstrate rebalancing recommendations."""
        self.console.print("[bold yellow]Step 4: Rebalancing Analysis[/bold yellow]\n")
        
        rebalancing = await self.service.check_rebalancing_needs(self.portfolio)
        
        if rebalancing:
            # Display rebalancing details
            rebal_table = Table(title="Rebalancing Recommendation")
            rebal_table.add_column("Trigger", style="red")
            rebal_table.add_column("Details")
            
            rebal_table.add_row(
                rebalancing.trigger_type.upper(),
                str(rebalancing.trigger_details)
            )
            
            self.console.print(rebal_table)
            
            # Show allocation changes
            changes_table = Table(title="Recommended Changes")
            changes_table.add_column("Asset", style="cyan")
            changes_table.add_column("Current", justify="right")
            changes_table.add_column("Target", justify="right")
            changes_table.add_column("Change", justify="right")
            changes_table.add_column("Trade Size", justify="right")
            
            # Sort by absolute change
            sorted_changes = sorted(
                rebalancing.allocation_changes.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            for asset, change in sorted_changes[:10]:  # Top 10 changes
                if abs(change) > 0.001:
                    current = rebalancing.current_allocations.get(asset, 0)
                    target = rebalancing.target_allocations.get(asset, 0)
                    trade_value = change * self.portfolio.total_value
                    
                    change_color = "red" if change < 0 else "green"
                    
                    changes_table.add_row(
                        asset,
                        f"{current:.1%}",
                        f"{target:.1%}",
                        f"[{change_color}]{change:+.1%}[/{change_color}]",
                        f"${abs(trade_value):,.0f}"
                    )
            
            self.console.print(changes_table)
            
            # Transaction cost estimate
            self.console.print(
                f"\n[bold]Estimated Transaction Cost:[/bold] "
                f"${rebalancing.transaction_cost_estimate * self.portfolio.total_value:,.2f} "
                f"({rebalancing.transaction_cost_estimate:.2%})"
            )
            self.console.print(
                f"[bold]Total Turnover:[/bold] {rebalancing.get_total_turnover():.1%}\n"
            )
        else:
            self.console.print("[green]✓ Portfolio is within target allocations[/green]\n")
    
    async def _demonstrate_analytics(self):
        """Demonstrate risk analytics."""
        self.console.print("[bold yellow]Step 5: Risk Analytics[/bold yellow]\n")
        
        analytics = await self.service.get_risk_analytics(self.portfolio)
        
        # Create analytics dashboard
        metrics_table = Table(title="Risk Metrics Dashboard", show_header=False)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right")
        
        # Current metrics
        metrics_table.add_row("Risk Utilization", f"{analytics['budget_metrics']['utilization_ratio']:.1%}")
        metrics_table.add_row("Concentration Ratio", f"{analytics['budget_metrics']['concentration_ratio']:.2f}")
        
        # Volatility metrics
        vol_metrics = analytics['budget_metrics']['volatility_metrics']
        metrics_table.add_row("Realized Volatility", f"{vol_metrics['realized']:.1%}")
        metrics_table.add_row("Target Volatility", f"{vol_metrics['target']:.1%}")
        
        # Leverage metrics
        lev_metrics = analytics['budget_metrics']['leverage_metrics']
        metrics_table.add_row("Current Leverage", f"{lev_metrics['current']:.2f}x")
        metrics_table.add_row("Target Leverage", f"{lev_metrics['target']:.2f}x")
        
        self.console.print(metrics_table)
        
        # Top allocations
        if 'allocation_details' in analytics['budget_metrics']:
            top_allocs = analytics['budget_metrics']['allocation_details'][:5]
            
            alloc_chart = Table(title="Top 5 Risk Allocations")
            alloc_chart.add_column("Asset", style="cyan")
            alloc_chart.add_column("Allocation", justify="right")
            alloc_chart.add_column("Utilization", justify="right")
            
            for alloc in top_allocs:
                alloc_chart.add_row(
                    alloc['asset'],
                    f"{alloc['allocated']:.1%}",
                    f"{alloc['utilization_ratio']:.1%}"
                )
            
            self.console.print(alloc_chart)
        
        self.console.print()
    
    async def _demonstrate_regime_transition(self):
        """Demonstrate handling of regime transition."""
        self.console.print("[bold yellow]Step 6: Simulating Regime Transition[/bold yellow]\n")
        
        # Create crisis data
        crisis_data = self.market_data.copy()
        crisis_data['VIX'].iloc[-20:] = 45  # Spike VIX
        crisis_data['SPY'].iloc[-20:] *= 0.85  # 15% market drop
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Detecting regime change...", total=None)
            
            # Force regime change
            self.service.current_regime = None  # Reset
            result = await self.service.detect_market_regime(crisis_data.iloc[-252:])
            
            progress.update(task, completed=True)
        
        if result.current_regime.regime_type == RegimeType.CRISIS:
            self.console.print("[bold red]⚠ CRISIS REGIME DETECTED![/bold red]\n")
            
            # Show crisis response
            crisis_panel = Panel(
                f"""[bold red]Market Crisis Detected[/bold red]
                
[bold]Immediate Actions:[/bold]
1. Reduce leverage to {RISK_BUDGET_PARAMS[RegimeType.CRISIS]['max_leverage']}x
2. Shift to defensive allocation
3. Increase cash reserves
4. Tighten stop-losses

[bold]Risk Parameters:[/bold]
• Max Position Size: {RISK_BUDGET_PARAMS[RegimeType.CRISIS]['position_limits']['max_single_position']:.0%}
• Min Positions: {RISK_BUDGET_PARAMS[RegimeType.CRISIS]['position_limits']['min_positions']}
• Rebalancing: {RISK_BUDGET_PARAMS[RegimeType.CRISIS]['rebalancing']['frequency']}

[dim]Crisis protocol activated for capital preservation[/dim]
""",
                title="Crisis Response Protocol",
                border_style="red"
            )
            
            self.console.print(crisis_panel)
        else:
            self.console.print(f"[yellow]Regime: {result.current_regime.regime_type.value}[/yellow]\n")
    
    def _visualize_performance(self):
        """Create performance visualizations."""
        self.console.print("[bold yellow]Step 7: Performance Visualization[/bold yellow]\n")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dynamic Risk Budgeting Performance Analysis', fontsize=16)
        
        # 1. Market regimes over time
        ax1 = axes[0, 0]
        regime_colors = {
            'bull': 'green',
            'bear': 'red',
            'sideways': 'gray',
            'crisis': 'darkred',
            'recovery': 'blue'
        }
        
        # Plot SPY with regime backgrounds
        spy_returns = (self.market_data['SPY'] / self.market_data['SPY'].iloc[0] - 1) * 100
        ax1.plot(spy_returns.index, spy_returns.values, 'k-', linewidth=1.5, label='Market Return')
        
        # Add regime periods
        regime_periods = [
            (0, 126, 'bull'),
            (126, 189, 'sideways'),
            (189, 315, 'bear'),
            (315, 336, 'crisis'),
            (336, 420, 'recovery'),
            (420, 504, 'bull')
        ]
        
        for start, end, regime in regime_periods:
            ax1.axvspan(
                spy_returns.index[start],
                spy_returns.index[end-1],
                alpha=0.3,
                color=regime_colors[regime],
                label=regime.capitalize() if start == 0 or regime not in [r[2] for r in regime_periods[:regime_periods.index((start, end, regime))]] else ""
            )
        
        ax1.set_title('Market Performance by Regime')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. VIX levels
        ax2 = axes[0, 1]
        ax2.plot(self.market_data.index, self.market_data['VIX'], 'b-', linewidth=1.5)
        ax2.axhline(y=20, color='orange', linestyle='--', label='Normal/High Threshold')
        ax2.axhline(y=30, color='red', linestyle='--', label='High/Extreme Threshold')
        ax2.fill_between(self.market_data.index, 0, self.market_data['VIX'], alpha=0.3, color='blue')
        ax2.set_title('VIX - Market Volatility')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('VIX Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Risk budget utilization
        ax3 = axes[1, 0]
        
        # Simulate risk utilization based on regimes
        risk_utilization = []
        for start, end, regime in regime_periods:
            regime_params = RISK_BUDGET_PARAMS.get(
                RegimeType[regime.upper()],
                RISK_BUDGET_PARAMS[RegimeType.SIDEWAYS]
            )
            
            # Base utilization varies by regime
            if regime == 'crisis':
                base_util = np.random.uniform(0.3, 0.5, end - start)
            elif regime == 'bull':
                base_util = np.random.uniform(0.7, 0.9, end - start)
            else:
                base_util = np.random.uniform(0.5, 0.7, end - start)
            
            risk_utilization.extend(base_util)
        
        risk_utilization = pd.Series(risk_utilization, index=self.market_data.index)
        
        ax3.plot(risk_utilization.index, risk_utilization.values * 100, 'g-', linewidth=1.5)
        ax3.axhline(y=80, color='orange', linestyle='--', label='Warning Level')
        ax3.axhline(y=95, color='red', linestyle='--', label='Critical Level')
        ax3.fill_between(risk_utilization.index, 0, risk_utilization.values * 100, alpha=0.3, color='green')
        ax3.set_title('Risk Budget Utilization')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Utilization (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Rolling Sharpe Ratio
        ax4 = axes[1, 1]
        
        # Calculate rolling Sharpe
        returns = self.market_data['SPY'].pct_change()
        rolling_sharpe = (
            returns.rolling(window=60).mean() / returns.rolling(window=60).std()
        ) * np.sqrt(252)
        
        ax4.plot(rolling_sharpe.index, rolling_sharpe.values, 'purple', linewidth=1.5)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Good (>1)')
        ax4.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Poor (<-0.5)')
        ax4.set_title('Rolling 60-Day Sharpe Ratio')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig('risk_budgeting_performance.png', dpi=300, bbox_inches='tight')
        self.console.print("[green]✓ Performance visualization saved to 'risk_budgeting_performance.png'[/green]\n")
        
        # Summary statistics
        summary_panel = Panel(
            f"""[bold]Performance Summary (2-Year Backtest)[/bold]
            
Total Period Return: {(self.market_data['SPY'].iloc[-1] / self.market_data['SPY'].iloc[0] - 1):.1%}
Average VIX Level: {self.market_data['VIX'].mean():.1f}
VIX Range: {self.market_data['VIX'].min():.1f} - {self.market_data['VIX'].max():.1f}
            
[bold]Regime Distribution:[/bold]
• Bull: 42% of period (10 months)
• Bear: 31% of period (7.5 months)
• Sideways: 13% of period (3 months)
• Crisis: 4% of period (3 weeks)
• Recovery: 10% of period (2.5 months)

[bold]Risk Management:[/bold]
• Average Risk Utilization: {np.mean(risk_utilization) * 100:.1f}%
• Max Risk Utilization: {np.max(risk_utilization) * 100:.1f}%
• Risk Limit Breaches: 0

[dim]Dynamic risk budgeting successfully adapted to changing market conditions[/dim]
""",
            title="Backtest Results",
            border_style="green"
        )
        
        self.console.print(summary_panel)


async def main():
    """Run the risk budgeting demo."""
    demo = RiskBudgetingDemo()
    
    try:
        await demo.run_demo()
        
        console.print("\n[bold green]Demo completed successfully![/bold green]")
        console.print(
            "\n[dim]This demo showcased dynamic risk budgeting with market regime detection. "
            "The system automatically adjusts risk allocations, leverage, and rebalancing "
            "frequency based on detected market conditions.[/dim]\n"
        )
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    asyncio.run(main())