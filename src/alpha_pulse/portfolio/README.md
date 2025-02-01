# Portfolio Rebalancing Module

This module provides advanced portfolio rebalancing capabilities using modern portfolio optimization techniques. It supports multiple allocation strategies and is exchange-agnostic, working with any exchange that implements the required interface.

## Overview

The portfolio module helps you:
- Analyze current portfolio allocations
- Calculate optimal target allocations using different strategies
- Generate rebalancing recommendations
- Track portfolio metrics and performance

## Allocation Strategies

### 1. Modern Portfolio Theory (MPT)

MPT, developed by Harry Markowitz, optimizes portfolios based on the trade-off between expected return and risk. The strategy:
- Calculates expected returns and covariances from historical data
- Finds the optimal weights that maximize the Sharpe ratio
- Considers constraints like minimum/maximum weights
- Provides efficient frontier analysis

Key features:
- Mean-variance optimization
- Risk-adjusted returns using Sharpe ratio
- Support for custom constraints
- Efficient frontier visualization

### 2. Hierarchical Risk Parity (HRP)

HRP, developed by Marcos Lopez de Prado, provides a more robust alternative to MPT that:
- Uses hierarchical clustering to organize assets
- Allocates weights based on the risk hierarchy
- Doesn't rely on expected returns estimation
- Better handles numerical instabilities

Advantages:
- More robust to estimation errors
- Works well with high-dimensional portfolios
- No matrix inversion required
- Often outperforms traditional optimization

### 3. Black-Litterman Model

The Black-Litterman model combines market equilibrium returns with investor views:
- Uses market capitalization weights as a starting point
- Incorporates investor views with confidence levels
- Provides more intuitive and stable allocations
- Handles missing data and uncertainty

### 4. LLM-Assisted Strategy

Optional LLM enhancement that can wrap any base strategy:
- Analyzes market sentiment and news
- Provides risk assessments
- Suggests allocation adjustments
- Explains reasoning in natural language

## Usage Example

```python
from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from your_exchange_implementation import YourExchange

# Initialize exchange with your credentials
exchange = YourExchange(
    api_key="your_key",
    api_secret="your_secret"
)

# Initialize portfolio manager with config
manager = PortfolioManager("portfolio_config.yaml")

# Get current allocation
current_allocation = manager.get_current_allocation(exchange)

# Execute rebalancing if needed
result = manager.rebalance_portfolio(exchange)

if result['status'] == 'completed':
    print("\nRebalancing completed:")
    print("\nInitial allocation:")
    for asset, weight in result['initial_allocation'].items():
        print(f"{asset}: {weight:.2%}")
        
    print("\nTarget allocation:")
    for asset, weight in result['target_allocation'].items():
        print(f"{asset}: {weight:.2%}")
        
    print("\nExecuted trades:")
    for trade in result['trades']:
        print(f"{trade['type'].upper()} {trade['asset']}: "
              f"${abs(trade['value']):,.2f}")
```

## Exchange Integration

To use your preferred exchange, implement the `IExchange` interface:

```python
from alpha_pulse.portfolio.interfaces import IExchange
from decimal import Decimal

class YourExchange(IExchange):
    def get_account_balances(self) -> Dict[str, Decimal]:
        # Implement getting account balances
        pass
        
    def get_ticker_prices(self, assets: List[str]) -> Dict[str, Decimal]:
        # Implement getting current prices
        pass
        
    def get_portfolio_value(self) -> Decimal:
        # Implement getting total portfolio value
        pass
        
    def execute_trade(self, asset: str, amount: Decimal,
                     side: str, order_type: str = "market") -> Dict:
        # Implement trade execution
        pass
        
    def get_historical_data(self, assets: List[str],
                          start_time: pd.Timestamp,
                          end_time: Optional[pd.Timestamp] = None,
                          interval: str = "1d") -> pd.DataFrame:
        # Implement historical data fetching
        pass
```

## Configuration

The module uses a YAML configuration file for all settings:

```yaml
# Portfolio Type Settings
portfolio_type: "mixed_crypto_stable"
risk_profile: "moderate"
stablecoin_fraction: 0.3
rebalancing_frequency: "daily"

# Risk Management Settings
volatility_target: 0.15
max_drawdown_limit: 0.25
correlation_threshold: 0.7

# Strategy Settings
strategy:
  name: "hierarchical_risk_parity"
  lookback_period: 180
  rebalancing_threshold: 0.1

# Trading Settings
trading:
  execution_style: "twap"
  max_slippage: 0.01
  min_trade_value: 10.0
  base_currency: "USDT"
```

## Risk Considerations

1. Transaction Costs
   - Consider exchange fees when rebalancing
   - Use `min_trade_value` to avoid small trades
   - Balance rebalancing frequency with costs

2. Market Impact
   - Large orders may affect market prices
   - Consider splitting large trades
   - Use limit orders when possible

3. Data Quality
   - Historical data may not predict future behavior
   - Market conditions can change rapidly
   - Consider using longer lookback periods for stability

4. Portfolio Constraints
   - Set reasonable min/max weights
   - Consider liquidity constraints
   - Account for exchange-specific limitations

## Best Practices

1. Regular Monitoring
   - Track portfolio metrics regularly
   - Monitor rebalancing scores
   - Review strategy performance

2. Risk Management
   - Set appropriate position limits
   - Use stop-loss orders
   - Diversify across assets

3. Rebalancing Frequency
   - Balance costs vs. tracking error
   - Consider threshold-based rebalancing
   - Monitor market conditions

4. Testing
   - Start with small allocations
   - Test strategies in different market conditions
   - Monitor trading costs and slippage

## Disclaimer

This module is for informational purposes only. Always:
- Test thoroughly before live trading
- Start with small positions
- Monitor performance closely
- Consider your risk tolerance
- Understand exchange risks and limitations