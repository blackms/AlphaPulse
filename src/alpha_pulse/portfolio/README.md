# Portfolio Rebalancing Module

This module provides advanced portfolio rebalancing capabilities using modern portfolio optimization techniques. It supports multiple allocation strategies and integrates with various cryptocurrency exchanges.

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

## Usage Example

```python
from alpha_pulse.exchange_conn.binance import BinanceConnector
from alpha_pulse.portfolio.analyzer import PortfolioAnalyzer
from alpha_pulse.portfolio.mpt_strategy import MPTStrategy

async def main():
    # Initialize exchange connection
    exchange = BinanceConnector(
        api_key="your_key",
        api_secret="your_secret"
    )
    
    # Create portfolio analyzer
    analyzer = PortfolioAnalyzer(
        exchange=exchange,
        strategy=MPTStrategy(),
        lookback_days=30
    )
    
    # Get current allocation
    current_weights = await analyzer.get_current_allocation()
    
    # Calculate optimal allocation
    result = await analyzer.analyze_portfolio(
        constraints={
            'min_weight': 0.05,  # Minimum 5% per asset
            'max_weight': 0.40   # Maximum 40% per asset
        }
    )
    
    # Get rebalancing trades
    trades = analyzer.get_rebalancing_trades(
        current_weights=current_weights,
        target_weights=result.weights,
        total_value=100000,
        min_trade_value=10
    )
```

## Configuration

### Exchange Setup

1. Create API keys with appropriate permissions:
   - Read access for balances
   - (Optional) Trade access for automated rebalancing

2. Configure environment variables:
```bash
EXCHANGE_API_KEY=your_api_key
EXCHANGE_API_SECRET=your_api_secret
```

### Strategy Parameters

MPT Strategy:
- `risk_free_rate`: Annual risk-free rate (default: 2%)
- `lookback_days`: Historical data period (default: 30)

HRP Strategy:
- `risk_free_rate`: Annual risk-free rate (default: 2%)
- `lookback_days`: Historical data period (default: 30)

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