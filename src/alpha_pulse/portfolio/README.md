# 📊 Portfolio Rebalancing Module

This module provides advanced portfolio rebalancing capabilities using modern portfolio optimization techniques. It supports multiple allocation strategies and is exchange-agnostic, working with any exchange that implements the required interface.

## 🎯 Overview

The portfolio module helps you:
- 📈 Analyze current portfolio allocations in real-time
- 🎯 Calculate optimal target allocations using multiple strategies
- 🔄 Generate smart rebalancing recommendations
- 📊 Track portfolio metrics and performance
- 🤖 Get AI-powered portfolio insights
- 📈 Generate interactive HTML reports

## 💼 Allocation Strategies

### 1. 📈 Modern Portfolio Theory (MPT)

MPT, developed by Harry Markowitz, optimizes portfolios based on the trade-off between expected return and risk. The strategy:
- 🧮 Uses quadratic programming for optimization
- 📊 Calculates expected returns using exponentially weighted moving averages
- 🔄 Computes covariance matrices with shrinkage estimators
- ⚖️ Finds optimal weights maximizing the Sharpe ratio
- 🎯 Handles constraints through sequential quadratic programming

Key features:
- 📊 Mean-variance optimization with robust covariance estimation
- 📈 Risk-adjusted returns using configurable risk-free rate
- 🎯 Support for complex constraints (min/max weights, sector exposure)
- 📉 Interactive efficient frontier visualization

### 2. 🌳 Hierarchical Risk Parity (HRP)

HRP, developed by Marcos Lopez de Prado, provides a more robust alternative to MPT that:
- 🔍 Uses correlation-based distance metrics
- 🌳 Implements quasi-diagonalization for clustering
- ⚖️ Allocates weights through recursive bisection
- 📊 Handles numerical instabilities with advanced techniques

Implementation details:
- 🔢 Uses UPGMA clustering algorithm
- 📊 Implements inverse-variance allocation
- 🔄 Supports dynamic reclustering
- 📈 Provides cluster visualization
- 🔍 Enhanced debug logging for allocation decisions

### 3. 🎯 Black-Litterman Model

The Black-Litterman model combines market equilibrium returns with investor views:
- 🌍 Uses market cap weights as Bayesian prior
- 📊 Implements full covariance estimation
- 🎯 Supports multiple confidence levels
- 🔄 Handles missing data through statistical inference

Technical implementation:
- 🧮 Uses reverse optimization for equilibrium returns
- 📊 Implements Bayesian updating with uncertainty
- 🎯 Supports both absolute and relative views
- 📈 Provides posterior distribution analysis

### 4. 🤖 LLM-Assisted Strategy

AI-powered portfolio enhancement that can wrap any base strategy:
- 📰 Processes real-time news and sentiment data
- 🔍 Analyzes market trends and correlations
- 🎯 Provides dynamic risk assessments
- 💡 Generates natural language explanations
- 📊 Creates interactive HTML reports

Implementation details:
- 🔄 Uses async processing for real-time updates
- 📊 Implements sentiment scoring algorithms
- 🎯 Supports multiple LLM providers (OpenAI, etc.)
- 📈 Provides confidence metrics for suggestions
- 🔍 Detailed logging of analysis process
- 📱 Interactive visualization components
- 📊 Historical performance tracking
- 💡 Natural language recommendations

## 💻 Usage Examples

### Basic Portfolio Management

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

### LLM-Enhanced Analysis

```python
from alpha_pulse.portfolio.llm_analysis import OpenAILLMAnalyzer
from alpha_pulse.portfolio.html_report import HTMLReportGenerator

# Initialize LLM analyzer
analyzer = OpenAILLMAnalyzer(
    api_key="your_openai_api_key",
    model_name="o3-mini"  # or other supported models
)

# Get AI-powered analysis
portfolio_data = await manager.get_portfolio_data(exchange)
analysis = await manager.analyze_portfolio_with_llm(analyzer, exchange)

# Generate interactive HTML report
report_path = HTMLReportGenerator.generate_report(
    portfolio_data=portfolio_data,
    analysis_result=analysis,
    output_dir="reports"
)

# Access analysis components
print("Recommendations:", analysis.recommendations)
print("Risk Assessment:", analysis.risk_assessment)
print(f"Confidence Score: {analysis.confidence_score:.2%}")
print("Reasoning:", analysis.reasoning)

if analysis.rebalancing_suggestions:
    print("\nSuggested Rebalancing:")
    for suggestion in analysis.rebalancing_suggestions:
        print(f"- {suggestion.asset}: {suggestion.target_allocation:.2%}")
```

## 🔌 Exchange Integration

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

## ⚙️ Configuration

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

# LLM Analysis Settings
llm:
  enabled: true
  model: "o3-mini"
  confidence_threshold: 0.7
  update_frequency: "daily"
  max_tokens: 2000
```

## ⚠️ Risk Considerations

1. 💰 Transaction Costs
   - 📊 Implements advanced fee estimation
   - 🎯 Uses dynamic trade size optimization
   - 📈 Considers market impact costs
   - 🔄 Optimizes rebalancing frequency

2. 📊 Market Impact
   - 📈 Implements price impact models
   - 🔄 Uses smart order routing
   - 🎯 Supports TWAP/VWAP execution
   - ⚖️ Balances urgency vs. impact

3. 📊 Data Quality
   - 🔍 Implements outlier detection
   - 📈 Uses robust statistical methods
   - 🔄 Handles missing data points
   - 📊 Provides data quality metrics

4. 🎯 Portfolio Constraints
   - 📊 Supports complex constraint sets
   - 🔄 Implements feasibility checking
   - 📈 Provides constraint visualization
   - ⚖️ Handles soft constraints

## 🌟 Best Practices

1. 📊 Regular Monitoring
   - 📈 Track real-time metrics
   - 🔍 Monitor risk exposures
   - 📊 Review performance attribution
   - 🎯 Analyze tracking error

2. 🛡️ Risk Management
   - 📊 Implement position limits
   - 🎯 Use dynamic stop-losses
   - 📈 Monitor correlation changes
   - 🔄 Track risk factor exposures

3. 🔄 Rebalancing Frequency
   - 📊 Use adaptive thresholds
   - 📈 Monitor market volatility
   - 🎯 Consider trading volumes
   - 💰 Track transaction costs

4. 🧪 Testing
   - 📊 Implement backtesting
   - 🔄 Use Monte Carlo simulation
   - 📈 Stress test strategies
   - 🎯 Monitor live performance

## 📊 Logging and Monitoring

1. 🔍 Debug Logging
   - 📊 Detailed allocation calculations
   - 🎯 Trade execution steps
   - 📈 Risk metric computations
   - 🔄 Rebalancing decisions

2. 📈 Performance Tracking
   - 📊 Real-time metrics
   - 🎯 Historical comparisons
   - 📈 Attribution analysis
   - 🔄 Risk-adjusted returns

3. 🤖 LLM Analysis Logging
   - 📊 Confidence scores
   - 🎯 Recommendation tracking
   - 📈 Market context
   - 🔄 Decision reasoning

4. 📱 Visualization
   - 📊 Interactive charts
   - 🎯 Risk dashboards
   - 📈 Performance metrics
   - 🔄 Allocation history

## ⚠️ Disclaimer

This module is for informational purposes only. Always:
- 🧪 Test thoroughly before live trading
- 🎯 Start with small positions
- 📊 Monitor performance closely
- ⚖️ Consider your risk tolerance
- 🔍 Understand exchange risks and limitations