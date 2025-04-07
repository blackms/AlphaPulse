# Long/Short Strategy Configuration Guide

## 1. Overview

This document provides guidance on configuring the long/short trading strategy for the S&P 500. The strategy combines trend-following, mean reversion, and volatility signals to determine market positioning, with weekly or monthly rebalancing.

## 2. Configuration File Structure

The strategy is configured through the `config/backtest_sp500_config.yaml` file. Below is the structure of the configuration specific to the long/short strategy:

```yaml
# Long/Short Strategy Configuration
long_short_strategy:
  enabled: true
  rebalance_frequency: "weekly"  # "weekly" or "monthly"
  
  # Data Configuration
  data:
    symbols:
      market: "^GSPC"  # S&P 500 index
      volatility: "^VIX"  # VIX index
    cache_dir: "./data/cache"
    lookback_period: 252  # Trading days for initial data load
  
  # Indicator Configuration
  indicators:
    trend:
      ma_period: 40  # 40-week moving average
    mean_reversion:
      rsi_period: 14  # 14-period RSI
      overbought_threshold: 70  # RSI overbought level
      oversold_threshold: 30  # RSI oversold level
    volatility:
      vix_high_threshold: 30  # High volatility threshold
      vix_extreme_threshold: 40  # Extreme volatility threshold
  
  # Signal Configuration
  signals:
    trend_weight: 1.0  # Weight for trend signal
    mean_reversion_weight: 0.5  # Weight for mean reversion signal
    volatility_weight: 0.5  # Weight for volatility signal
    signal_threshold: 0.1  # Minimum signal strength to generate position
  
  # Position Management
  position:
    max_position_size: 1.0  # Maximum position size (1.0 = 100% of capital)
    min_position_size: 0.1  # Minimum position size (0.1 = 10% of capital)
    size_by_signal: true  # Scale position size by signal strength
  
  # Risk Management
  risk:
    stop_loss_pct: 0.08  # 8% stop loss
    trailing_stop_pct: 0.05  # 5% trailing stop
    max_drawdown_pct: 0.25  # 25% maximum drawdown
    reduce_size_drawdown_pct: 0.15  # Reduce position at 15% drawdown
    reduction_factor: 0.5  # Reduce to 50% of original size
```

## 3. Configuration Parameters

### 3.1 General Settings

- **enabled**: Enable or disable the long/short strategy
- **rebalance_frequency**: Frequency of portfolio rebalancing ("weekly" or "monthly")

### 3.2 Data Configuration

- **symbols.market**: Symbol for the market index (default: "^GSPC" for S&P 500)
- **symbols.volatility**: Symbol for the volatility index (default: "^VIX")
- **cache_dir**: Directory for caching data to avoid redundant API calls
- **lookback_period**: Number of trading days to load initially (default: 252, approximately 1 year)

### 3.3 Indicator Configuration

#### 3.3.1 Trend Indicators
- **ma_period**: Period for the moving average in weeks (default: 40)

#### 3.3.2 Mean Reversion Indicators
- **rsi_period**: Period for the RSI calculation (default: 14)
- **overbought_threshold**: RSI threshold for overbought condition (default: 70)
- **oversold_threshold**: RSI threshold for oversold condition (default: 30)

#### 3.3.3 Volatility Indicators
- **vix_high_threshold**: VIX threshold for high volatility (default: 30)
- **vix_extreme_threshold**: VIX threshold for extreme volatility (default: 40)

### 3.4 Signal Configuration

- **trend_weight**: Weight for the trend-following signal (default: 1.0)
- **mean_reversion_weight**: Weight for the mean reversion signal (default: 0.5)
- **volatility_weight**: Weight for the volatility-based signal (default: 0.5)
- **signal_threshold**: Minimum signal strength to generate a position (default: 0.1)

### 3.5 Position Management

- **max_position_size**: Maximum position size as a fraction of capital (default: 1.0 = 100%)
- **min_position_size**: Minimum position size as a fraction of capital (default: 0.1 = 10%)
- **size_by_signal**: Whether to scale position size by signal strength (default: true)

### 3.6 Risk Management

- **stop_loss_pct**: Stop loss percentage from entry price (default: 0.08 = 8%)
- **trailing_stop_pct**: Trailing stop percentage (default: 0.05 = 5%)
- **max_drawdown_pct**: Maximum allowed drawdown before exiting all positions (default: 0.25 = 25%)
- **reduce_size_drawdown_pct**: Drawdown threshold to reduce position size (default: 0.15 = 15%)
- **reduction_factor**: Factor to reduce position size by when drawdown threshold is reached (default: 0.5 = 50%)

## 4. Configuration Guidelines

### 4.1 Rebalancing Frequency

- **Weekly Rebalancing**: More responsive to market changes, but higher transaction costs
- **Monthly Rebalancing**: Lower transaction costs, but less responsive to market changes

Recommendation: Start with weekly rebalancing and monitor transaction costs.

### 4.2 Indicator Parameters

#### 4.2.1 Moving Average Period
- **Shorter Period (20-30 weeks)**: More responsive, but more false signals
- **Medium Period (40-50 weeks)**: Balance between responsiveness and stability
- **Longer Period (60+ weeks)**: More stable, but slower to respond to trend changes

Recommendation: Start with a 40-week MA and adjust based on performance.

#### 4.2.2 RSI Parameters
- **RSI Period**: Standard is 14 periods, can be adjusted for sensitivity
- **Overbought/Oversold Thresholds**: Standard is 70/30, can be widened (80/20) for less frequent signals or narrowed (60/40) for more frequent signals

Recommendation: Start with standard 14-period RSI and 70/30 thresholds.

#### 4.2.3 VIX Thresholds
- **High Volatility**: Typically 25-30
- **Extreme Volatility**: Typically 35-40

Recommendation: Analyze historical VIX data to identify appropriate thresholds for your risk tolerance.

### 4.3 Signal Weights

- **Trend Weight**: Primary signal, typically highest weight (1.0)
- **Mean Reversion Weight**: Secondary signal, typically lower weight (0.3-0.7)
- **Volatility Weight**: Risk adjustment signal, typically lower weight (0.3-0.7)

Recommendation: Start with trend_weight=1.0, mean_reversion_weight=0.5, volatility_weight=0.5, and adjust based on performance.

### 4.4 Position Sizing

- **Fixed Size**: Set size_by_signal=false for consistent position sizes
- **Variable Size**: Set size_by_signal=true to scale position size by signal strength

Recommendation: Use variable sizing (size_by_signal=true) to reflect confidence in signals.

### 4.5 Risk Management

- **Stop Loss**: Typically 5-10% depending on volatility and time horizon
- **Trailing Stop**: Typically 5-8% to lock in profits
- **Drawdown Control**: Typically 15-25% for maximum drawdown

Recommendation: Be conservative with risk parameters initially, then adjust based on performance.

## 5. Example Configurations

### 5.1 Conservative Configuration

```yaml
long_short_strategy:
  enabled: true
  rebalance_frequency: "monthly"
  
  indicators:
    trend:
      ma_period: 50  # Longer MA period for stability
    mean_reversion:
      rsi_period: 14
      overbought_threshold: 75  # Higher threshold for fewer signals
      oversold_threshold: 25  # Lower threshold for fewer signals
    volatility:
      vix_high_threshold: 25  # Lower threshold for earlier risk reduction
      vix_extreme_threshold: 35  # Lower threshold for earlier exit
  
  signals:
    trend_weight: 1.0
    mean_reversion_weight: 0.3  # Lower weight on mean reversion
    volatility_weight: 0.7  # Higher weight on volatility
  
  position:
    max_position_size: 0.8  # Maximum 80% of capital
    min_position_size: 0.2
  
  risk:
    stop_loss_pct: 0.05  # Tighter stop loss
    trailing_stop_pct: 0.03  # Tighter trailing stop
    max_drawdown_pct: 0.15  # Lower maximum drawdown
    reduce_size_drawdown_pct: 0.10
    reduction_factor: 0.5
```

### 5.2 Aggressive Configuration

```yaml
long_short_strategy:
  enabled: true
  rebalance_frequency: "weekly"
  
  indicators:
    trend:
      ma_period: 30  # Shorter MA period for responsiveness
    mean_reversion:
      rsi_period: 10  # Shorter RSI period for responsiveness
      overbought_threshold: 65  # Lower threshold for more signals
      oversold_threshold: 35  # Higher threshold for more signals
    volatility:
      vix_high_threshold: 35  # Higher threshold for later risk reduction
      vix_extreme_threshold: 45  # Higher threshold for later exit
  
  signals:
    trend_weight: 1.0
    mean_reversion_weight: 0.7  # Higher weight on mean reversion
    volatility_weight: 0.3  # Lower weight on volatility
  
  position:
    max_position_size: 1.0  # Maximum 100% of capital
    min_position_size: 0.1
  
  risk:
    stop_loss_pct: 0.10  # Wider stop loss
    trailing_stop_pct: 0.07  # Wider trailing stop
    max_drawdown_pct: 0.25  # Higher maximum drawdown
    reduce_size_drawdown_pct: 0.20
    reduction_factor: 0.7  # Less reduction
```

## 6. Backtesting Configuration

To run a backtest with the long/short strategy, use the following command:

```bash
python run_backtest.py --config config/backtest_sp500_config.yaml --output-dir ./results/long_short_backtest
```

You can also specify a custom configuration file:

```bash
python run_backtest.py --config config/my_custom_config.yaml --output-dir ./results/custom_backtest
```

## 7. Performance Monitoring

After running a backtest, review the following metrics to evaluate performance:

- **Sharpe Ratio**: Target > 1.0
- **Maximum Drawdown**: Target < 25%
- **CAGR**: Target > 8%
- **Win Rate**: Target > 55%

If the strategy doesn't meet these targets, consider adjusting the configuration parameters based on the guidelines in this document.