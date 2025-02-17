# PPO Trend Following Strategy

This implementation uses Proximal Policy Optimization (PPO) to implement a trend following trading strategy. The strategy combines traditional technical analysis with deep reinforcement learning to identify and trade market trends.

## Strategy Overview

The trend following strategy:
- Uses multiple timeframe analysis (20, 50, 200 periods)
- Combines momentum and volatility indicators
- Implements dynamic position sizing based on trend strength
- Features risk management with stop-loss and take-profit levels

### Key Features

1. **Multi-Timeframe Analysis**
   - Short-term (20 periods) for entry timing
   - Medium-term (50 periods) for trend confirmation
   - Long-term (200 periods) for overall trend direction

2. **Technical Indicators**
   - Moving Averages for trend direction
   - RSI and MACD for momentum
   - ADX for trend strength
   - ATR for volatility-based position sizing
   - Volume indicators for trend confirmation

3. **Risk Management**
   - Dynamic position sizing based on trend strength
   - Trailing stop-loss for trend following
   - Take-profit targets based on volatility
   - Maximum position limits

4. **Neural Network Architecture**
   - LSTM layers for temporal pattern recognition
   - Attention mechanism for key price level detection
   - Deep MLP for feature processing
   - Dropout for regularization

## Configuration

The strategy can be configured through `config.yaml`:
- Environment parameters (capital, position sizing, etc.)
- Network architecture (layers, activation functions)
- Training parameters (learning rate, batch size, etc.)
- Feature engineering settings

## Usage

```bash
# Train the model
python ppo_trend_following.py --asset-class crypto --model-name btc_trend_v1

# Different asset classes
python ppo_trend_following.py --asset-class stocks --model-name spy_trend_v1
python ppo_trend_following.py --asset-class forex --model-name eurusd_trend_v1

# GPU acceleration
python ppo_trend_following.py --gpu
```

## Model Organization

Models are saved in the following structure:
```
trained_models/rl/ppo/
├── crypto/
│   ├── btc_trend_v1/
│   │   ├── model.zip
│   │   └── checkpoints/
│   └── eth_trend_v1/
├── stocks/
└── forex/
```

## Performance Metrics

The strategy tracks:
- Win rate and profit factor
- Risk-adjusted returns (Sharpe, Sortino)
- Maximum drawdown and recovery time
- Position holding periods
- Trade execution accuracy

## Requirements

- PyTorch
- Stable-baselines3
- Pandas
- NumPy
- TA-Lib (for technical indicators)