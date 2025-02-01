# 🤖 Reinforcement Learning Trading Module

This module implements advanced reinforcement learning capabilities for algorithmic trading in AlphaPulse. It provides a gym-compatible environment and sophisticated training utilities to develop RL-based trading agents with deep learning architectures.

## 🧩 Components

### 🌍 Trading Environment (`env.py`)

A custom OpenAI Gym environment for training RL agents on financial data. 

Key features:
- 📊 **State Space**: 
  * Normalized price changes with multiple timeframes
  * Technical indicators with adaptive normalization
  * Market microstructure features
  * Order book imbalance metrics
  * Volume profile indicators

- 🎯 **Action Space**: 
  * Discrete actions (sell, hold, buy)
  * Continuous position sizing
  * Multiple order types support
  * Risk-aware action masking

- 💰 **Reward Function**: 
  * PnL-based rewards with risk adjustment
  * Sharpe ratio component
  * Transaction cost penalties
  * Position holding costs
  * Custom reward shaping options

- 📈 **Position Management**: 
  * Dynamic position sizing
  * Risk-adjusted trade execution
  * Stop-loss/take-profit handling
  * Position unwinding logic

- 💸 **Transaction Costs**: 
  * Multi-tier fee structure
  * Slippage modeling
  * Market impact estimation
  * Spread considerations

Configuration options:
```python
TradingEnvConfig(
    initial_capital=100000.0,    # Starting capital
    commission=0.001,            # Trading commission (0.1%)
    position_size=1.0,           # Position size as fraction of capital
    window_size=10,              # Historical window for state
    reward_scaling=1.0,          # Reward scaling factor
    risk_aversion=0.1,           # Risk aversion parameter
    max_position=5.0,            # Maximum position size
    stop_loss_pct=0.02,         # Stop-loss percentage
    take_profit_pct=0.05        # Take-profit percentage
)
```

### 🧠 RL Trainer (`rl_trainer.py`)

Advanced training framework built on stable-baselines3, supporting:

- 🔄 Multiple algorithms:
  * PPO (Proximal Policy Optimization)
  * SAC (Soft Actor-Critic)
  * TD3 (Twin Delayed DDPG)
  * A2C (Advantage Actor-Critic)

- 🏗️ Neural network features:
  * Custom architectures
  * Attention mechanisms
  * LSTM/GRU layers
  * Residual connections
  * Batch normalization

- 📊 Training enhancements:
  * Curriculum learning
  * Experience replay
  * Priority sampling
  * Multi-agent training
  * Distributed training support

Configuration options:
```python
NetworkConfig(
    hidden_sizes=[128, 64, 32],  # Network architecture
    activation_fn="relu",        # Activation function
    use_lstm=True,              # Enable LSTM layers
    lstm_units=64,              # LSTM layer size
    attention_heads=4,          # Number of attention heads
    dropout_rate=0.1            # Dropout probability
)

TrainingConfig(
    total_timesteps=1_000_000,  # Training duration
    learning_rate=3e-4,         # Learning rate
    batch_size=256,             # Batch size
    n_steps=2048,               # Steps per update
    gamma=0.99,                 # Discount factor
    gae_lambda=0.95,            # GAE parameter
    clip_range=0.2,             # PPO clip range
    ent_coef=0.01,             # Entropy coefficient
    vf_coef=0.5,               # Value function coefficient
    max_grad_norm=0.5,         # Gradient clipping
    eval_freq=10_000,          # Evaluation frequency
    n_eval_episodes=10         # Episodes per evaluation
)
```

## 🚀 Usage Example

See `examples/demo_rl_trading.py` for a complete example:

```python
from alpha_pulse.rl.env import TradingEnv, TradingEnvConfig
from alpha_pulse.rl.rl_trainer import RLTrainer, TrainingConfig, NetworkConfig

# Configure environment with advanced features
env_config = TradingEnvConfig(
    initial_capital=100000.0,
    commission=0.001,
    position_size=0.2,
    risk_aversion=0.1,
    stop_loss_pct=0.02
)

# Configure sophisticated training setup
training_config = TrainingConfig(
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    batch_size=256,
    use_lstm=True,
    attention_heads=4
)

# Create trainer and train with monitoring
trainer = RLTrainer(env_config, training_config)
model = trainer.train(
    prices=train_prices,
    features=train_features,
    eval_prices=test_prices,
    eval_features=test_features
)

# Comprehensive performance evaluation
metrics = trainer.evaluate(
    model=model,
    prices=test_prices,
    features=test_features,
    n_episodes=10
)
```

## 🏗️ Model Architecture

The advanced model architecture includes:

- 🧠 Policy Network:
  * Multi-layer perceptron base
  * LSTM for temporal dependencies
  * Self-attention mechanisms
  * Residual connections
  * Dual heads for policy and value

- 📊 Feature Processing:
  * Adaptive normalization
  * Feature crossing
  * Temporal embedding
  * Market regime detection

## 🔧 Dependencies

- 🎮 gymnasium: Environment framework
- 🤖 stable-baselines3: RL algorithms
- 🔥 torch: Neural network backend
- 🐼 pandas: Data handling
- 🔢 numpy: Numerical computations

## 📁 Directory Structure

```
rl/
├── env.py              # Trading environment implementation
├── rl_trainer.py       # Training utilities and configuration
└── README.md          # This documentation
```

## ⚡ Performance Considerations

1. 🚀 Training Optimization:
   - Vectorized environment for parallel training
   - Gradient accumulation for large batches
   - Mixed precision training
   - Custom CUDA kernels for speed

2. 📊 State Space Design:
   - Feature selection via importance analysis
   - Efficient state representation
   - Adaptive feature normalization
   - Market regime conditioning

3. 🎯 Action Space:
   - Dynamic action masking
   - Risk-aware action filtering
   - Position size optimization
   - Order type selection

4. 💰 Risk Management:
   - Integrated stop-loss mechanisms
   - Dynamic position sizing
   - Portfolio-aware trading
   - Transaction cost optimization

## 🔍 Best Practices

1. 🎯 Training:
   - Start with simple architectures
   - Use curriculum learning
   - Monitor reward stability
   - Validate on multiple market regimes

2. 📊 Hyperparameters:
   - Tune learning rate carefully
   - Adjust batch size for stability
   - Balance exploration/exploitation
   - Consider market characteristics

3. 🛡️ Risk Control:
   - Implement position limits
   - Use stop-loss consistently
   - Monitor drawdown
   - Test in various market conditions

4. 📈 Evaluation:
   - Use multiple metrics
   - Test on unseen data
   - Consider transaction costs
   - Validate robustness