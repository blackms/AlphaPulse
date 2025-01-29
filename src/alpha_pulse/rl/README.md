# Reinforcement Learning Trading Module

This module implements reinforcement learning capabilities for algorithmic trading in AlphaPulse. It provides a gym-compatible environment and training utilities to develop RL-based trading agents.

## Components

### Trading Environment (`env.py`)

A custom OpenAI Gym environment for training RL agents on financial data. Key features:

- **State Space**: Combines normalized price changes and technical indicators
- **Action Space**: Discrete actions (sell, hold, buy)
- **Reward Function**: Based on realized and unrealized PnL
- **Position Management**: Handles position sizing and trade execution
- **Transaction Costs**: Includes configurable commission fees

Configuration options:
```python
TradingEnvConfig(
    initial_capital=100000.0,    # Starting capital
    commission=0.001,            # Trading commission (0.1%)
    position_size=1.0,           # Position size as fraction of capital
    window_size=10,              # Historical window for state
    reward_scaling=1.0           # Reward scaling factor
)
```

### RL Trainer (`rl_trainer.py`)

Training framework built on stable-baselines3, supporting:

- PPO (Proximal Policy Optimization) algorithm
- Configurable neural network architecture
- Training with evaluation callbacks
- Model checkpointing and logging
- Performance evaluation

Configuration options:
```python
NetworkConfig(
    hidden_sizes=[64, 64],       # Network architecture
    activation_fn="tanh"         # Activation function
)

TrainingConfig(
    total_timesteps=100_000,     # Training duration
    learning_rate=0.0003,        # Learning rate
    batch_size=64,               # Batch size
    n_steps=2048,                # Steps per update
    gamma=0.99,                  # Discount factor
    eval_freq=10_000,            # Evaluation frequency
    n_eval_episodes=5            # Episodes per evaluation
)
```

## Usage Example

See `examples/demo_rl_trading.py` for a complete example. Basic usage:

```python
from alpha_pulse.rl.env import TradingEnv, TradingEnvConfig
from alpha_pulse.rl.rl_trainer import RLTrainer, TrainingConfig, NetworkConfig

# Configure environment
env_config = TradingEnvConfig(
    initial_capital=100000.0,
    commission=0.001,
    position_size=0.2
)

# Configure training
training_config = TrainingConfig(
    total_timesteps=10000,
    learning_rate=0.0003
)

# Create trainer and train
trainer = RLTrainer(env_config, training_config)
model = trainer.train(
    prices=train_prices,
    features=train_features,
    eval_prices=test_prices,
    eval_features=test_features
)

# Evaluate performance
metrics = trainer.evaluate(
    model=model,
    prices=test_prices,
    features=test_features,
    n_episodes=5
)
```

## Model Architecture

The default model uses a PPO policy with:
- MLP (Multi-Layer Perceptron) architecture
- Configurable hidden layer sizes
- Separate value and policy networks
- Tanh activation functions

## Dependencies

- gymnasium: Environment framework
- stable-baselines3: RL algorithms
- torch: Neural network backend
- pandas: Data handling
- numpy: Numerical computations

## Directory Structure

```
rl/
├── env.py              # Trading environment implementation
├── rl_trainer.py       # Training utilities and configuration
└── README.md          # This documentation
```

## Performance Considerations

- The environment supports vectorization for parallel training
- Reward scaling can be adjusted to stabilize training
- Window size affects the state space dimension
- Position size and commission impact risk/reward