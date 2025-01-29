"""
Trading environment for reinforcement learning.

This module implements a gym-like environment for training RL agents on financial data.
It integrates with AlphaPulse's existing backtesting framework for consistency.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import gymnasium as gym
import numpy as np
import pandas as pd
from loguru import logger

from ..backtesting.models import Position


@dataclass
class TradingEnvConfig:
    """Configuration for the trading environment."""
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1% commission per trade
    position_size: float = 1.0  # Fraction of capital to risk per trade
    window_size: int = 10  # Number of past observations to include in state
    reward_scaling: float = 1.0  # Scale factor for rewards


class TradingEnv(gym.Env):
    """OpenAI Gym environment for trading using historical market data."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        config: Optional[TradingEnvConfig] = None
    ):
        """
        Initialize the trading environment.
        
        Args:
            prices: Time series of asset prices
            features: DataFrame of features (technical indicators, etc.)
            config: Environment configuration parameters
        """
        super().__init__()
        
        if not prices.index.equals(features.index):
            raise ValueError("Price and feature data must have matching timestamps")
            
        if (prices <= 0).any():
            raise ValueError("Prices must be positive")
            
        self.prices = prices
        self.features = features
        self.config = config or TradingEnvConfig()
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(3)  # -1 (sell), 0 (hold), 1 (buy)
        
        # Observation space includes price changes and features
        obs_dim = self.config.window_size * (1 + features.shape[1])
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Calculate feature statistics for normalization
        self._setup_feature_normalization()
        
        # Initialize state variables
        self._reset_state()
        
    def _setup_feature_normalization(self) -> None:
        """Calculate feature means and standard deviations for normalization."""
        # Calculate price change statistics
        price_changes = self.prices.pct_change().fillna(0)
        self.price_mean = price_changes.mean()
        self.price_std = price_changes.std()
        if self.price_std == 0:
            self.price_std = 1.0
            
        # Calculate feature statistics
        self.feature_means = self.features.mean()
        self.feature_stds = self.features.std()
        # Replace zero standard deviations with 1.0 to avoid division by zero
        self.feature_stds = self.feature_stds.replace(0, 1.0)
        
    def _reset_state(self) -> None:
        """Reset the environment state."""
        self.current_step = self.config.window_size
        self.equity = self.config.initial_capital
        self.current_position: Optional[Position] = None
        self.positions: List[Position] = []
        self.equity_curve: Dict[pd.Timestamp, float] = {}
        
    def _calculate_state(self) -> np.ndarray:
        """Calculate the current state observation."""
        # Get window of price changes and features
        end_idx = self.current_step
        start_idx = end_idx - self.config.window_size
        
        # Calculate normalized price changes
        price_window = self.prices.iloc[start_idx:end_idx]
        price_changes = price_window.pct_change().fillna(0)
        normalized_price_changes = (price_changes - self.price_mean) / self.price_std
        
        # Get and normalize feature window
        feature_window = self.features.iloc[start_idx:end_idx]
        normalized_features = (feature_window - self.feature_means) / self.feature_stds
        
        # Combine and flatten
        state = np.concatenate([
            normalized_price_changes.values.reshape(-1, 1),
            normalized_features.values
        ], axis=1)
        
        # Replace any remaining NaN or infinite values
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return state.flatten().astype(np.float32)
        
    def _calculate_reward(self) -> float:
        """Calculate the step reward based on PnL and other factors."""
        reward = 0.0
        
        if self.current_position is not None:
            # Calculate unrealized PnL
            current_price = self.prices.iloc[self.current_step]
            unrealized_pnl = (
                self.current_position.size *
                (current_price - self.current_position.entry_price)
            )
            # Scale the reward
            reward = (unrealized_pnl / self.config.initial_capital) * self.config.reward_scaling
            
        return float(reward)
        
    def _update_position(self, action: int) -> float:
        """
        Update the current position based on the action.
        
        Args:
            action: 0 (sell), 1 (hold), 2 (buy)
            
        Returns:
            float: Realized PnL if any
        """
        current_price = self.prices.iloc[self.current_step]
        realized_pnl = 0.0
        
        # Convert action to direction (-1, 0, 1)
        direction = action - 1
        
        # Close existing position if we have one and action is opposite
        if self.current_position is not None:
            if direction != 0:  # If not holding
                # Calculate PnL including commission
                exit_value = (
                    self.current_position.size * 
                    current_price * 
                    (1 - self.config.commission)
                )
                entry_value = (
                    self.current_position.size *
                    self.current_position.entry_price *
                    (1 + self.config.commission)
                )
                realized_pnl = exit_value - entry_value
                
                # Update position
                self.current_position.exit_price = current_price
                self.current_position.exit_time = self.prices.index[self.current_step]
                self.current_position.pnl = realized_pnl
                
                # Store position history
                self.positions.append(self.current_position)
                self.current_position = None
                
        # Open new position if action is not hold
        if direction != 0 and self.current_position is None:
            position_size = (
                self.config.position_size * 
                self.equity / 
                current_price
            )
            if position_size > 0:
                self.current_position = Position(
                    entry_price=current_price,
                    entry_time=self.prices.index[self.current_step],
                    size=position_size * direction  # Negative for shorts
                )
                
        return realized_pnl
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0 (sell), 1 (hold), 2 (buy)
            
        Returns:
            tuple containing:
            - observation: The current state
            - reward: The step reward
            - terminated: Whether the episode is done
            - truncated: Whether the episode was truncated
            - info: Additional information
        """
        if not 0 <= action <= 2:
            raise ValueError(f"Invalid action: {action}")
            
        # Update position and get realized PnL
        realized_pnl = self._update_position(action)
        
        # Update equity
        self.equity += realized_pnl
        self.equity_curve[self.prices.index[self.current_step]] = self.equity
        
        # Calculate reward (includes both realized and unrealized PnL)
        reward = self._calculate_reward()
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.prices) - 1
        
        # Get new state
        state = self._calculate_state()
        
        # Calculate info dict
        info = {
            'equity': self.equity,
            'realized_pnl': realized_pnl,
            'position': self.current_position
        }
        
        return state, reward, done, False, info
        
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options (unused)
            
        Returns:
            tuple containing:
            - observation: Initial state
            - info: Additional information
        """
        super().reset(seed=seed)
        self._reset_state()
        
        state = self._calculate_state()
        info = {
            'equity': self.equity,
            'realized_pnl': 0.0,
            'position': None
        }
        
        return state, info
        
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Currently only logs basic info. Could be extended to plot equity curve, etc.
        
        Args:
            mode: Rendering mode ('human' only for now)
            
        Returns:
            None
        """
        if mode != 'human':
            raise ValueError(f"Unsupported render mode: {mode}")
            
        logger.info(
            f"Step: {self.current_step}, "
            f"Equity: ${self.equity:.2f}, "
            f"Position: {self.current_position}"
        )
        
    def close(self) -> None:
        """Clean up environment resources."""
        pass