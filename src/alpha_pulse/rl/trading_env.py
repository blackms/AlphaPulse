"""
Advanced trading environment for reinforcement learning in AlphaPulse.

This module implements a Gymnasium-compatible environment for training RL agents
with sophisticated feature engineering and risk management.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import gymnasium as gym
import numpy as np
import pandas as pd
from decimal import Decimal
import talib.abstract as ta
from loguru import logger

from ..portfolio.data_models import Position, PortfolioData
from ..data_pipeline.models import MarketData


class Actions(Enum):
    """Trading actions available to the agent."""
    Neutral = 0
    Long = 1
    ExitLong = 2
    Short = 3
    ExitShort = 4


@dataclass
class TradingEnvConfig:
    """Configuration for the trading environment."""
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1% commission per trade
    position_size: float = 0.2  # Max 20% of capital per trade
    window_size: int = 10  # Historical window size
    reward_scaling: float = 1.0
    risk_aversion: float = 0.1
    max_position: float = 5.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    max_trade_duration_candles: int = 100


class TradingEnv(gym.Env):
    """
    Advanced trading environment with comprehensive feature engineering
    and risk management.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        market_data: MarketData,
        config: Optional[TradingEnvConfig] = None
    ):
        """
        Initialize the trading environment.
        
        Args:
            market_data: Market data including prices and features
            config: Environment configuration
        """
        super().__init__()
        
        self.config = config or TradingEnvConfig()
        self.market_data = market_data
        
        # Validate data
        if not isinstance(market_data.prices, pd.DataFrame):
            raise ValueError("Market data must include price DataFrame")
            
        if (market_data.prices <= 0).any().any():
            raise ValueError("Prices must be positive")
            
        # Set up action and observation spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        
        # Calculate observation space dimension
        self.feature_dim = self._calculate_feature_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.feature_dim,),
            dtype=np.float32
        )
        
        # Initialize state
        self._reset_state()
        
    def _calculate_feature_dim(self) -> int:
        """Calculate total feature dimension."""
        # Base features (OHLCV)
        n_features = 5
        
        # Technical indicators
        n_features += len(self._calculate_technical_indicators(
            self.market_data.prices.iloc[:5]
        ).columns)
        
        # Historical window
        return n_features * self.config.window_size
        
    def _reset_state(self) -> None:
        """Reset the environment state."""
        self.current_step = self.config.window_size
        self.equity = Decimal(str(self.config.initial_capital))
        self.current_position: Optional[Position] = None
        self.positions: List[Position] = []
        self.portfolio_history: Dict[pd.Timestamp, PortfolioData] = {}
        
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for feature engineering."""
        df = data.copy()
        
        # Trend indicators
        df['rsi'] = ta.RSI(df['close'], timeperiod=14)
        df['mfi'] = ta.MFI(df[['high', 'low', 'close', 'volume']], timeperiod=14)
        df['adx'] = ta.ADX(df[['high', 'low', 'close']], timeperiod=14)
        df['cci'] = ta.CCI(df[['high', 'low', 'close']], timeperiod=14)
        
        # Momentum indicators
        macd = ta.MACD(df['close'])
        df['macd'] = macd['macd']
        df['macdsignal'] = macd['macdsignal']
        df['macdhist'] = macd['macdhist']
        
        # Volatility indicators
        df['atr'] = ta.ATR(df[['high', 'low', 'close']], timeperiod=14)
        df['natr'] = df['atr'] / df['close']
        
        # Volume indicators
        df['obv'] = ta.OBV(df['close'], df['volume'])
        df['cmf'] = self._chaikin_money_flow(df)
        
        # Clean up NaN values
        return df.fillna(0)
        
    def _chaikin_money_flow(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow."""
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfv = mfv.fillna(0.0) * df['volume']
        return mfv.rolling(period).sum() / df['volume'].rolling(period).sum()
        
    def _calculate_state(self) -> np.ndarray:
        """Calculate the current state observation."""
        end_idx = self.current_step
        start_idx = end_idx - self.config.window_size
        
        # Get price and feature windows
        price_window = self.market_data.prices.iloc[start_idx:end_idx]
        
        # Calculate technical indicators
        features = self._calculate_technical_indicators(price_window)
        
        # Normalize features
        normalized = (features - features.mean()) / features.std()
        normalized = normalized.fillna(0)
        
        # Flatten and convert to numpy array
        state = normalized.values.flatten()
        
        return state.astype(np.float32)
        
    def _calculate_reward(self, action: int) -> float:
        """
        Calculate reward based on PnL and risk-adjusted metrics.
        
        Args:
            action: The action taken by the agent
            
        Returns:
            float: The calculated reward
        """
        reward = 0.0
        
        if self.current_position is not None:
            # Calculate unrealized PnL
            current_price = Decimal(str(self.market_data.prices['close'].iloc[self.current_step]))
            position_value = self.current_position.quantity * current_price
            entry_value = self.current_position.quantity * self.current_position.entry_price
            unrealized_pnl = position_value - entry_value
            
            # Risk-adjusted reward
            volatility = float(self.market_data.prices['close'].pct_change().std())
            risk_adjustment = 1.0 / (1.0 + self.config.risk_aversion * volatility)
            
            # Scale reward
            reward = float(unrealized_pnl) / float(self.config.initial_capital)
            reward *= self.config.reward_scaling * risk_adjustment
            
            # Penalize for holding positions too long
            trade_duration = self.current_step - self.positions[-1].entry_time
            if trade_duration > self.config.max_trade_duration_candles:
                reward *= 0.5
                
        return reward
        
    def _update_position(self, action: Actions) -> Tuple[float, bool]:
        """
        Update position based on action.
        
        Args:
            action: The action to take
            
        Returns:
            tuple: (realized PnL, whether trade was executed)
        """
        current_price = Decimal(str(self.market_data.prices['close'].iloc[self.current_step]))
        realized_pnl = Decimal('0')
        trade_executed = False
        
        # Close existing position if we have one and action is exit or opposite
        if self.current_position is not None:
            if (
                (action == Actions.ExitLong and self.current_position.quantity > 0) or
                (action == Actions.ExitShort and self.current_position.quantity < 0) or
                (action == Actions.Long and self.current_position.quantity < 0) or
                (action == Actions.Short and self.current_position.quantity > 0)
            ):
                # Calculate PnL including commission
                exit_value = abs(self.current_position.quantity) * current_price * \
                    (Decimal('1') - Decimal(str(self.config.commission)))
                entry_value = abs(self.current_position.quantity) * self.current_position.entry_price * \
                    (Decimal('1') + Decimal(str(self.config.commission)))
                    
                realized_pnl = exit_value - entry_value if self.current_position.quantity > 0 \
                    else entry_value - exit_value
                
                # Update position
                self.current_position.exit_price = current_price
                self.current_position.exit_time = self.current_step
                self.current_position.pnl = realized_pnl
                
                # Store position history
                self.positions.append(self.current_position)
                self.current_position = None
                trade_executed = True
                
        # Open new position if action is entry and we don't have a position
        if self.current_position is None:
            if action in [Actions.Long, Actions.Short]:
                # Calculate position size
                position_value = self.equity * Decimal(str(self.config.position_size))
                quantity = position_value / current_price
                
                if action == Actions.Short:
                    quantity = -quantity
                    
                if quantity != 0:
                    self.current_position = Position(
                        asset_id="TRADING_PAIR",
                        quantity=quantity,
                        entry_price=current_price,
                        entry_time=self.current_step
                    )
                    trade_executed = True
                    
        return float(realized_pnl), trade_executed
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Index of the action to take
            
        Returns:
            tuple containing:
            - observation: The current state
            - reward: The step reward
            - terminated: Whether the episode is done
            - truncated: Whether the episode was truncated
            - info: Additional information
        """
        if not 0 <= action < len(Actions):
            raise ValueError(f"Invalid action: {action}")
            
        # Take action and get reward
        realized_pnl, trade_executed = self._update_position(Actions(action))
        
        # Update equity
        self.equity += Decimal(str(realized_pnl))
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.market_data.prices) - 1
        
        # Get new state
        state = self._calculate_state()
        
        # Calculate info dict
        info = {
            'equity': float(self.equity),
            'realized_pnl': realized_pnl,
            'position': self.current_position,
            'trade_executed': trade_executed
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
            'equity': float(self.equity),
            'realized_pnl': 0.0,
            'position': None,
            'trade_executed': False
        }
        
        return state, info
        
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment state."""
        if mode != 'human':
            raise ValueError(f"Unsupported render mode: {mode}")
            
        logger.info(
            f"Step: {self.current_step}, "
            f"Equity: ${float(self.equity):.2f}, "
            f"Position: {self.current_position}"
        )
        
    def close(self) -> None:
        """Clean up environment resources."""
        pass