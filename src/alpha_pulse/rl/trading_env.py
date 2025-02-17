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
import json
from pathlib import Path

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
        
        # Store log directory path as string
        self.log_dir = "logs/rl"
        
        self.config = config or TradingEnvConfig()
        self.market_data = market_data
        
        # Validate data
        if not isinstance(market_data.prices, pd.DataFrame):
            raise ValueError("Market data must include price DataFrame")
            
        # Extract OHLCV columns if they exist in the features
        price_cols = ['open', 'high', 'low', 'close']
        if all(col in market_data.prices.columns for col in price_cols):
            prices = market_data.prices[price_cols]
            if (prices <= 0).any().any():
                raise ValueError("Prices must be positive")
        else:
            # If no OHLCV columns, assume the data is already preprocessed features
            prices = market_data.prices
            
        # Set up action and observation spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        
        # Calculate feature dimensions
        sample_features = self._calculate_technical_indicators(
            self.market_data.prices.iloc[:5]
        )
        n_features = len(sample_features.columns)
        
        # Set up observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
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
        macd_line, signal_line, hist_line = ta.MACD(df['close'])
        df['macd'] = macd_line
        df['macdsignal'] = signal_line
        df['macdhist'] = hist_line
        
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
        # Get current price data
        current_data = self.market_data.prices.iloc[self.current_step:self.current_step+1]
        
        # Calculate technical indicators for current state
        features = self._calculate_technical_indicators(current_data)
        
        # Normalize features
        normalized = (features - features.mean()) / features.std()
        normalized = normalized.fillna(0)
        
        # Convert to numpy array
        state = normalized.iloc[0].values
        
        return state.astype(np.float32)
        
    def _calculate_reward(self, action: int) -> float:
        """
        Calculate reward based on PnL and risk-adjusted metrics.
        
        Args:
            action: The action taken by the agent
            
        Returns:
            float: The calculated reward
        """
        logger.debug(f"\nCalculating reward for step {self.current_step}:")
        logger.debug(f"  Action: {Actions(action).name}")
        
        reward = 0.0
        reward_components = {}
        
        if self.current_position is not None:
            # Calculate unrealized PnL
            current_price = float(self.market_data.prices['close'].iloc[self.current_step])
            position_value = float(self.current_position.quantity) * current_price
            entry_value = float(self.current_position.quantity) * float(self.current_position.avg_entry_price)
            unrealized_pnl = position_value - entry_value
            
            logger.debug("  Position metrics:")
            logger.debug(f"    Current price: {current_price}")
            logger.debug(f"    Position value: {position_value}")
            logger.debug(f"    Entry value: {entry_value}")
            logger.debug(f"    Unrealized PnL: {unrealized_pnl}")
            
            # Calculate base reward from PnL
            base_reward = unrealized_pnl / float(self.config.initial_capital)
            reward_components['base_reward'] = base_reward
            logger.debug(f"  Base reward (PnL based): {base_reward}")
            
            # Calculate risk adjustment
            price_series = self.market_data.prices['close'].iloc[max(0, self.current_step-20):self.current_step+1]
            volatility = float(price_series.pct_change().std())
            risk_adjustment = 1.0 / (1.0 + self.config.risk_aversion * volatility)
            reward_components['risk_adjustment'] = risk_adjustment
            logger.debug(f"  Risk metrics:")
            logger.debug(f"    Volatility: {volatility}")
            logger.debug(f"    Risk adjustment factor: {risk_adjustment}")
            
            # Apply risk adjustment and scaling
            reward = base_reward * risk_adjustment * self.config.reward_scaling
            reward_components['scaled_reward'] = reward
            logger.debug(f"  After risk adjustment and scaling: {reward}")
            
            # Check trade duration penalty
            if len(self.positions) > 0:
                trade_duration = self.current_step - self.positions[-1].timestamp
                logger.debug(f"  Trade duration: {trade_duration} steps")
                
                if trade_duration > self.config.max_trade_duration_candles:
                    duration_penalty = 0.5
                    reward *= duration_penalty
                    reward_components['duration_penalty'] = duration_penalty
                    logger.debug(f"  Applied duration penalty: {duration_penalty}")
                    logger.debug(f"  Final reward after penalty: {reward}")
                else:
                    logger.debug("  No duration penalty applied")
                    reward_components['duration_penalty'] = 1.0
        else:
            logger.debug("  No active position, reward remains 0")
            reward_components = {
                'base_reward': 0.0,
                'risk_adjustment': 1.0,
                'scaled_reward': 0.0,
                'duration_penalty': 1.0
            }
        
        # Log final reward and components
        logger.debug("  Reward components:")
        for component, value in reward_components.items():
            logger.debug(f"    {component}: {value}")
        logger.debug(f"  Final reward: {reward}")
        
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
        
        # Log initial state
        logger.debug(f"Position update - Step {self.current_step}:")
        logger.debug(f"  Action: {action}")
        logger.debug(f"  Current price: {current_price}")
        logger.debug(f"  Current position: {self.current_position}")
        logger.debug(f"  Current equity: {self.equity}")
        
        # Close existing position if we have one and action is exit or opposite
        if self.current_position is not None:
            close_conditions = {
                'exit_long': action == Actions.ExitLong and self.current_position.quantity > 0,
                'exit_short': action == Actions.ExitShort and self.current_position.quantity < 0,
                'reverse_to_long': action == Actions.Long and self.current_position.quantity < 0,
                'reverse_to_short': action == Actions.Short and self.current_position.quantity > 0
            }
            
            logger.debug(f"  Close conditions: {close_conditions}")
            
            if any(close_conditions.values()):
                # Calculate PnL including commission
                quantity = float(self.current_position.quantity)
                exit_value = abs(quantity) * float(current_price) * \
                    (1.0 - float(self.config.commission))
                entry_value = abs(quantity) * float(self.current_position.avg_entry_price) * \
                    (1.0 + float(self.config.commission))
                    
                realized_pnl = exit_value - entry_value if quantity > 0 \
                    else entry_value - exit_value
                
                # Log position closing details
                logger.debug("  Closing position:")
                logger.debug(f"    Quantity: {quantity}")
                logger.debug(f"    Exit value: {exit_value}")
                logger.debug(f"    Entry value: {entry_value}")
                logger.debug(f"    Realized PnL: {realized_pnl}")
                
                # Update position
                self.current_position.exit_price = current_price
                self.current_position.exit_time = self.current_step
                self.current_position.pnl = realized_pnl
                
                # Store position history
                self.positions.append(self.current_position)
                logger.debug(f"  Added position to history. Total positions: {len(self.positions)}")
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
                
                # Log position sizing calculations
                logger.debug("  Opening new position:")
                logger.debug(f"    Position value: {position_value}")
                logger.debug(f"    Calculated quantity: {quantity}")
                    
                if quantity != 0:
                    self.current_position = Position(
                        symbol="TRADING_PAIR",
                        quantity=float(quantity),
                        avg_entry_price=float(current_price),
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        timestamp=float(self.current_step)
                    )
                    trade_executed = True
                    logger.debug(f"  Opened new position: {self.current_position}")
                else:
                    logger.warning("  Zero quantity calculated, no position opened")
                    
        # Log final state
        logger.debug(f"  Trade executed: {trade_executed}")
        logger.debug(f"  Final position: {self.current_position}")
        logger.debug(f"  Realized PnL: {realized_pnl}")
                    
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
        # Log step start
        logger.debug(f"\nStep {self.current_step} start:")
        logger.debug(f"  Action received: {action} ({Actions(action).name})")
        
        if not 0 <= action < len(Actions):
            raise ValueError(f"Invalid action: {action}")
            
        # Get current market state
        current_price = float(self.market_data.prices['close'].iloc[self.current_step])
        previous_equity = float(self.equity)
        logger.debug(f"  Current price: {current_price}")
        logger.debug(f"  Starting equity: {previous_equity}")
        
        # Take action and get reward
        realized_pnl, trade_executed = self._update_position(Actions(action))
        logger.debug(f"  Position update result:")
        logger.debug(f"    Realized PnL: {realized_pnl}")
        logger.debug(f"    Trade executed: {trade_executed}")
        
        # Update equity
        self.equity += Decimal(str(realized_pnl))
        equity_change = float(self.equity) - previous_equity
        logger.debug(f"  Equity change: {equity_change}")
        
        # Calculate reward
        reward = self._calculate_reward(action)
        logger.debug(f"  Calculated reward: {reward}")
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.market_data.prices) - 1
        logger.debug(f"  Episode done: {done}")
        
        # Get new state
        state = self._calculate_state()
        logger.debug(f"  New state shape: {state.shape}")
        logger.debug(f"  State summary - Mean: {state.mean():.4f}, Std: {state.std():.4f}")
        
        # Calculate additional metrics
        unrealized_pnl = 0.0
        if self.current_position is not None:
            position_value = float(self.current_position.quantity) * current_price
            entry_value = float(self.current_position.quantity) * float(self.current_position.avg_entry_price)
            unrealized_pnl = position_value - entry_value
            logger.debug(f"  Current position details:")
            logger.debug(f"    Position value: {position_value}")
            logger.debug(f"    Entry value: {entry_value}")
            logger.debug(f"    Unrealized PnL: {unrealized_pnl}")
        
        # Calculate info dict with extended information
        info = {
            'equity': float(self.equity),
            'equity_change': equity_change,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'position': self.current_position,
            'trade_executed': trade_executed,
            'current_price': current_price,
            'total_trades': len(self.positions),
            'step_reward': reward
        }
        
        # Log step completion
        logger.debug("Step completion info:")
        for key, value in info.items():
            logger.debug(f"  {key}: {value}")
        
        # Log step information
        self._log_step_info(
            step=self.current_step,
            action=action,
            reward=reward,
            done=done,
            info=info,
            state=state
        )
            
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
        logger.debug("\nResetting environment:")
        logger.debug(f"  Seed: {seed}")
        logger.debug(f"  Options: {options}")
        
        # Store previous state for debugging
        previous_state = {
            'step': getattr(self, 'current_step', None),
            'equity': float(self.equity) if hasattr(self, 'equity') else None,
            'positions': len(self.positions) if hasattr(self, 'positions') else 0,
            'current_position': self.current_position if hasattr(self, 'current_position') else None
        }
        logger.debug("  Previous state:")
        for key, value in previous_state.items():
            logger.debug(f"    {key}: {value}")
        
        # Reset random state
        super().reset(seed=seed)
        if seed is not None:
            logger.debug(f"  Random state reset with seed: {seed}")
        
        # Reset environment state
        self._reset_state()
        logger.debug("  Environment state reset:")
        logger.debug(f"    Current step: {self.current_step}")
        logger.debug(f"    Initial equity: {float(self.equity)}")
        logger.debug(f"    Positions cleared: {len(self.positions)}")
        
        # Calculate initial state
        state = self._calculate_state()
        logger.debug("  Initial state metrics:")
        logger.debug(f"    Shape: {state.shape}")
        logger.debug(f"    Mean: {state.mean():.4f}")
        logger.debug(f"    Std: {state.std():.4f}")
        logger.debug(f"    Min: {state.min():.4f}")
        logger.debug(f"    Max: {state.max():.4f}")
        
        # Verify market data availability
        available_steps = len(self.market_data.prices) - self.current_step
        logger.debug("  Market data status:")
        logger.debug(f"    Total timesteps available: {len(self.market_data.prices)}")
        logger.debug(f"    Remaining timesteps: {available_steps}")
        logger.debug(f"    Starting at step: {self.current_step}")
        
        # Calculate initial metrics
        initial_price = float(self.market_data.prices['close'].iloc[self.current_step])
        initial_volume = float(self.market_data.prices['volume'].iloc[self.current_step])
        
        # Prepare detailed info dict
        info = {
            'equity': float(self.equity),
            'realized_pnl': 0.0,
            'position': None,
            'trade_executed': False,
            'initial_price': initial_price,
            'initial_volume': initial_volume,
            'available_steps': available_steps,
            'window_size': self.config.window_size,
            'reset_count': getattr(self, '_reset_count', 0) + 1,
            'previous_state': previous_state
        }
        
        # Update reset counter
        self._reset_count = info['reset_count']
        
        logger.debug("  Reset complete. Initial info:")
        for key, value in info.items():
            logger.debug(f"    {key}: {value}")
        
        # Log reset information
        self._log_reset_info(
            seed=seed,
            previous_state=previous_state,
            state=state,
            info=info
        )
        
        return state, info
        
    def _ensure_log_dirs(self) -> None:
        """Ensure log directories exist."""
        try:
            # Create main log directory
            log_dir = Path(self.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            for subdir in ["steps", "resets", "trades"]:
                (log_dir / subdir).mkdir(exist_ok=True)
                
        except Exception as e:
            logger.error(f"Failed to create log directories: {str(e)}")
            raise

    def _log_step_info(self, step: int, action: int, reward: float, done: bool, info: dict, state: np.ndarray) -> None:
        """Log step information to file."""
        try:
            # Ensure directories exist
            self._ensure_log_dirs()
            
            step_log = {
                'step': int(step),
                'action': int(action),
                'action_name': Actions(action).name,
                'reward': float(reward),
                'done': bool(done),
                'info': {
                    'equity': float(info['equity']),
                    'equity_change': float(info['equity_change']),
                    'realized_pnl': float(info['realized_pnl']),
                    'unrealized_pnl': float(info['unrealized_pnl']),
                    'position': str(info['position']),
                    'trade_executed': bool(info['trade_executed']),
                    'current_price': float(info['current_price']),
                    'total_trades': int(info['total_trades']),
                    'step_reward': float(info['step_reward'])
                },
                'state_mean': float(state.mean()),
                'state_std': float(state.std())
            }
            
            log_path = Path(self.log_dir) / "steps" / "trading_env_steps.log"
            with open(log_path, "a") as f:
                f.write(f"{json.dumps(step_log)}\n")
                
        except Exception as e:
            logger.error(f"Failed to log step information: {str(e)}")
            
    def _log_reset_info(self, seed: Optional[int], previous_state: dict, state: np.ndarray, info: dict) -> None:
        """Log reset information to file."""
        try:
            # Ensure directories exist
            self._ensure_log_dirs()
            
            reset_log = {
                'timestamp': str(pd.Timestamp.now()),
                'reset_count': int(self._reset_count),
                'seed': None if seed is None else int(seed),
                'previous_state': {
                    'step': None if previous_state['step'] is None else int(previous_state['step']),
                    'equity': None if previous_state['equity'] is None else float(previous_state['equity']),
                    'positions': int(previous_state['positions']),
                    'current_position': str(previous_state['current_position'])
                },
                'initial_state': {
                    'shape': tuple(int(x) for x in state.shape),
                    'mean': float(state.mean()),
                    'std': float(state.std()),
                    'min': float(state.min()),
                    'max': float(state.max())
                },
                'info': {
                    'equity': float(info['equity']),
                    'realized_pnl': float(info['realized_pnl']),
                    'position': str(info['position']),
                    'trade_executed': bool(info['trade_executed']),
                    'initial_price': float(info['initial_price']),
                    'initial_volume': float(info['initial_volume']),
                    'available_steps': int(info['available_steps']),
                    'window_size': int(info['window_size']),
                    'reset_count': int(info['reset_count'])
                }
            }
            
            log_path = Path(self.log_dir) / "resets" / "trading_env_resets.log"
            with open(log_path, "a") as f:
                f.write(f"{json.dumps(reset_log)}\n")
                
        except Exception as e:
            logger.error(f"Failed to log reset information: {str(e)}")
            
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