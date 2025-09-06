"""
Hierarchical Reinforcement Learning agents for empirical validation.

Implements a simplified 2-layer HRL system:
- L1: Strategic Allocation Agent (Portfolio level)
- L3: Execution Agent (Trade execution level)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Try to import stable-baselines3, fall back to simple Q-learning if not available
try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    import gymnasium as gym
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    logging.warning("stable-baselines3 not available, using simple Q-learning implementation")

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Trading actions"""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class Position:
    """Current position information"""
    asset: str = "BTC"
    quantity: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Order:
    """Trading order from strategic to execution level"""
    action: ActionType
    target_quantity: float
    urgency: float  # 0-1, how urgent the execution is
    price_limit: Optional[float] = None
    max_market_impact: float = 0.01  # Max acceptable market impact


class BaseHRLAgent(ABC):
    """Base class for HRL agents"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.is_trained = False
        
    @abstractmethod
    def get_state(self, market_data: pd.DataFrame, **kwargs) -> np.ndarray:
        """Extract state representation from market data"""
        pass
    
    @abstractmethod
    def act(self, state: np.ndarray, **kwargs) -> Any:
        """Choose action based on current state"""
        pass
    
    @abstractmethod
    def learn(self, experience: Dict) -> Dict:
        """Learn from experience"""
        pass


class StrategicAllocationAgent(BaseHRLAgent):
    """
    HRL-L1: Strategic Allocation Agent
    
    Responsibilities:
    - Overall portfolio allocation
    - Risk management at portfolio level
    - Generate orders for execution agents
    """
    
    def __init__(self, initial_cash: float = 100000.0, max_position_size: float = 0.8):
        super().__init__("strategic_agent")
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        self.max_position_size = max_position_size
        self.position = Position()
        self.portfolio_history = []
        
        # Simple Q-table for discrete actions (if no SB3)
        if not HAS_SB3:
            self.q_table = {}
            self.learning_rate = 0.1
            self.gamma = 0.95
            self.epsilon = 0.1
            self.epsilon_decay = 0.995
            
    def get_state(self, market_data: pd.DataFrame, lstm_signals: Dict = None) -> np.ndarray:
        """
        Extract strategic state features.
        
        State includes:
        - Portfolio metrics (cash, position, returns)
        - Market regime indicators
        - LSTM signals
        - Risk metrics
        """
        if len(market_data) < 24:
            # Not enough data, return default state
            return np.zeros(12)
        
        recent_data = market_data.tail(24)
        current_price = market_data.iloc[-1]['close']
        
        # Portfolio features
        total_value = self.current_cash + self.position.quantity * current_price
        portfolio_return = (total_value - self.initial_cash) / self.initial_cash
        cash_ratio = self.current_cash / total_value
        position_ratio = (self.position.quantity * current_price) / total_value
        
        # Market features
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std()
        momentum = (current_price - recent_data.iloc[0]['close']) / recent_data.iloc[0]['close']
        
        # LSTM signals (if available)
        lstm_price_signal = lstm_signals.get('price_signal', 0.0) if lstm_signals else 0.0
        lstm_vol_signal = lstm_signals.get('volatility_signal', 1.0) if lstm_signals else 1.0
        lstm_confidence = lstm_signals.get('confidence', 0.5) if lstm_signals else 0.5
        
        # Risk features
        max_drawdown = self._calculate_max_drawdown()
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        state = np.array([
            portfolio_return,
            cash_ratio,
            position_ratio, 
            volatility,
            momentum,
            lstm_price_signal,
            lstm_vol_signal,
            lstm_confidence,
            max_drawdown,
            sharpe_ratio,
            self.position.unrealized_pnl / self.initial_cash,
            len(self.portfolio_history) / 1000.0  # Normalized time
        ])
        
        return state
    
    def act(self, state: np.ndarray, market_data: pd.DataFrame, **kwargs) -> Order:
        """
        Generate strategic trading order.
        
        Args:
            state: Current state vector
            market_data: Recent market data
            
        Returns:
            Order for execution agent
        """
        current_price = market_data.iloc[-1]['close']
        total_value = self.current_cash + self.position.quantity * current_price
        
        if HAS_SB3 and hasattr(self, 'model'):
            # Use trained RL model
            action = self.model.predict(state.reshape(1, -1), deterministic=True)[0]
        else:
            # Use simple Q-learning or heuristic policy
            action = self._simple_policy(state, current_price, total_value)
        
        # Convert action to order
        if action == ActionType.BUY.value:
            # Buy signal - calculate target position
            target_cash_allocation = min(0.8, max(0.1, state[5] * 0.5 + 0.5))  # Based on LSTM signal
            target_value = total_value * target_cash_allocation
            target_quantity = target_value / current_price
            
            if target_quantity > self.position.quantity:
                return Order(
                    action=ActionType.BUY,
                    target_quantity=target_quantity - self.position.quantity,
                    urgency=min(0.8, abs(state[5])),  # Based on signal strength
                    max_market_impact=0.01
                )
        
        elif action == ActionType.SELL.value:
            # Sell signal
            if self.position.quantity > 0:
                sell_fraction = min(1.0, abs(state[5]) * 0.8 + 0.2)
                target_quantity = self.position.quantity * sell_fraction
                
                return Order(
                    action=ActionType.SELL,
                    target_quantity=target_quantity,
                    urgency=min(0.9, abs(state[5]) + 0.1),
                    max_market_impact=0.015
                )
        
        # Default: HOLD
        return Order(action=ActionType.HOLD, target_quantity=0.0, urgency=0.0)
    
    def _simple_policy(self, state: np.ndarray, current_price: float, total_value: float) -> int:
        """Simple heuristic policy when RL model not available"""
        lstm_signal = state[5]  # LSTM price signal
        confidence = state[7]   # LSTM confidence
        position_ratio = state[2]  # Current position ratio
        volatility = state[3]
        
        # Risk-adjusted signal
        adjusted_signal = lstm_signal * confidence * (1 - volatility)
        
        # Decision thresholds
        if adjusted_signal > 0.3 and position_ratio < self.max_position_size:
            return ActionType.BUY.value
        elif adjusted_signal < -0.3 and position_ratio > 0.1:
            return ActionType.SELL.value
        else:
            return ActionType.HOLD.value
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio history"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        values = [entry['total_value'] for entry in self.portfolio_history]
        peak = values[0]
        max_dd = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
            
        return max_dd
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        annual_return = returns.mean() * 8760  # Hourly to annual
        annual_vol = returns.std() * np.sqrt(8760)
        
        if annual_vol == 0:
            return 0.0
        
        return (annual_return - risk_free_rate) / annual_vol
    
    def learn(self, experience: Dict) -> Dict:
        """Learn from trading experience"""
        if not HAS_SB3:
            return {"message": "Simple policy - no learning implemented"}
        
        # This would be called after each episode to update the RL model
        return {"learning_rate": self.learning_rate if hasattr(self, 'learning_rate') else 0.001}
    
    def update_position(self, executed_quantity: float, executed_price: float, action: ActionType):
        """Update position after trade execution"""
        if action == ActionType.BUY:
            old_quantity = self.position.quantity
            old_entry = self.position.entry_price
            
            self.position.quantity += executed_quantity
            self.current_cash -= executed_quantity * executed_price
            
            # Update entry price (weighted average)
            if old_quantity > 0:
                total_cost = old_quantity * old_entry + executed_quantity * executed_price
                self.position.entry_price = total_cost / self.position.quantity
            else:
                self.position.entry_price = executed_price
                
        elif action == ActionType.SELL:
            self.position.quantity -= executed_quantity
            self.current_cash += executed_quantity * executed_price
            
            # Calculate realized PnL
            if self.position.entry_price > 0:
                pnl = executed_quantity * (executed_price - self.position.entry_price)
                self.position.realized_pnl += pnl


class ExecutionAgent(BaseHRLAgent):
    """
    HRL-L3: Execution Agent
    
    Responsibilities:
    - Optimize trade execution
    - Minimize market impact and slippage
    - Handle order splitting and timing
    """
    
    def __init__(self, max_order_size: float = 0.05):  # Max 5% of volume
        super().__init__("execution_agent")
        self.max_order_size = max_order_size
        self.execution_history = []
        
    def get_state(self, market_data: pd.DataFrame, order: Order) -> np.ndarray:
        """
        Extract execution state features.
        
        State includes:
        - Order characteristics
        - Market microstructure
        - Recent execution performance
        """
        if len(market_data) < 5:
            return np.zeros(10)
        
        recent_data = market_data.tail(5)
        current_price = market_data.iloc[-1]['close']
        
        # Order features
        order_size_ratio = order.target_quantity / (recent_data['volume'].mean() + 1e-8)
        urgency = order.urgency
        
        # Market microstructure
        spread = 0.001  # Simplified spread
        volatility = recent_data['close'].pct_change().std()
        volume_ratio = recent_data.iloc[-1]['volume'] / recent_data['volume'].mean()
        
        # Price momentum (for timing)
        short_momentum = (current_price - recent_data.iloc[-3]['close']) / recent_data.iloc[-3]['close']
        
        # Market impact estimation
        estimated_impact = min(0.05, order_size_ratio * 0.01)
        
        # Recent execution performance
        avg_slippage = np.mean([ex.get('slippage', 0) for ex in self.execution_history[-10:]]) if self.execution_history else 0
        
        state = np.array([
            order_size_ratio,
            urgency,
            spread,
            volatility,
            volume_ratio,
            short_momentum,
            estimated_impact,
            avg_slippage,
            float(order.action.value),
            len(self.execution_history) / 100.0  # Normalized experience
        ])
        
        return state
    
    def act(self, state: np.ndarray, order: Order, market_data: pd.DataFrame) -> Dict:
        """
        Execute the order with optimized parameters.
        
        Args:
            state: Current execution state
            order: Order to execute
            market_data: Current market data
            
        Returns:
            Execution result
        """
        if order.action == ActionType.HOLD:
            return {"executed_quantity": 0.0, "executed_price": 0.0, "slippage": 0.0}
        
        current_price = market_data.iloc[-1]['close']
        current_volume = market_data.iloc[-1]['volume']
        
        # Simple execution strategy based on urgency and market conditions
        urgency = order.urgency
        volatility = state[3]
        volume_ratio = state[4]
        
        # Determine execution parameters
        if urgency > 0.8:
            # High urgency - execute immediately (market order style)
            executed_quantity = order.target_quantity
            slippage = 0.001 + volatility * 0.5  # Higher slippage for urgent orders
            
        elif urgency > 0.4:
            # Medium urgency - execute with some market impact
            executed_quantity = min(order.target_quantity, current_volume * self.max_order_size)
            slippage = 0.0005 + volatility * 0.2
            
        else:
            # Low urgency - careful execution
            max_safe_quantity = current_volume * self.max_order_size * 0.5
            executed_quantity = min(order.target_quantity, max_safe_quantity)
            slippage = 0.0002 + volatility * 0.1
        
        # Apply slippage based on order direction
        if order.action == ActionType.BUY:
            executed_price = current_price * (1 + slippage)
        else:
            executed_price = current_price * (1 - slippage)
        
        # Record execution
        execution_result = {
            "executed_quantity": executed_quantity,
            "executed_price": executed_price,
            "slippage": slippage,
            "market_impact": slippage,  # Simplified
            "timestamp": len(market_data)
        }
        
        self.execution_history.append(execution_result)
        
        return execution_result
    
    def learn(self, experience: Dict) -> Dict:
        """Learn from execution experience"""
        # Simple learning: adjust execution parameters based on recent performance
        if len(self.execution_history) > 10:
            recent_slippage = np.mean([ex['slippage'] for ex in self.execution_history[-10:]])
            if recent_slippage > 0.002:
                self.max_order_size *= 0.95  # Reduce order size if slippage too high
            elif recent_slippage < 0.0005:
                self.max_order_size *= 1.02  # Increase if performing well
                
            self.max_order_size = np.clip(self.max_order_size, 0.01, 0.1)
        
        return {"avg_slippage": np.mean([ex['slippage'] for ex in self.execution_history[-10:]]) if len(self.execution_history) > 0 else 0}


class HierarchicalTradingSystem:
    """
    Main HRL trading system coordinator.
    
    Coordinates between Strategic and Execution agents.
    """
    
    def __init__(self, initial_cash: float = 100000.0):
        self.strategic_agent = StrategicAllocationAgent(initial_cash)
        self.execution_agent = ExecutionAgent()
        self.performance_history = []
        
    def step(self, market_data: pd.DataFrame, lstm_signals: Dict = None) -> Dict:
        """
        Execute one trading step.
        
        Args:
            market_data: Current market data
            lstm_signals: Signals from LSTM agent
            
        Returns:
            Step results and metrics
        """
        # Get strategic decision
        strategic_state = self.strategic_agent.get_state(market_data, lstm_signals)
        order = self.strategic_agent.act(strategic_state, market_data)
        
        # Execute the order
        execution_state = self.execution_agent.get_state(market_data, order)
        execution_result = self.execution_agent.act(execution_state, order, market_data)
        
        # Update strategic agent position
        if execution_result["executed_quantity"] > 0:
            self.strategic_agent.update_position(
                execution_result["executed_quantity"],
                execution_result["executed_price"],
                order.action
            )
        
        # Record performance
        current_price = market_data.iloc[-1]['close']
        total_value = (self.strategic_agent.current_cash + 
                      self.strategic_agent.position.quantity * current_price)
        
        step_result = {
            "timestamp": market_data.index[-1] if isinstance(market_data.index, pd.DatetimeIndex) else len(market_data),
            "order": order,
            "execution": execution_result,
            "portfolio_value": total_value,
            "cash": self.strategic_agent.current_cash,
            "position": self.strategic_agent.position.quantity,
            "unrealized_pnl": self.strategic_agent.position.quantity * (current_price - self.strategic_agent.position.entry_price) if self.strategic_agent.position.entry_price > 0 else 0,
            "realized_pnl": self.strategic_agent.position.realized_pnl
        }
        
        self.performance_history.append(step_result)
        
        return step_result
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics for the trading system"""
        if len(self.performance_history) < 2:
            return {}
        
        values = [entry['portfolio_value'] for entry in self.performance_history]
        returns = pd.Series(values).pct_change().dropna()
        
        total_return = (values[-1] - values[0]) / values[0]
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(8760) if returns.std() > 0 else 0
        max_drawdown = self.strategic_agent._calculate_max_drawdown()
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": sum(1 for entry in self.performance_history if entry['execution']['executed_quantity'] > 0),
            "final_value": values[-1],
            "win_rate": self._calculate_win_rate()
        }
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate of trades"""
        trades = [entry for entry in self.performance_history if entry['execution']['executed_quantity'] > 0]
        if len(trades) < 2:
            return 0.0
        
        winning_trades = 0
        for i in range(1, len(trades)):
            if trades[i]['portfolio_value'] > trades[i-1]['portfolio_value']:
                winning_trades += 1
        
        return winning_trades / len(trades) if trades else 0.0


if __name__ == "__main__":
    # Example usage
    from ..simulation.market_simulator import MarketDataSimulator
    from .lstm_agent import LSTMForecastingAgent
    
    # Generate test data
    simulator = MarketDataSimulator(random_seed=42)
    data = simulator.generate_dataset(n_days=5)
    
    # Initialize system
    hrl_system = HierarchicalTradingSystem()
    
    # Run simulation
    for i in range(50, len(data), 6):  # Every 6 hours
        window_data = data.iloc[:i]
        step_result = hrl_system.step(window_data)
        
        if i % 24 == 0:  # Print daily updates
            print(f"Step {i}: Portfolio Value = ${step_result['portfolio_value']:.2f}")
    
    # Print final performance
    metrics = hrl_system.get_performance_metrics()
    print("Final Performance:", metrics)