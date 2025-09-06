"""
Baseline trading strategies for performance comparison.

Implements simple strategies to benchmark against the HRL system:
- Buy and Hold
- Moving Average Crossover
- Random Walk
- Single RL Agent (non-hierarchical)
- Mean Reversion
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Single trade result"""
    timestamp: pd.Timestamp
    action: str  # 'buy', 'sell', 'hold'
    quantity: float
    price: float
    portfolio_value: float
    cash: float
    position: float


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, initial_cash: float = 100000.0, name: str = "BaseStrategy"):
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        self.position = 0.0
        self.name = name
        self.trades = []
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, index: int) -> Dict[str, float]:
        """Generate trading signal based on current market data"""
        pass
    
    def execute_trade(self, signal: Dict[str, float], current_price: float, timestamp: pd.Timestamp) -> TradeResult:
        """Execute trade based on signal"""
        action = signal.get('action', 'hold')
        target_position = signal.get('target_position', self.position)
        
        # Calculate trade quantity
        quantity_to_trade = target_position - self.position
        
        if abs(quantity_to_trade) < 0.01:  # Minimum trade threshold
            action = 'hold'
            quantity_to_trade = 0
        
        # Execute trade
        if quantity_to_trade > 0:  # Buy
            max_buyable = self.current_cash / current_price
            quantity_to_trade = min(quantity_to_trade, max_buyable)
            
            if quantity_to_trade > 0.01:
                self.position += quantity_to_trade
                self.current_cash -= quantity_to_trade * current_price
                action = 'buy'
            else:
                quantity_to_trade = 0
                action = 'hold'
                
        elif quantity_to_trade < 0:  # Sell
            quantity_to_trade = max(quantity_to_trade, -self.position)
            
            if abs(quantity_to_trade) > 0.01:
                self.position += quantity_to_trade  # quantity_to_trade is negative
                self.current_cash -= quantity_to_trade * current_price  # Adds cash (negative * negative)
                action = 'sell'
            else:
                quantity_to_trade = 0
                action = 'hold'
        
        # Calculate portfolio value
        portfolio_value = self.current_cash + self.position * current_price
        
        # Record trade
        trade_result = TradeResult(
            timestamp=timestamp,
            action=action,
            quantity=abs(quantity_to_trade),
            price=current_price,
            portfolio_value=portfolio_value,
            cash=self.current_cash,
            position=self.position
        )
        
        self.trades.append(trade_result)
        return trade_result
    
    def run_backtest(self, data: pd.DataFrame) -> List[TradeResult]:
        """Run complete backtest on historical data"""
        logger.info(f"Running backtest for {self.name} on {len(data)} data points")
        
        results = []
        
        for i in range(max(1, 20), len(data)):  # Start after warmup period
            current_data = data.iloc[:i+1]
            signal = self.generate_signal(current_data, i)
            
            current_price = data.iloc[i]['close']
            timestamp = data.index[i] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.now()
            
            result = self.execute_trade(signal, current_price, timestamp)
            results.append(result)
        
        logger.info(f"Backtest completed. Final portfolio value: ${results[-1].portfolio_value:.2f}")
        return results
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if len(self.trades) < 2:
            return {}
        
        values = [trade.portfolio_value for trade in self.trades]
        returns = pd.Series(values).pct_change().dropna()
        
        total_return = (values[-1] - values[0]) / values[0]
        
        # Annualized metrics (assuming hourly data)
        annual_return = returns.mean() * 8760
        annual_vol = returns.std() * np.sqrt(8760)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Max drawdown
        peak = values[0]
        max_dd = 0.0
        for value in values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        # Trade statistics
        total_trades = sum(1 for trade in self.trades if trade.action != 'hold')
        win_rate = self._calculate_win_rate()
        
        return {
            'strategy': self.name,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'final_value': values[-1]
        }
    
    def _calculate_win_rate(self) -> float:
        """Calculate percentage of profitable trades"""
        if len(self.trades) < 2:
            return 0.0
        
        profitable_trades = 0
        total_trades = 0
        
        for i in range(1, len(self.trades)):
            if self.trades[i].action != 'hold':
                total_trades += 1
                if self.trades[i].portfolio_value > self.trades[i-1].portfolio_value:
                    profitable_trades += 1
        
        return profitable_trades / total_trades if total_trades > 0 else 0.0


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy and hold strategy"""
    
    def __init__(self, initial_cash: float = 100000.0):
        super().__init__(initial_cash, "Buy and Hold")
        self.bought = False
        
    def generate_signal(self, data: pd.DataFrame, index: int) -> Dict[str, float]:
        """Buy once at the beginning, then hold"""
        if not self.bought:
            # Buy everything at the first opportunity
            current_price = data.iloc[index]['close']
            target_position = self.initial_cash / current_price
            self.bought = True
            return {'action': 'buy', 'target_position': target_position}
        else:
            return {'action': 'hold', 'target_position': self.position}


class MovingAverageCrossoverStrategy(BaseStrategy):
    """Moving average crossover strategy"""
    
    def __init__(self, initial_cash: float = 100000.0, short_window: int = 5, long_window: int = 20):
        super().__init__(initial_cash, f"MA Crossover ({short_window},{long_window})")
        self.short_window = short_window
        self.long_window = long_window
        
    def generate_signal(self, data: pd.DataFrame, index: int) -> Dict[str, float]:
        """Generate signal based on moving average crossover"""
        if len(data) < self.long_window:
            return {'action': 'hold', 'target_position': self.position}
        
        # Calculate moving averages
        ma_short = data['close'].tail(self.short_window).mean()
        ma_long = data['close'].tail(self.long_window).mean()
        
        current_price = data.iloc[index]['close']
        
        # Generate signal
        if ma_short > ma_long:
            # Golden cross - buy signal
            target_position = self.current_cash * 0.8 / current_price  # Use 80% of cash
            return {'action': 'buy', 'target_position': target_position}
        elif ma_short < ma_long and self.position > 0:
            # Death cross - sell signal
            return {'action': 'sell', 'target_position': 0}
        else:
            return {'action': 'hold', 'target_position': self.position}


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy based on z-score"""
    
    def __init__(self, initial_cash: float = 100000.0, window: int = 20, threshold: float = 2.0):
        super().__init__(initial_cash, f"Mean Reversion (z>{threshold})")
        self.window = window
        self.threshold = threshold
        
    def generate_signal(self, data: pd.DataFrame, index: int) -> Dict[str, float]:
        """Generate signal based on price deviation from mean"""
        if len(data) < self.window:
            return {'action': 'hold', 'target_position': self.position}
        
        # Calculate z-score
        recent_prices = data['close'].tail(self.window)
        mean_price = recent_prices.mean()
        std_price = recent_prices.std()
        current_price = data.iloc[index]['close']
        
        if std_price == 0:
            return {'action': 'hold', 'target_position': self.position}
        
        z_score = (current_price - mean_price) / std_price
        
        # Generate signal
        if z_score < -self.threshold:
            # Price below mean - buy signal
            target_position = self.current_cash * 0.6 / current_price
            return {'action': 'buy', 'target_position': target_position}
        elif z_score > self.threshold and self.position > 0:
            # Price above mean - sell signal
            return {'action': 'sell', 'target_position': 0}
        else:
            return {'action': 'hold', 'target_position': self.position}


class MomentumStrategy(BaseStrategy):
    """Momentum strategy based on recent returns"""
    
    def __init__(self, initial_cash: float = 100000.0, lookback: int = 12, threshold: float = 0.02):
        super().__init__(initial_cash, f"Momentum ({lookback}h, {threshold:.1%})")
        self.lookback = lookback
        self.threshold = threshold
        
    def generate_signal(self, data: pd.DataFrame, index: int) -> Dict[str, float]:
        """Generate signal based on price momentum"""
        if len(data) < self.lookback + 1:
            return {'action': 'hold', 'target_position': self.position}
        
        # Calculate momentum
        current_price = data.iloc[index]['close']
        past_price = data.iloc[index - self.lookback]['close']
        momentum = (current_price - past_price) / past_price
        
        # Generate signal
        if momentum > self.threshold:
            # Strong positive momentum - buy signal
            target_position = self.current_cash * 0.7 / current_price
            return {'action': 'buy', 'target_position': target_position}
        elif momentum < -self.threshold and self.position > 0:
            # Strong negative momentum - sell signal
            return {'action': 'sell', 'target_position': 0}
        else:
            return {'action': 'hold', 'target_position': self.position}


class RandomStrategy(BaseStrategy):
    """Random trading strategy for comparison"""
    
    def __init__(self, initial_cash: float = 100000.0, trade_probability: float = 0.05):
        super().__init__(initial_cash, "Random")
        self.trade_probability = trade_probability
        np.random.seed(42)  # For reproducibility
        
    def generate_signal(self, data: pd.DataFrame, index: int) -> Dict[str, float]:
        """Generate random trading signals"""
        if np.random.random() < self.trade_probability:
            action = np.random.choice(['buy', 'sell'])
            current_price = data.iloc[index]['close']
            
            if action == 'buy':
                target_position = min(self.position + self.current_cash * 0.2 / current_price, 
                                    self.current_cash * 0.8 / current_price)
                return {'action': 'buy', 'target_position': target_position}
            elif action == 'sell' and self.position > 0:
                target_position = max(0, self.position * 0.8)
                return {'action': 'sell', 'target_position': target_position}
        
        return {'action': 'hold', 'target_position': self.position}


class BaselineComparison:
    """Class to run and compare all baseline strategies"""
    
    def __init__(self, initial_cash: float = 100000.0):
        self.strategies = [
            BuyAndHoldStrategy(initial_cash),
            MovingAverageCrossoverStrategy(initial_cash, short_window=5, long_window=20),
            MovingAverageCrossoverStrategy(initial_cash, short_window=10, long_window=50),
            MeanReversionStrategy(initial_cash, window=20, threshold=2.0),
            MomentumStrategy(initial_cash, lookback=12, threshold=0.02),
            RandomStrategy(initial_cash, trade_probability=0.05)
        ]
        
    def run_all_backtests(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Run backtests for all baseline strategies"""
        logger.info(f"Running baseline comparison on {len(data)} data points")
        
        results = {}
        
        for strategy in self.strategies:
            try:
                strategy.run_backtest(data.copy())
                metrics = strategy.get_performance_metrics()
                results[strategy.name] = metrics
                
                logger.info(f"{strategy.name}: Total Return = {metrics.get('total_return', 0):.2%}, "
                           f"Sharpe = {metrics.get('sharpe_ratio', 0):.2f}")
                           
            except Exception as e:
                logger.error(f"Error running {strategy.name}: {e}")
                results[strategy.name] = {'error': str(e)}
        
        return results
    
    def get_comparison_table(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Create comparison table of all strategies"""
        comparison_data = []
        
        for strategy_name, metrics in results.items():
            if 'error' not in metrics:
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Total Return': f"{metrics.get('total_return', 0):.2%}",
                    'Annual Return': f"{metrics.get('annual_return', 0):.2%}",
                    'Annual Volatility': f"{metrics.get('annual_volatility', 0):.2%}",
                    'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                    'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                    'Total Trades': int(metrics.get('total_trades', 0)),
                    'Win Rate': f"{metrics.get('win_rate', 0):.2%}",
                    'Final Value': f"${metrics.get('final_value', 0):.2f}"
                })
        
        return pd.DataFrame(comparison_data)


if __name__ == "__main__":
    # Example usage
    from ..simulation.market_simulator import MarketDataSimulator
    
    # Generate synthetic data
    simulator = MarketDataSimulator(random_seed=42)
    data = simulator.generate_dataset(n_days=30)
    
    # Run baseline comparison
    comparison = BaselineComparison()
    results = comparison.run_all_backtests(data)
    
    # Display results
    comparison_table = comparison.get_comparison_table(results)
    print("\nBaseline Strategy Comparison:")
    print(comparison_table.to_string(index=False))