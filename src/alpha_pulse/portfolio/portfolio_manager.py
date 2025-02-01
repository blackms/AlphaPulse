"""
Portfolio Manager implementation.
Coordinates portfolio strategies, risk management, and rebalancing operations.
"""

import yaml
from typing import Dict, Optional, Type, List
import pandas as pd
import numpy as np
from pathlib import Path
from decimal import Decimal
from collections import defaultdict
from datetime import datetime, UTC

from .interfaces import IRebalancingStrategy
from .data_models import Position, PortfolioData
from .strategies.mpt_strategy import MPTStrategy
from .strategies.hrp_strategy import HRPStrategy
from .strategies.black_litterman_strategy import BlackLittermanStrategy
from .strategies.llm_assisted_strategy import LLMAssistedStrategy


def create_price_matrix(historical_data: List[Dict], assets: List[str]) -> pd.DataFrame:
    """
    Create a price matrix from historical data.

    Args:
        historical_data: List of dictionaries with timestamp and asset prices
        assets: List of asset symbols

    Returns:
        DataFrame with timestamps as index and assets as columns
    """
    # Group data by timestamp
    data_by_timestamp = defaultdict(dict)
    timestamps = set()
    
    for entry in historical_data:
        timestamp = entry['timestamp']
        timestamps.add(timestamp)
        for asset, price in entry.items():
            if asset != 'timestamp':
                data_by_timestamp[timestamp][asset] = price

    # Create DataFrame with all timestamps and assets
    price_matrix = pd.DataFrame(index=sorted(timestamps))
    for asset in assets:
        prices = []
        for timestamp in sorted(timestamps):
            # Use previous price if missing, or 1.0 for stablecoins
            if asset in data_by_timestamp[timestamp]:
                prices.append(data_by_timestamp[timestamp][asset])
            else:
                last_price = prices[-1] if prices else (1.0 if asset.endswith('USDT') else None)
                prices.append(last_price)
        price_matrix[asset] = prices

    # Forward fill any remaining NaN values
    price_matrix = price_matrix.ffill()
    
    # For any columns that are still all NaN, fill with 1.0 (stablecoins)
    for col in price_matrix.columns:
        if price_matrix[col].isna().all():
            price_matrix[col] = 1.0

    return price_matrix


class PortfolioManager:
    """Manages portfolio allocation and rebalancing operations."""

    STRATEGY_MAP = {
        'mpt': MPTStrategy,
        'hierarchical_risk_parity': HRPStrategy,
        'black_litterman': BlackLittermanStrategy
    }

    def __init__(self, config_path: str):
        """
        Initialize portfolio manager with configuration.

        Args:
            config_path: Path to portfolio configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.strategy = self._initialize_strategy()
        self.risk_constraints = self._get_risk_constraints()
        self.last_rebalance_time = None
        self.trading_config = self.config.get('trading', {})
        self.base_currency = self.trading_config.get('base_currency', 'USDT')

    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _initialize_strategy(self) -> IRebalancingStrategy:
        """
        Initialize the appropriate portfolio strategy based on configuration.

        Returns:
            Configured strategy instance
        """
        strategy_name = self.config['strategy']['name']
        strategy_class = self.STRATEGY_MAP.get(strategy_name)
        
        if not strategy_class:
            raise ValueError(f"Unknown strategy: {strategy_name}")
            
        # Initialize base strategy
        base_strategy = strategy_class(self.config)
        
        # Wrap with LLM if enabled
        if self.config.get('llm', {}).get('enabled', False):
            return LLMAssistedStrategy(base_strategy, self.config)
            
        return base_strategy

    def _get_risk_constraints(self) -> Dict[str, float]:
        """
        Extract risk constraints from configuration.

        Returns:
            Dictionary of risk constraints
        """
        return {
            'volatility_target': self.config.get('volatility_target', 0.15),
            'max_drawdown_limit': self.config.get('max_drawdown_limit', 0.25),
            'correlation_threshold': self.config.get('correlation_threshold', 0.7)
        }

    async def get_current_allocation(self, exchange) -> Dict[str, float]:
        """
        Get current portfolio allocation from exchange.

        Args:
            exchange: Exchange interface instance

        Returns:
            Dictionary of current asset weights
        """
        # Get account balances
        balances = await exchange.get_balances()
        
        # Get current prices for conversion to common currency
        prices = {}
        for asset, balance in balances.items():
            if asset == self.base_currency:
                prices[asset] = Decimal('1.0')
            else:
                price = await exchange.get_ticker_price(f"{asset}/{self.base_currency}")
                if price:
                    prices[asset] = price
        
        # Calculate total portfolio value and weights
        total_value = sum(
            balance.total * prices.get(asset, Decimal('1.0'))
            for asset, balance in balances.items()
        )
        
        if total_value == Decimal('0'):
            return {}
        
        return {
            asset: balance.total * prices.get(asset, Decimal('1.0')) / total_value
            for asset, balance in balances.items()
            if balance.total > 0
        }

    def compute_rebalancing_trades(
        self,
        current: Dict[str, Decimal],
        target: Dict[str, Decimal],
        total_value: Decimal
    ) -> List[Dict]:
        """
        Compute required trades to achieve target allocation.

        Args:
            current: Current portfolio weights
            target: Target portfolio weights
            total_value: Total portfolio value in base currency

        Returns:
            List of trade dictionaries
        """
        trades = []
        min_trade_value = Decimal(str(self.trading_config.get('min_trade_value', 10.0)))
        rebalancing_threshold = Decimal(str(self.config['strategy']['rebalancing_threshold']))
        
        # Compare current and target allocations
        all_assets = set(current) | set(target)
        for asset in all_assets:
            current_weight = current.get(asset, Decimal('0'))
            target_weight = target.get(asset, Decimal('0'))
            
            weight_diff = abs(current_weight - target_weight)
            if weight_diff > rebalancing_threshold:
                trade_value = (target_weight - current_weight) * total_value
                
                # Skip small trades
                if abs(trade_value) < min_trade_value:
                    continue
                
                trades.append({
                    'asset': asset,
                    'value': trade_value,
                    'type': 'buy' if trade_value > 0 else 'sell',
                    'weight_change': target_weight - current_weight
                })
                
        return trades

    async def needs_rebalancing(
        self,
        exchange,
        current_allocation: Optional[Dict[str, Decimal]] = None
    ) -> bool:
        """
        Check if portfolio needs rebalancing based on time and deviation.

        Args:
            exchange: Exchange interface instance
            current_allocation: Optional current allocation (will be fetched if not provided)

        Returns:
            Boolean indicating if rebalancing is needed
        """
        if not self.last_rebalance_time:
            return True
            
        # Check rebalancing frequency
        frequency = pd.Timedelta(self.config['rebalancing_frequency'])
        if pd.Timestamp.now() - self.last_rebalance_time < frequency:
            return False
            
        # Get current allocation if not provided
        if current_allocation is None:
            current_allocation = await self.get_current_allocation(exchange)
            
        # Get historical data for analysis
        lookback_days = self.config['strategy']['lookback_period']
        end_time = pd.Timestamp.now()
        start_time = end_time - pd.Timedelta(days=lookback_days)
        
        # Get historical data for each non-stablecoin asset
        price_data = []
        for asset in current_allocation.keys():
            if asset == self.base_currency:
                continue
                
            symbol = f"{asset}/{self.base_currency}"
            candles = await exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe="1d",
                since=int(start_time.timestamp() * 1000),
                limit=lookback_days
            )
            
            for candle in candles:
                price_data.append({
                    'timestamp': candle.timestamp,
                    asset: float(candle.close)
                })
        
        # Create price matrix
        price_matrix = create_price_matrix(
            price_data,
            list(current_allocation.keys())
        )
        
        # Get target allocation
        target = self.strategy.compute_target_allocation(
            current_allocation,
            price_matrix,
            self.risk_constraints
        )
        
        # Check if any asset deviates more than threshold
        threshold = Decimal(str(self.config['strategy']['rebalancing_threshold']))
        for asset in set(current_allocation) | set(target):
            current = current_allocation.get(asset, Decimal('0'))
            target_weight = Decimal(str(target.get(asset, 0)))
            if abs(current - target_weight) > threshold:
                return True
                
        return False

    async def get_portfolio_data(self, exchange) -> PortfolioData:
        """
        Get current portfolio data from exchange.

        Args:
            exchange: Exchange interface instance

        Returns:
            PortfolioData object containing current portfolio state
        """
        # Get current allocation and portfolio value
        current_allocation = await self.get_current_allocation(exchange)
        total_value = await exchange.get_portfolio_value()
        
        # Get positions data
        positions = []
        for asset, weight in current_allocation.items():
            if asset == self.base_currency:
                continue
                
            symbol = f"{asset}/{self.base_currency}"
            current_price = await exchange.get_ticker_price(symbol)
            
            # Get average entry price from order history
            avg_entry_price = await exchange.get_average_entry_price(symbol)
            if not avg_entry_price:
                avg_entry_price = current_price  # Fallback to current price if no history

            # Calculate quantity from weight and total value
            quantity = (weight * total_value) / current_price
            
            # Calculate profit/loss
            profit_loss = (current_price - avg_entry_price) * quantity if avg_entry_price else Decimal('0')
            
            positions.append(Position(
                asset_id=asset,
                quantity=quantity,
                entry_price=avg_entry_price,
                current_price=current_price,
                timestamp=datetime.now(UTC)
            ))

        # Create portfolio data object
        return PortfolioData(
            positions=positions,
            total_value=total_value,
            cash_balance=current_allocation.get(self.base_currency, Decimal('0')) * total_value,
            timestamp=datetime.now(UTC),
            risk_metrics={
                'volatility_target': self.risk_constraints['volatility_target'],
                'max_drawdown_limit': self.risk_constraints['max_drawdown_limit'],
                'correlation_threshold': self.risk_constraints['correlation_threshold']
            }
        )

    async def analyze_portfolio_with_llm(
        self,
        analyzer: 'OpenAILLMAnalyzer',
        exchange
    ) -> 'LLMAnalysisResult':
        """
        Analyze the current portfolio using LLM.

        Args:
            analyzer: Configured OpenAILLMAnalyzer instance
            exchange: Exchange interface instance

        Returns:
            LLMAnalysisResult containing the analysis
        """
        # Get portfolio data
        portfolio_data = await self.get_portfolio_data(exchange)
        
        # Get LLM analysis
        return await analyzer.analyze_portfolio(portfolio_data)

    async def rebalance_portfolio(
        self,
        exchange,
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Execute full portfolio rebalancing operation.

        Args:
            exchange: Exchange interface instance
            historical_data: Optional historical price data

        Returns:
            Dictionary containing rebalancing results
        """
        # Get current allocation
        current_allocation = await self.get_current_allocation(exchange)
        
        if not await self.needs_rebalancing(exchange, current_allocation):
            return {
                'status': 'skipped',
                'reason': 'Rebalancing not needed'
            }
            
        # Get historical data if not provided
        if historical_data is None:
            lookback_days = self.config['strategy']['lookback_period']
            end_time = pd.Timestamp.now()
            start_time = end_time - pd.Timedelta(days=lookback_days)
            
            # Get historical data for each non-stablecoin asset
            price_data = []
            for asset in current_allocation.keys():
                if asset == self.base_currency:
                    continue
                    
                symbol = f"{asset}/{self.base_currency}"
                candles = await exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe="1d",
                    since=int(start_time.timestamp() * 1000),
                    limit=lookback_days
                )
                
                for candle in candles:
                    price_data.append({
                        'timestamp': candle.timestamp,
                        asset: float(candle.close)
                    })
            
            # Create price matrix
            historical_data = create_price_matrix(
                price_data,
                list(current_allocation.keys())
            )
            
        # Compute target allocation
        target_allocation = self.strategy.compute_target_allocation(
            current_allocation,
            historical_data,
            self.risk_constraints
        )
        
        # Validate target allocation
        if not self.strategy.validate_constraints(target_allocation):
            return {
                'status': 'failed',
                'reason': 'Target allocation violates constraints'
            }
            
        # Compute required trades
        total_value = await exchange.get_portfolio_value()
        trades = self.compute_rebalancing_trades(
            current_allocation,
            {k: Decimal(str(v)) for k, v in target_allocation.items()},
            total_value
        )
        
        # Execute trades
        executed_trades = []
        for trade in trades:
            try:
                result = await exchange.execute_trade(
                    asset=trade['asset'],
                    amount=abs(trade['value']),
                    side=trade['type'],
                    order_type=self.trading_config.get('execution_style', 'market')
                )
                executed_trades.append({
                    **trade,
                    'status': 'success',
                    'execution_details': result
                })
            except Exception as e:
                executed_trades.append({
                    **trade,
                    'status': 'failed',
                    'error': str(e)
                })
                
        self.last_rebalance_time = pd.Timestamp.now()
        
        return {
            'status': 'completed',
            'initial_allocation': current_allocation,
            'target_allocation': target_allocation,
            'trades': executed_trades
        }