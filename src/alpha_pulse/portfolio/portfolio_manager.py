"""
Portfolio Manager implementation.
Coordinates portfolio strategies, risk management, and rebalancing operations.
"""

import yaml
from typing import Dict, Optional, Type, List, Any, Union, Set
import pandas as pd
import numpy as np
from pathlib import Path
from decimal import Decimal, DivisionByZero
from collections import defaultdict
from datetime import datetime, timezone
import asyncio
from functools import lru_cache
from asyncio import TimeoutError
from loguru import logger
from contextlib import asynccontextmanager

from .interfaces import IRebalancingStrategy
from .data_models import Position, PortfolioData
from .strategies.mpt_strategy import MPTStrategy
from .strategies.hrp_strategy import HRPStrategy
from .strategies.black_litterman_strategy import BlackLittermanStrategy
from .strategies.llm_assisted_strategy import LLMAssistedStrategy
from .llm_analysis import OpenAILLMAnalyzer, LLMAnalysisResult
from alpha_pulse.decorators.audit_decorators import (
    audit_portfolio_action,
    audit_trade_decision,
    audit_risk_check
)
from alpha_pulse.hedging.risk.manager import HedgeManager




@lru_cache(maxsize=100)
def create_price_matrix(historical_data_str: str, assets_str: str) -> pd.DataFrame:
    """
    Create a price matrix from historical data.

    Args:
        historical_data: List of dictionaries with timestamp and asset prices
        assets: List of asset symbols

    Returns:
        DataFrame with timestamps as index and assets as columns
    """
    # Convert string inputs to objects for cache compatibility
    historical_data = eval(historical_data_str)  # Safe since we control the input
    assets = eval(assets_str)
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
                last_price = prices[-1] if prices else (1.0 if asset.endswith(('USDT', 'USDC', 'DAI', 'BUSD', 'UST', 'TUSD')) else None)
                prices.append(last_price)
        price_matrix[asset] = prices

    # Forward fill any remaining NaN values
    price_matrix = price_matrix.ffill(limit=5)  # Limit forward fill to 5 periods
    
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
        'black_litterman': BlackLittermanStrategy,
        'llm_assisted': LLMAssistedStrategy
    }

    def __init__(self, config_path: str, hedge_manager: HedgeManager = None):
        """
        Initialize portfolio manager with configuration.

        Args:
            config_path: Path to portfolio configuration YAML file
            hedge_manager: Optional hedge manager for tail risk protection
        """
        logger.info(f"Initializing PortfolioManager with config from {config_path}")
        self.config = self._load_config(config_path)
        self._validate_config()
        self.strategy = self._initialize_strategy()
        self.risk_constraints = self._get_risk_constraints()
        self.last_rebalance_time = None
        self.trading_config = self.config.get('trading', {})
        self.base_currency = self.trading_config.get('base_currency', 'USDT')
        
        # Initialize stablecoin patterns
        self.stablecoin_patterns: Set[str] = {
            'USDT', 'USDC', 'DAI', 'BUSD', 'UST', 'TUSD'
        }
        
        # Tail risk hedging integration
        self.hedge_manager = hedge_manager
        self.tail_risk_enabled = config.get('tail_risk_hedging', {}).get('enabled', True) if hedge_manager else False
        self.tail_risk_threshold = config.get('tail_risk_hedging', {}).get('threshold', 0.05)  # 5% tail risk

    def _validate_config(self) -> None:
        """Validate configuration fields."""
        required_fields = {
            'strategy': {
                'name': str,
                'lookback_period': int,
                'rebalancing_threshold': (int, float)
            },
            'rebalancing_frequency': str,
            'trading': {
                'base_currency': str,
                'min_trade_value': (int, float)
            }
        }
        
        def validate_dict(config: Dict, fields: Dict, path: str = '') -> None:
            for key, value in fields.items():
                if key not in config:
                    raise ValueError(f"Missing required config field: {path + key}")
                if isinstance(value, dict):
                    validate_dict(config[key], value, f"{path}{key}.")
                    
        validate_dict(self.config, required_fields)

    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except (yaml.YAMLError, IOError) as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ValueError(f"Failed to load configuration from {config_path}: {str(e)}")
            
        return config
        
    def _is_stablecoin(self, asset: str) -> bool:
        """
        Check if an asset is a stablecoin.
        
        Args:
            asset: Asset symbol to check
        """
        return any(asset.endswith(pattern) for pattern in self.stablecoin_patterns)

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

    async def get_current_allocation(self, exchange: Any) -> Dict[str, float]:
        """
        Get current portfolio allocation from exchange.

        Args:
            exchange: Exchange interface instance

        Returns:
            Dictionary of current asset weights
            
        Raises:
            ValueError: If exchange interface is invalid or operations fail
        """
        self._validate_exchange(exchange)
        
        # Get account balances with retry and timeout
        balances = await self._retry_with_timeout(
            lambda: exchange.get_balances(),
            max_retries=3,
            timeout=15.0
        )
        
        # Get current prices for conversion to common currency
        prices = {}
        for asset, balance in balances.items():
            if asset == self.base_currency:
                prices[asset] = Decimal('1.0')
            else:
                try:
                    price = await self._retry_with_timeout(
                        lambda: exchange.get_ticker_price(f"{asset}/{self.base_currency}"),
                        max_retries=3,
                        timeout=10.0
                    )
                    if price:
                        prices[asset] = price
                except (TimeoutError, Exception) as e:
                    logger.error(f"Price fetch error for {asset}: {str(e)}")
                    raise ValueError(f"Failed to fetch price for {asset}: {str(e)}")
        
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

    @audit_trade_decision(extract_reasoning=True, include_market_data=False)
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

    @audit_risk_check(risk_type='rebalancing_threshold', threshold_param='threshold', value_param='deviation')
    async def needs_rebalancing(
        self,
        exchange: Any,
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
        self._validate_exchange(exchange)
        
        if not self.last_rebalance_time:
            return True
            
        # Check rebalancing frequency
        # Convert frequency string to timedelta
        freq_map = {
            'hourly': pd.Timedelta(hours=1),
            'daily': pd.Timedelta(days=1),
            'weekly': pd.Timedelta(weeks=1)
        }
        frequency = freq_map.get(self.config['rebalancing_frequency'])
        if not frequency:
            raise ValueError(f"Invalid rebalancing frequency: {self.config['rebalancing_frequency']}")
            
        if datetime.now(timezone.utc) - self.last_rebalance_time < frequency.to_pytimedelta():
            return False
            
        # Get current allocation if not provided
        if current_allocation is None:
            current_allocation = await self.get_current_allocation(exchange)
            
        # Get historical data for analysis
        historical_data = await self._fetch_historical_data(exchange, list(current_allocation.keys()))
        
        # Get target allocation
        target = self.strategy.compute_target_allocation(
            current_allocation,
            historical_data,
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

    async def _fetch_historical_data(
        self,
        exchange: Any,
        assets: List[str]
    ) -> pd.DataFrame:
        """
        Fetch historical data for multiple assets in parallel.
        
        Args:
            exchange: Exchange interface instance
            assets: List of asset symbols
            
        Returns:
            DataFrame with historical price data
        """
        lookback_days = self.config['strategy']['lookback_period']
        start_time = int((pd.Timestamp.now() - pd.Timedelta(days=lookback_days)).timestamp() * 1000)

        async def fetch_asset_data(asset: str) -> List[Dict]:
            """Fetch historical data for a single asset."""
            if asset == self.base_currency or self._is_stablecoin(asset):
                return []
                
            symbol = f"{asset}/{self.base_currency}"
            try:
                candles = await exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe="1d",
                    since=start_time,
                    limit=lookback_days
                )
                # Format timestamp as string representation of datetime constructor
                return [{
                    'timestamp': f"datetime({c.timestamp.year}, {c.timestamp.month}, {c.timestamp.day}, {c.timestamp.hour}, {c.timestamp.minute}, tzinfo=timezone.utc)",
                    asset: float(c.close)
                } for c in candles]
            except Exception as e:
                logger.error(f"Failed to fetch data for {asset}: {e}")
                return []

        # Fetch data for all assets in parallel with limited concurrency
        tasks = [fetch_asset_data(asset) for asset in assets]
        results = await self._gather_with_concurrency(10, tasks)
        
        # Combine results
        price_data = []
        for asset_data in results:
            price_data.extend(asset_data)
            
        # Convert to string for cache compatibility
        return create_price_matrix(
            str(price_data),
            str(list(assets))
        )

    @asynccontextmanager
    async def _timeout_context(self, timeout: float = 30.0):
        """Context manager for timeout handling."""
        try:
            yield
        except TimeoutError:
            logger.error(f"Operation timed out after {timeout} seconds")
            raise

    async def _retry_with_timeout(self, coro_func, max_retries: int = 3, timeout: float = 30.0):
        """
        Execute a coroutine-producing function with retry and timeout.
        
        Args:
            coro_func: A function that returns a coroutine when called
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds
            
        Returns:
            Result from the coroutine
        """
        for attempt in range(max_retries):
            try:
                # We need the function to return a fresh coroutine each time
                # since coroutines can only be awaited once
                coro = coro_func()
                return await asyncio.wait_for(coro, timeout=timeout)
            except (TimeoutError, Exception) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Operation failed after {max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def _gather_with_concurrency(self, n: int, tasks: List) -> List[Any]:
        """Execute tasks with limited concurrency and timeout handling."""
        semaphore = asyncio.Semaphore(n)
        
        async def _wrap_task(task):
            async with semaphore:
                # Directly await the task instead of passing it through _retry_with_timeout
                # This is because the task is already a coroutine, not a function returning a coroutine
                return await task
        
        return await asyncio.gather(*[_wrap_task(task) for task in tasks])

    def _validate_exchange(self, exchange: Any) -> None:
        """
        Validate exchange interface has required methods.
        
        Args:
            exchange: Exchange interface to validate
        
        Raises:
            ValueError: If exchange interface is missing required methods
        """
        required_methods = [
            'get_balances',
            'get_ticker_price',
            'get_portfolio_value',
            'fetch_ohlcv',
            'execute_trade',
            'get_average_entry_price'
        ]
        
        missing_methods = [
            method for method in required_methods
            if not hasattr(exchange, method) or not callable(getattr(exchange, method))
        ]
        
        if missing_methods:
            raise ValueError(
                f"Exchange interface missing required methods: {', '.join(missing_methods)}"
            )

    async def get_portfolio_data(self, exchange: Any) -> PortfolioData:
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
            try:
                quantity = (weight * total_value) / current_price if current_price else Decimal('0')
            except DivisionByZero:
                quantity = Decimal('0')
            
            # Calculate profit/loss
            profit_loss = (current_price - avg_entry_price) * quantity if avg_entry_price else Decimal('0')
            
            # Create Position object with attributes matching the dataclass definition
            pos = Position(
                symbol=asset,
                quantity=float(quantity),
                avg_entry_price=float(avg_entry_price),
                unrealized_pnl=float(profit_loss),
                timestamp=datetime.now(timezone.utc).timestamp()
            )
            
            # Store current_price as an attribute even though it's not in the dataclass definition
            # This allows the portfolio.py code to access it
            pos.current_price = float(current_price)
            
            positions.append(pos)

        # Create portfolio data object
        return PortfolioData(
            positions=positions,
            total_value=total_value,
            cash_balance=current_allocation.get(self.base_currency, Decimal('0')) * total_value,
            asset_allocation=current_allocation,
            risk_metrics={
                'volatility_target': self.risk_constraints['volatility_target'],
                'max_drawdown_limit': self.risk_constraints['max_drawdown_limit'],
                'correlation_threshold': self.risk_constraints['correlation_threshold']
            }
        )

    async def analyze_portfolio_with_llm(
        self,
        analyzer: OpenAILLMAnalyzer,
        exchange: Any
    ) -> LLMAnalysisResult:
        """
        Analyze the current portfolio using LLM.

        Args:
            analyzer: Configured OpenAILLMAnalyzer instance
            exchange: Exchange interface instance

        Returns:
            LLMAnalysisResult containing the analysis
            
        Raises:
            ValueError: If exchange interface is invalid or operations fail
        """
        self._validate_exchange(exchange)
        
        # Get portfolio data
        portfolio_data = await self.get_portfolio_data(exchange)
        
        # Get LLM analysis
        return await analyzer.analyze_portfolio(portfolio_data)

    @audit_portfolio_action(action_type='rebalance')
    async def rebalance_portfolio(self, exchange, historical_data: Optional[pd.DataFrame] = None) -> Dict:
        """Execute full portfolio rebalancing operation.

        Args:
            exchange: Exchange interface instance
            historical_data: Optional historical price data

        Returns:
            Dictionary containing rebalancing results
        """
        # Get current allocation
        current_allocation = await self.get_current_allocation(exchange)
        
        # Check if rebalancing is needed
        needs_rebalance = await self.needs_rebalancing(exchange, current_allocation)
        if not needs_rebalance:
            return {
                'status': 'skipped',
                'reason': 'Rebalancing not needed'
            }
            
        # Get historical data if not provided
        historical_data = historical_data or await self._fetch_historical_data(exchange, list(current_allocation.keys()))
        # Compute target allocation
        target_allocation = self.strategy.compute_target_allocation(
            current_allocation,
            historical_data,
            self.risk_constraints
        )
        
        # Analyze tail risk before proceeding
        tail_risk_analysis = None
        hedge_trades = []
        if self.tail_risk_enabled and self.hedge_manager:
            tail_risk_analysis = await self._analyze_tail_risk(exchange, target_allocation)
            
            # If tail risk is elevated, get hedge recommendations
            if tail_risk_analysis.get('tail_risk_score', 0) > self.tail_risk_threshold:
                logger.warning(f"Elevated tail risk detected: {tail_risk_analysis['tail_risk_score']:.3f}")
                hedge_recommendations = await self._get_hedge_recommendations(exchange, target_allocation)
                if hedge_recommendations:
                    hedge_trades = hedge_recommendations.get('trades', [])
                    logger.info(f"Generated {len(hedge_trades)} hedge trades")
        
        # Validate target allocation
        if not self.strategy.validate_constraints(target_allocation):
            violations = self.strategy.get_constraint_violations(target_allocation)
            return {
                'status': 'failed',
                'reason': f'Target allocation violates constraints: {violations}'
            }
            
        # Compute required trades
        total_value = await exchange.get_portfolio_value()
        trades = self.compute_rebalancing_trades(
            current_allocation,
            {k: Decimal(str(v)) for k, v in target_allocation.items()},
            total_value
        )
        
        # Add hedge trades if any
        all_trades = trades + hedge_trades
        
        # Execute trades
        executed_trades = []
        hedge_executed = []
        
        # Execute rebalancing trades first
        for trade in trades:
            try:
                # Validate trade parameters
                if not isinstance(trade['value'], (Decimal, float)) or trade['value'] == 0:
                    raise ValueError(f"Invalid trade value for {trade['asset']}")
                if trade['type'] not in ('buy', 'sell'):
                    raise ValueError(f"Invalid trade type: {trade['type']}")
                
                # Get current price for the asset
                symbol = f"{trade['asset']}/{self.base_currency}"
                current_price = await exchange.get_ticker_price(symbol)
                if not current_price:
                    raise ValueError(f"Could not get price for {symbol}")
                
                # Calculate quantity from value and price
                quantity = abs(trade['value']) / current_price
                
                # Execute trade with retry and timeout
                result = await self._retry_with_timeout(
                    lambda: exchange.execute_trade(
                        symbol=symbol,  # Use symbol instead of asset
                        side=trade['type'],
                        amount=float(quantity),  # Convert to float for exchange API
                        price=float(current_price)  # Include current price
                    ),
                    max_retries=2,  # Fewer retries for trades to avoid duplicates
                    timeout=20.0
                )
                executed_trades.append({
                    **trade,
                    'status': 'success',
                    'execution_details': result
                })
            except Exception as e:
                logger.error(f"Trade execution failed for {trade['asset']}: {str(e)}")
                executed_trades.append({
                    **trade,
                    'status': 'failed',
                    'error': str(e)
                })
                await self._attempt_trade_rollback(exchange, executed_trades)
                
        # Execute hedge trades if any
        if hedge_trades:
            logger.info(f"Executing {len(hedge_trades)} hedge trades")
            for hedge_trade in hedge_trades:
                try:
                    result = await self._execute_hedge_trade(exchange, hedge_trade)
                    hedge_executed.append(result)
                    logger.info(f"Hedge trade executed: {result}")
                except Exception as e:
                    logger.error(f"Hedge trade failed: {str(e)}")
                    hedge_executed.append({
                        'trade': hedge_trade,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        self.last_rebalance_time = datetime.now(timezone.utc)
        
        return {
            'status': 'completed',
            'initial_allocation': current_allocation,
            'target_allocation': target_allocation,
            'trades': executed_trades,
            'hedge_trades': hedge_executed,
            'tail_risk_analysis': tail_risk_analysis
        }
        
    async def _attempt_trade_rollback(
        self,
        exchange: Any,
        executed_trades: List[Dict]
    ) -> None:
        """
        Attempt to rollback successful trades after a failure.
        
        Args:
            exchange: Exchange interface instance
            executed_trades: List of executed trades
        """
        logger.warning("Attempting trade rollback due to execution failure")
        for trade in executed_trades:
            if trade['status'] == 'success':
                try:
                    # Get current price for the asset
                    symbol = f"{trade['asset']}/{self.base_currency}"
                    current_price = await exchange.get_ticker_price(symbol)
                    if not current_price:
                        logger.error(f"Could not get price for {symbol} during rollback")
                        continue
                        
                    # Calculate quantity from value and price
                    quantity = abs(trade['value']) / current_price
                    
                    # Execute reverse trade
                    await exchange.execute_trade(
                        symbol=symbol,  # Use symbol instead of asset
                        side='sell' if trade['type'] == 'buy' else 'buy',
                        amount=float(quantity),  # Convert to float for exchange API
                        price=float(current_price)  # Include current price
                    )
                except Exception as e:
                    logger.error(f"Rollback failed for {trade['asset']}: {str(e)}")
    
    async def _analyze_tail_risk(self, exchange: Any, target_allocation: Dict) -> Dict:
        """Analyze tail risk for the target portfolio allocation."""
        if not self.hedge_manager:
            return {'tail_risk_score': 0}
        
        try:
            # Get current portfolio data
            portfolio_data = await self.get_portfolio_data(exchange)
            
            # Simulate portfolio with target allocation
            simulated_positions = []
            total_value = portfolio_data.total_value
            
            for asset, weight in target_allocation.items():
                if asset == self.base_currency:
                    continue
                    
                symbol = f"{asset}/{self.base_currency}"
                current_price = await exchange.get_ticker_price(symbol)
                if current_price:
                    quantity = (float(weight) * float(total_value)) / float(current_price)
                    simulated_positions.append({
                        'symbol': asset,
                        'quantity': quantity,
                        'current_price': float(current_price),
                        'value': quantity * float(current_price)
                    })
            
            # Calculate tail risk metrics
            tail_risk_score = 0
            for position in simulated_positions:
                # Simple tail risk estimation based on position concentration
                position_weight = position['value'] / float(total_value)
                if position_weight > 0.2:  # More than 20% concentration
                    tail_risk_score += position_weight * 0.5
                    
            # Add correlation-based tail risk (simplified)
            if len(simulated_positions) < 5:  # Low diversification
                tail_risk_score += 0.1
                
            return {
                'tail_risk_score': min(tail_risk_score, 1.0),  # Cap at 1.0
                'position_count': len(simulated_positions),
                'max_position_weight': max((p['value'] / float(total_value) for p in simulated_positions), default=0),
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Tail risk analysis failed: {str(e)}")
            return {'tail_risk_score': 0, 'error': str(e)}
    
    async def _get_hedge_recommendations(self, exchange: Any, target_allocation: Dict) -> Dict:
        """Get hedge recommendations from hedge manager."""
        if not self.hedge_manager:
            return {}
        
        try:
            # Get current portfolio data
            portfolio_data = await self.get_portfolio_data(exchange)
            
            # Create hedge analysis request
            hedge_request = {
                'portfolio_value': float(portfolio_data.total_value),
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'quantity': pos.quantity,
                        'current_price': getattr(pos, 'current_price', 0),
                        'value': pos.quantity * getattr(pos, 'current_price', 0)
                    }
                    for pos in portfolio_data.positions
                ],
                'target_allocation': target_allocation,
                'risk_tolerance': 'moderate'  # Could be configurable
            }
            
            # Get hedge recommendations (simplified approach)
            hedge_trades = []
            
            # Calculate total portfolio risk
            total_value = float(portfolio_data.total_value)
            high_risk_threshold = 0.15  # 15% position limit
            
            for asset, weight in target_allocation.items():
                if asset == self.base_currency:
                    continue
                    
                if float(weight) > high_risk_threshold:
                    # Suggest hedging for oversized positions
                    hedge_amount = (float(weight) - high_risk_threshold) * total_value
                    
                    # Simple hedge trade (could be more sophisticated)
                    hedge_trades.append({
                        'asset': asset,
                        'type': 'hedge_short',
                        'value': hedge_amount * 0.5,  # Hedge 50% of excess exposure
                        'reason': f'Hedge oversized position ({weight:.1%} > {high_risk_threshold:.1%})'
                    })
            
            return {
                'trades': hedge_trades,
                'total_hedge_value': sum(t['value'] for t in hedge_trades),
                'recommendation_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Hedge recommendation failed: {str(e)}")
            return {'error': str(e)}
    
    async def _execute_hedge_trade(self, exchange: Any, hedge_trade: Dict) -> Dict:
        """Execute a hedge trade."""
        try:
            symbol = f"{hedge_trade['asset']}/{self.base_currency}"
            current_price = await exchange.get_ticker_price(symbol)
            
            if not current_price:
                raise ValueError(f"Could not get price for {symbol}")
            
            # Calculate quantity
            quantity = hedge_trade['value'] / float(current_price)
            
            # Execute the hedge trade
            result = await exchange.execute_trade(
                asset=hedge_trade['asset'],
                amount=quantity,
                side='sell',  # Hedge trades are typically short positions
                order_type='market'
            )
            
            return {
                'trade': hedge_trade,
                'status': 'success',
                'executed_quantity': quantity,
                'executed_price': float(current_price),
                'trade_result': result
            }
            
        except Exception as e:
            logger.error(f"Hedge trade execution failed: {str(e)}")
            return {
                'trade': hedge_trade,
                'status': 'failed',
                'error': str(e)
            }