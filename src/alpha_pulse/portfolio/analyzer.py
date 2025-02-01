"""
Portfolio analyzer for analyzing and optimizing portfolio allocations.
"""
from decimal import Decimal
from typing import Dict, List, Optional, Type
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from alpha_pulse.exchanges import BaseExchange, Balance
from alpha_pulse.portfolio.allocation_strategy import AllocationStrategy, AllocationResult
from alpha_pulse.portfolio.mpt_strategy import MPTStrategy
from alpha_pulse.portfolio.hrp_strategy import HRPStrategy

# Default parameters for simulated returns
DEFAULT_RETURN_MEAN = 0.0002  # 0.02% daily return
DEFAULT_RETURN_VOL = 0.02    # 2% daily volatility


class PortfolioAnalyzer:
    """Analyzes and optimizes portfolio allocations."""
    
    def __init__(
        self,
        exchange: BaseExchange,
        strategy: Optional[AllocationStrategy] = None,
        lookback_days: int = 30
    ):
        """Initialize portfolio analyzer.
        
        Args:
            exchange: Exchange instance
            strategy: Allocation strategy (defaults to MPT)
            lookback_days: Days of historical data to use
        """
        self.exchange = exchange
        self.strategy = strategy or MPTStrategy()
        self.lookback_days = lookback_days
    
    async def get_current_allocation(self) -> Dict[str, Decimal]:
        """Get current portfolio allocation.
        
        Returns:
            Dict mapping assets to their current weights
        """
        try:
            # Get balances from exchange
            balances = await self.exchange.get_balances()
            
            # Calculate total portfolio value in base currency
            total_value = sum(balance.in_base_currency for balance in balances.values())
            
            if total_value == 0:
                return {}
            
            # Calculate weights
            weights = {
                asset: balance.in_base_currency / total_value
                for asset, balance in balances.items()
                if balance.total > 0
            }
            
            return weights
            
        except Exception as e:
            logger.error(f"Error getting current allocation: {e}")
            raise
    
    async def get_historical_returns(
        self,
        assets: List[str],
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        """Get historical returns for assets.
        
        In this implementation, we simulate returns data for demonstration.
        In a production environment, this would fetch real historical data
        from the exchange.
        
        Args:
            assets: List of asset symbols
            timeframe: Return timeframe
            
        Returns:
            DataFrame of historical returns
        
        Raises:
            ValueError: If no valid assets provided
        """
        if not assets:
            raise ValueError("No assets provided")
        
        # Generate date range
        end_date = datetime.now()
        dates = pd.date_range(
            end=end_date,
            periods=self.lookback_days,
            freq='D'
        )
        
        returns_data = {}
        valid_assets = 0
        
        for asset in assets:
            try:
                if asset == "USDT":
                    # Use risk-free rate for stable coins
                    returns = pd.Series(
                        [self.strategy.risk_free_rate / 365] * self.lookback_days,
                        index=dates
                    )
                    valid_assets += 1
                else:
                    # Verify asset exists
                    symbol = f"{asset}/USDT"
                    price = await self.exchange.get_ticker_price(symbol)
                    
                    if price is None:
                        logger.warning(f"Could not get price for {symbol}")
                        continue
                    
                    # Get historical OHLCV data
                    ohlcv_data = await self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=self.lookback_days
                    )
                    
                    if not ohlcv_data:
                        logger.warning(f"No historical data for {symbol}")
                        continue
                    
                    # Calculate returns from close prices
                    prices = pd.Series(
                        [float(candle.close) for candle in ohlcv_data],
                        index=dates[-len(ohlcv_data):]
                    )
                    returns = prices.pct_change().fillna(0)
                    valid_assets += 1
                
                returns_data[asset] = returns
                
            except Exception as e:
                logger.error(f"Error getting returns for {asset}: {e}")
                continue
        
        if valid_assets == 0:
            raise ValueError("Could not get returns for any assets")
        
        return pd.DataFrame(returns_data)
    
    async def analyze_portfolio(
        self,
        strategy_class: Optional[Type[AllocationStrategy]] = None,
        constraints: Optional[Dict] = None
    ) -> AllocationResult:
        """Analyze current portfolio and suggest optimal allocation.
        
        Args:
            strategy_class: Optional strategy class to use
            constraints: Optional allocation constraints
            
        Returns:
            AllocationResult with target weights and metrics
        """
        try:
            # Get current allocation
            current_weights = await self.get_current_allocation()
            if not current_weights:
                raise ValueError("No assets in portfolio")
            
            # Get historical returns
            returns = await self.get_historical_returns(list(current_weights.keys()))
            if returns.empty:
                raise ValueError("Could not get historical returns")
            
            # Use specified strategy or default
            strategy = (strategy_class or self.strategy.__class__)()
            
            # Calculate optimal allocation
            result = strategy.calculate_allocation(
                returns=returns,
                current_weights=current_weights,
                constraints=constraints
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {e}")
            raise
    
    def get_rebalancing_trades(
        self,
        current_weights: Dict[str, Decimal],
        target_weights: Dict[str, Decimal],
        total_value: Decimal,
        min_trade_value: Decimal = Decimal("10.0")
    ) -> List[Dict]:
        """Calculate trades needed to rebalance portfolio.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            total_value: Total portfolio value
            min_trade_value: Minimum trade value to execute
            
        Returns:
            List of trades to execute
        """
        trades = []
        
        # Calculate trade amounts
        for asset in set(current_weights.keys()) | set(target_weights.keys()):
            current = current_weights.get(asset, Decimal("0"))
            target = target_weights.get(asset, Decimal("0"))
            
            if current == target:
                continue
            
            # Calculate trade size
            trade_value = (target - current) * total_value
            
            # Skip small trades
            if abs(trade_value) < min_trade_value:
                continue
            
            trades.append({
                'asset': asset,
                'side': 'buy' if trade_value > 0 else 'sell',
                'value': abs(trade_value)
            })
        
        return sorted(trades, key=lambda x: abs(x['value']), reverse=True)