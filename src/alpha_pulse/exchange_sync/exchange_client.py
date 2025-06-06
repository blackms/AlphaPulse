"""
Exchange client for interacting with cryptocurrency exchange APIs.

This module abstracts communication with various exchanges, handling authentication,
rate limiting, and data conversion to internal models.
"""
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import ccxt.async_support as ccxt

from loguru import logger

# Import the credentials manager if available
try:
    from alpha_pulse.exchanges.credentials.manager import credentials_manager
    CREDENTIALS_MANAGER_AVAILABLE = True
except ImportError:
    CREDENTIALS_MANAGER_AVAILABLE = False

from .models import PortfolioItem, OrderData, SyncResult
from .config import get_exchange_config


class ExchangeError(Exception):
    """Exception raised for exchange-related errors."""
    pass


class ExchangeClient:
    """
    Client for interacting with cryptocurrency exchanges.
    
    This class abstracts the communication with exchange APIs,
    providing a consistent interface regardless of the underlying exchange.
    """
    
    def __init__(self, exchange_id: str):
        """
        Initialize the exchange client.
        
        Args:
            exchange_id: Identifier of the exchange (e.g., 'bybit', 'binance')
        """
        self.exchange_id = exchange_id.lower()
        self.exchange_config = get_exchange_config(exchange_id)
        self.exchange = None
        
        # Log credential source
        if CREDENTIALS_MANAGER_AVAILABLE:
            creds = credentials_manager.get_credentials(exchange_id)
            if creds:
                logger.info(f"Using credentials from AlphaPulse credentials manager for {exchange_id}")
            else:
                logger.info(f"No credentials found in credentials manager for {exchange_id}, using environment variables")
        else:
            logger.info(f"AlphaPulse credentials manager not available, using environment variables")
    
    async def connect(self) -> bool:
        """
        Connect to the exchange API.
        
        Returns:
            True if successfully connected
            
        Raises:
            ExchangeError: If connection fails
        """
        try:
            # Check if exchange is supported by ccxt
            if not hasattr(ccxt, self.exchange_id):
                raise ExchangeError(f"Unsupported exchange: {self.exchange_id}")
            
            # Get the exchange class from ccxt
            exchange_class = getattr(ccxt, self.exchange_id)
            
            # Configure the exchange
            config = {
                'apiKey': self.exchange_config['api_key'],
                'secret': self.exchange_config['api_secret'], 
                'enableRateLimit': True
            }
            
            # Log credential status (without revealing actual keys)
            if config['apiKey']:
                logger.info(f"Connecting to {self.exchange_id} with API key: {config['apiKey'][:4]}...{config['apiKey'][-4:] if len(config['apiKey']) > 8 else ''}")
            else:
                logger.warning(f"Connecting to {self.exchange_id} without API credentials (read-only mode)")
            
            # Add testnet if configured
            if self.exchange_config['testnet']:
                if self.exchange_id in ['bybit', 'binance', 'kucoin']:
                    config['options'] = {'testnet': True}
                logger.info(f"Using testnet mode for {self.exchange_id}")
            
            # Create exchange instance
            self.exchange = exchange_class(config)
            
            # Load markets to ensure connection is working
            await self.exchange.load_markets()
            
            logger.info(f"Successfully connected to {self.exchange_id}")
            return True
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error connecting to {self.exchange_id}: {str(e)}")
            raise ExchangeError(f"Network error: {str(e)}")
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error connecting to {self.exchange_id}: {str(e)}")
            raise ExchangeError(f"Exchange error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_id}: {str(e)}")
            raise ExchangeError(f"Connection error: {str(e)}")
    
    async def disconnect(self) -> None:
        """Safely close the exchange connection."""
        if self.exchange:
            try:
                await self.exchange.close()
                logger.info(f"Disconnected from {self.exchange_id}")
            except Exception as e:
                logger.warning(f"Error disconnecting from {self.exchange_id}: {str(e)}")
    
    async def get_portfolio(self) -> List[PortfolioItem]:
        """
        Get current portfolio data from the exchange.
        
        Returns:
            List of portfolio items
            
        Raises:
            ExchangeError: If fetching portfolio fails
        """
        if not self.exchange:
            await self.connect()
        
        try:
            # Fetch balance from exchange
            balance = await self.exchange.fetch_balance()
            
            if not balance or 'total' not in balance:
                logger.warning(f"No balance data returned from {self.exchange_id}")
                return []
            
            result = []
            
            # Process each asset with a non-zero balance
            for asset, total in balance['total'].items():
                if total <= 0:
                    continue
                
                # Create a basic portfolio item
                portfolio_item = PortfolioItem(
                    asset=asset,
                    quantity=float(total)
                )
                
                # Try to get current price
                try:
                    ticker = await self.get_ticker(f"{asset}/USDT")
                    if ticker and 'last' in ticker and ticker['last'] is not None:
                        portfolio_item.current_price = float(ticker['last'])
                except Exception as e:
                    logger.warning(f"Could not fetch price for {asset}: {str(e)}")
                
                # Try to calculate average entry price
                try:
                    avg_price = await self.calculate_average_entry_price(asset)
                    if avg_price is not None:
                        portfolio_item.avg_entry_price = avg_price
                except Exception as e:
                    logger.warning(f"Could not calculate average entry price for {asset}: {str(e)}")
                
                result.append(portfolio_item)
            
            logger.info(f"Retrieved {len(result)} portfolio items from {self.exchange_id}")
            return result
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching portfolio from {self.exchange_id}: {str(e)}")
            raise ExchangeError(f"Network error: {str(e)}")
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching portfolio from {self.exchange_id}: {str(e)}")
            raise ExchangeError(f"Exchange error: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching portfolio from {self.exchange_id}: {str(e)}")
            raise ExchangeError(f"Fetch error: {str(e)}")
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Ticker data dictionary
            
        Raises:
            ExchangeError: If fetching ticker fails
        """
        if not self.exchange:
            await self.connect()
        
        try:
            # Ensure symbol is properly formatted
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            # Normalize the symbol format according to the exchange
            try:
                symbol = self.exchange.market(symbol)['symbol']
            except Exception:
                # If the market isn't found, use the original symbol
                pass
            
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
            
        except ccxt.BadSymbol as e:
            logger.warning(f"Symbol {symbol} not found on {self.exchange_id}: {str(e)}")
            raise ExchangeError(f"Symbol not found: {symbol}")
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching ticker for {symbol}: {str(e)}")
            raise ExchangeError(f"Network error: {str(e)}")
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching ticker for {symbol}: {str(e)}")
            raise ExchangeError(f"Exchange error: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            raise ExchangeError(f"Fetch error: {str(e)}")
    
    async def get_orders(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent orders for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Maximum number of orders to retrieve
            
        Returns:
            List of orders
            
        Raises:
            ExchangeError: If fetching orders fails
        """
        if not self.exchange:
            await self.connect()
        
        try:
            # Ensure symbol is properly formatted
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            # Normalize the symbol format according to the exchange
            try:
                symbol = self.exchange.market(symbol)['symbol']
            except Exception:
                # If the market isn't found, use the original symbol
                pass
            
            # Some exchanges require closed=True to get completed orders
            try:
                orders = await self.exchange.fetch_orders(symbol, limit=limit)
            except ccxt.NotSupported:
                # If fetch_orders is not supported, try fetch_closed_orders
                try:
                    orders = await self.exchange.fetch_closed_orders(symbol, limit=limit)
                except ccxt.NotSupported:
                    logger.warning(f"Fetching orders not supported by {self.exchange_id}")
                    return []
            
            return orders
            
        except ccxt.BadSymbol as e:
            logger.warning(f"Symbol {symbol} not found on {self.exchange_id}: {str(e)}")
            return []
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching orders for {symbol}: {str(e)}")
            raise ExchangeError(f"Network error: {str(e)}")
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching orders for {symbol}: {str(e)}")
            raise ExchangeError(f"Exchange error: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching orders for {symbol}: {str(e)}")
            raise ExchangeError(f"Fetch error: {str(e)}")
    
    async def calculate_average_entry_price(self, asset: str) -> Optional[float]:
        """
        Calculate the average entry price for an asset based on order history.
        
        Args:
            asset: Asset symbol (e.g., 'BTC')
            
        Returns:
            Average entry price or None if it can't be calculated
        """
        try:
            symbol = f"{asset}/USDT"
            
            # Get recent orders for the symbol
            orders = await self.get_orders(symbol)
            
            if not orders:
                return None
            
            # Filter buy orders that have been filled
            buy_orders = [o for o in orders if 
                          o['side'].lower() == 'buy' and 
                          o['status'].lower() in ['closed', 'filled']]
            
            if not buy_orders:
                return None
            
            # Calculate total quantity and cost
            total_quantity = 0.0
            total_cost = 0.0
            
            for order in buy_orders:
                # Use filled amount rather than order amount where available
                quantity = order.get('filled', order.get('amount', 0))
                
                # Skip orders with zero quantity
                if quantity <= 0:
                    continue
                
                price = order.get('price', 0)
                
                # If price is 0 (market orders), try to use average price
                if price == 0:
                    price = order.get('average', 0)
                
                # If we still don't have a price, skip this order
                if price <= 0:
                    continue
                
                total_quantity += quantity
                total_cost += quantity * price
            
            if total_quantity > 0:
                return total_cost / total_quantity
            
            return None
            
        except Exception as e:
            logger.warning(f"Error calculating average entry price for {asset}: {str(e)}")
            return None
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.disconnect()