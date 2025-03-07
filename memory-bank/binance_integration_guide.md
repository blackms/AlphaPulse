# Binance Integration Testing Guide

This document provides detailed guidance for integrating and testing the AI Hedge Fund system with Binance's API, focusing on testnet usage for validation before moving to production.

## 1. Binance Testnet Setup

### Obtaining Testnet API Keys

1. Go to Binance Testnet at: https://testnet.binance.vision/
2. Log in using your Binance account or create a new account
3. Navigate to API Management
4. Create a new API key pair (API Key and Secret Key)
5. Store these credentials securely using environment variables or a secure vault

### Environment Configuration

```bash
# Store in a secure .env file (do not commit to version control)
BINANCE_TESTNET_API_KEY=your_testnet_api_key
BINANCE_TESTNET_SECRET_KEY=your_testnet_secret_key
BINANCE_TESTNET_BASE_URL=https://testnet.binance.vision/api
```

## 2. Connection Implementation

### Base Connector Class

```python
import hmac
import hashlib
import time
import aiohttp
import asyncio
import logging
from typing import Dict, Any, Optional
from urllib.parse import urlencode

class BinanceConnector:
    """Base connector for Binance API."""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str, testnet: bool = True):
        """Initialize connector with credentials and settings."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.testnet = testnet
        self.session = None
        self.logger = logging.getLogger("binance_connector")
        
    async def __aenter__(self):
        """Create session on context enter."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close session on context exit."""
        if self.session:
            await self.session.close()
            
    async def _get(self, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Make a GET request to Binance API."""
        return await self._request('GET', endpoint, params, signed)
            
    async def _post(self, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Make a POST request to Binance API."""
        return await self._request('POST', endpoint, params, signed)
        
    async def _request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Make a request to Binance API with appropriate headers and signature."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        
        if params is None:
            params = {}
            
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = urlencode(params)
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
        
        try:
            if method == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    data = await response.json()
                    if response.status >= 400:
                        self.logger.error(f"Binance API error: {response.status} - {data}")
                        raise BinanceAPIError(data)
                    return data
            elif method == 'POST':
                async with self.session.post(url, params=params, headers=headers) as response:
                    data = await response.json()
                    if response.status >= 400:
                        self.logger.error(f"Binance API error: {response.status} - {data}")
                        raise BinanceAPIError(data)
                    return data
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error during Binance API request: {e}")
            raise BinanceNetworkError(f"Network error: {e}")
            
class BinanceAPIError(Exception):
    """Exception raised for Binance API errors."""
    pass
    
class BinanceNetworkError(Exception):
    """Exception raised for network errors when connecting to Binance."""
    pass
```

### Specific Endpoints Implementation

```python
class BinanceMarketData(BinanceConnector):
    """Binance market data API implementation."""
    
    async def get_exchange_info(self) -> Dict:
        """Get exchange information."""
        return await self._get('/v3/exchangeInfo')
        
    async def get_ticker_price(self, symbol: Optional[str] = None) -> Dict:
        """Get ticker price for a symbol or all symbols."""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._get('/v3/ticker/price', params)
        
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book for a symbol."""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        return await self._get('/v3/depth', params)
        
    async def get_klines(self, 
                        symbol: str, 
                        interval: str, 
                        start_time: Optional[int] = None,
                        end_time: Optional[int] = None,
                        limit: Optional[int] = 500) -> Dict:
        """Get klines/candlestick data."""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        return await self._get('/v3/klines', params)
        
class BinanceTrading(BinanceConnector):
    """Binance trading API implementation."""
    
    async def get_account(self) -> Dict:
        """Get account information."""
        return await self._get('/v3/account', signed=True)
        
    async def create_order(self, 
                         symbol: str,
                         side: str,
                         order_type: str,
                         quantity: Optional[float] = None,
                         quote_order_qty: Optional[float] = None,
                         price: Optional[float] = None,
                         time_in_force: Optional[str] = None,
                         **kwargs) -> Dict:
        """Create a new order."""
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type
        }
        
        # Add optional parameters
        if quantity:
            params['quantity'] = quantity
        if quote_order_qty:
            params['quoteOrderQty'] = quote_order_qty
        if price:
            params['price'] = price
        if time_in_force:
            params['timeInForce'] = time_in_force
            
        # Add any additional parameters
        params.update(kwargs)
        
        return await self._post('/v3/order', params, signed=True)
        
    async def create_test_order(self,
                              symbol: str,
                              side: str,
                              order_type: str,
                              quantity: Optional[float] = None,
                              quote_order_qty: Optional[float] = None,
                              price: Optional[float] = None,
                              **kwargs) -> Dict:
        """Create a test order (doesn't actually place an order)."""
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type
        }
        
        # Add optional parameters
        if quantity:
            params['quantity'] = quantity
        if quote_order_qty:
            params['quoteOrderQty'] = quote_order_qty
        if price:
            params['price'] = price
            
        # Add any additional parameters
        params.update(kwargs)
        
        return await self._post('/v3/order/test', params, signed=True)
        
    async def get_order(self, symbol: str, order_id: Optional[int] = None, 
                      orig_client_order_id: Optional[str] = None) -> Dict:
        """Get order status."""
        params = {
            'symbol': symbol
        }
        
        if order_id:
            params['orderId'] = order_id
        elif orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        else:
            raise ValueError("Either order_id or orig_client_order_id must be provided")
            
        return await self._get('/v3/order', params, signed=True)
        
    async def cancel_order(self, symbol: str, order_id: Optional[int] = None,
                         orig_client_order_id: Optional[str] = None) -> Dict:
        """Cancel an order."""
        params = {
            'symbol': symbol
        }
        
        if order_id:
            params['orderId'] = order_id
        elif orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        else:
            raise ValueError("Either order_id or orig_client_order_id must be provided")
            
        return await self._post('/v3/order', params, signed=True)
```

## 3. Comprehensive Testing Methodology

### Test Suite Structure

Create a comprehensive test suite that covers all API interactions:

```python
import pytest
import os
import json
import asyncio
from datetime import datetime, timedelta

# Load test configuration
@pytest.fixture
def test_config():
    """Load test configuration from environment variables or test file."""
    return {
        "api_key": os.environ.get("BINANCE_TESTNET_API_KEY"),
        "api_secret": os.environ.get("BINANCE_TESTNET_SECRET_KEY"),
        "base_url": os.environ.get("BINANCE_TESTNET_BASE_URL"),
        "test_symbol": "BTCUSDT",
        "test_quantity": 0.001,  # Small quantity for test orders
        "test_price": 50000.0    # Example price for limit orders
    }

@pytest.fixture
async def market_data_client(test_config):
    """Create a market data client for testing."""
    async with BinanceMarketData(
        test_config["api_key"],
        test_config["api_secret"],
        test_config["base_url"],
        testnet=True
    ) as client:
        yield client

@pytest.fixture
async def trading_client(test_config):
    """Create a trading client for testing."""
    async with BinanceTrading(
        test_config["api_key"],
        test_config["api_secret"],
        test_config["base_url"],
        testnet=True
    ) as client:
        yield client

# Market Data Tests
class TestBinanceMarketData:
    """Test suite for Binance market data API."""
    
    @pytest.mark.asyncio
    async def test_exchange_info(self, market_data_client):
        """Test getting exchange information."""
        result = await market_data_client.get_exchange_info()
        assert "timezone" in result
        assert "serverTime" in result
        assert "symbols" in result
        assert isinstance(result["symbols"], list)
        
    @pytest.mark.asyncio
    async def test_ticker_price(self, market_data_client, test_config):
        """Test getting ticker price."""
        symbol = test_config["test_symbol"]
        
        # Test for specific symbol
        result = await market_data_client.get_ticker_price(symbol)
        assert "symbol" in result
        assert result["symbol"] == symbol
        assert "price" in result
        assert float(result["price"]) > 0
        
        # Test for all symbols
        result = await market_data_client.get_ticker_price()
        assert isinstance(result, list)
        assert len(result) > 0
        assert "symbol" in result[0]
        assert "price" in result[0]
        
    @pytest.mark.asyncio
    async def test_order_book(self, market_data_client, test_config):
        """Test getting order book."""
        symbol = test_config["test_symbol"]
        result = await market_data_client.get_order_book(symbol)
        
        assert "lastUpdateId" in result
        assert "bids" in result
        assert "asks" in result
        assert isinstance(result["bids"], list)
        assert isinstance(result["asks"], list)
        
    @pytest.mark.asyncio
    async def test_klines(self, market_data_client, test_config):
        """Test getting klines/candlestick data."""
        symbol = test_config["test_symbol"]
        interval = "1h"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        
        result = await market_data_client.get_klines(
            symbol, interval, start_time, end_time, 10
        )
        
        assert isinstance(result, list)
        assert len(result) <= 10
        
        # Check kline structure
        kline = result[0]
        assert len(kline) >= 6
        assert isinstance(kline[0], int)  # Open time
        assert isinstance(float(kline[1]), float)  # Open price
        assert isinstance(float(kline[2]), float)  # High price
        assert isinstance(float(kline[3]), float)  # Low price
        assert isinstance(float(kline[4]), float)  # Close price
        assert isinstance(float(kline[5]), float)  # Volume

# Trading Tests
class TestBinanceTrading:
    """Test suite for Binance trading API."""
    
    @pytest.mark.asyncio
    async def test_account_info(self, trading_client):
        """Test getting account information."""
        result = await trading_client.get_account()
        
        assert "makerCommission" in result
        assert "takerCommission" in result
        assert "balances" in result
        assert isinstance(result["balances"], list)
        
    @pytest.mark.asyncio
    async def test_create_test_order(self, trading_client, test_config):
        """Test creating a test order."""
        symbol = test_config["test_symbol"]
        quantity = test_config["test_quantity"]
        
        # Test market order
        market_result = await trading_client.create_test_order(
            symbol=symbol,
            side="BUY",
            order_type="MARKET",
            quantity=quantity
        )
        
        # Successful test orders typically return an empty dict
        assert isinstance(market_result, dict)
        
        # Test limit order
        price = test_config["test_price"]
        limit_result = await trading_client.create_test_order(
            symbol=symbol,
            side="BUY",
            order_type="LIMIT",
            quantity=quantity,
            price=price,
            timeInForce="GTC"
        )
        
        assert isinstance(limit_result, dict)
        
    @pytest.mark.asyncio
    async def test_create_and_cancel_order(self, trading_client, test_config):
        """Test creating and canceling a real order."""
        symbol = test_config["test_symbol"]
        quantity = test_config["test_quantity"]
        price = test_config["test_price"] * 0.5  # Set price below market for limit buy
        
        # Create a limit order that won't fill immediately
        order_result = await trading_client.create_order(
            symbol=symbol,
            side="BUY",
            order_type="LIMIT",
            quantity=quantity,
            price=price,
            timeInForce="GTC"
        )
        
        assert "orderId" in order_result
        assert order_result["symbol"] == symbol
        assert order_result["status"] in ["NEW", "PARTIALLY_FILLED"]
        
        order_id = order_result["orderId"]
        
        # Get order status
        get_result = await trading_client.get_order(symbol, order_id=order_id)
        assert get_result["orderId"] == order_id
        
        # Cancel order
        cancel_result = await trading_client.cancel_order(symbol, order_id=order_id)
        assert cancel_result["orderId"] == order_id
        assert cancel_result["status"] == "CANCELED"
```

### Automated Test Execution

Create an automated test execution script:

```python
#!/usr/bin/env python
"""
Binance API Integration Tester
Run all Binance API tests to validate integration.
"""
import asyncio
import pytest
import logging
import argparse
import sys
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"binance_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("binance_tester")

def check_environment():
    """Check if required environment variables are set."""
    required_vars = [
        "BINANCE_TESTNET_API_KEY",
        "BINANCE_TESTNET_SECRET_KEY",
        "BINANCE_TESTNET_BASE_URL"
    ]
    
    missing = [var for var in required_vars if not os.environ.get(var)]
    
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Please set these variables before running the tests.")
        return False
        
    return True

def run_tests():
    """Run all Binance API tests."""
    if not check_environment():
        return False
        
    logger.info("Starting Binance API integration tests")
    
    # Run pytest with parameters
    args = [
        "-xvs",  # Exit on first failure, verbose, don't capture output
        "tests/binance/",  # Path to test directory
        "--log-cli-level=INFO"  # Log level
    ]
    
    result = pytest.main(args)
    
    if result == 0:
        logger.info("All Binance API tests passed successfully!")
        return True
    else:
        logger.error(f"Binance API tests failed with exit code {result}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Binance API integration tests")
    parser.add_argument("--market-only", action="store_true", help="Run only market data tests")
    parser.add_argument("--trading-only", action="store_true", help="Run only trading tests")
    
    args = parser.parse_args()
    
    success = run_tests()
    sys.exit(0 if success else 1)
```

## 4. Common Error Handling

### Rate Limit Handling

Binance implements strict rate limits. Implement a rate limiter:

```python
class RateLimiter:
    """Rate limiter for Binance API requests."""
    
    def __init__(self, max_requests=1200, per_minute=True):
        """Initialize rate limiter with limits."""
        self.max_requests = max_requests
        self.interval = 60 if per_minute else 1  # seconds
        self.request_timestamps = []
        self.lock = asyncio.Lock()
        
    async def __aenter__(self):
        """Context manager entry."""
        await self.acquire()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
        
    async def acquire(self):
        """Acquire permission to make a request."""
        async with self.lock:
            now = time.time()
            
            # Remove timestamps older than our interval
            self.request_timestamps = [ts for ts in self.request_timestamps 
                                      if now - ts <= self.interval]
            
            # If we've hit the limit, wait until the oldest timestamp expires
            if len(self.request_timestamps) >= self.max_requests:
                oldest = min(self.request_timestamps)
                wait_time = self.interval - (now - oldest)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    
            # Add current timestamp and proceed
            self.request_timestamps.append(time.time())
```

### Handling Specific Binance Errors

Create specific error handling for common Binance errors:

```python
async def handle_binance_error(error, retry_count=0, max_retries=3):
    """Handle specific Binance API errors with appropriate strategies."""
    if not hasattr(error, 'code'):
        # Network or unknown error
        if retry_count < max_retries:
            wait_time = 2 ** retry_count  # Exponential backoff
            logging.warning(f"Network error, retrying in {wait_time}s: {error}")
            await asyncio.sleep(wait_time)
            return {'retry': True, 'retry_count': retry_count + 1}
        else:
            logging.error(f"Max retries exceeded for error: {error}")
            return {'retry': False, 'error': error}
    
    # Handle known error codes
    if error.code == -1003:  # Too many requests (IP rate limit)
        logging.warning("IP rate limit exceeded, backing off")
        await asyncio.sleep(60)  # Wait a full minute
        return {'retry': True, 'retry_count': retry_count}
        
    elif error.code == -1021:  # Timestamp for this request was outside the recvWindow
        logging.warning("Timestamp outside recvWindow, retrying with updated timestamp")
        return {'retry': True, 'retry_count': retry_count}
        
    elif error.code == -2010:  # Account has insufficient balance
        logging.error("Insufficient balance for order")
        return {'retry': False, 'error': error}
        
    elif error.code == -2011:  # Order would trigger immediately as maker
        # Adjust the order price and retry
        logging.warning("Order would trigger as taker, adjusting price")
        return {'retry': True, 'retry_count': retry_count, 'adjust_price': True}
        
    # Default error handling
    if retry_count < max_retries:
        wait_time = 2 ** retry_count
        logging.warning(f"Binance error {error.code}, retrying in {wait_time}s: {error}")
        await asyncio.sleep(wait_time)
        return {'retry': True, 'retry_count': retry_count + 1}
    else:
        logging.error(f"Max retries exceeded for error code {error.code}: {error}")
        return {'retry': False, 'error': error}
```

## 5. Binance Integration Validation Strategy

### Step 1: Validate Market Data Access

1. Test all market data endpoints:
   - Exchange information
   - Ticker prices
   - Order book data
   - Candlestick/kline data

2. Verify data quality:
   - Compare data from multiple sources
   - Ensure consistency with web interface
   - Check update frequency

### Step 2: Validate Trading Functions (Paper Only)

1. Test order functionality with test orders:
   - Market orders
   - Limit orders
   - Stop-loss orders
   - Take-profit orders

2. Validate order lifecycle:
   - Order creation
   - Order status updates
   - Order cancellation

### Step 3: Test Small Real Orders

1. Place small value orders:
   - Start with minimum order sizes
   - Test both buy and sell
   - Test different order types

2. Validate full execution flow:
   - Order placement
   - Order filling
   - Balance updates
   - Order history

### Step 4: Integration with Trading System

1. Connect the Binance adapter to the trading system:
   - Integrate with Exchange interface
   - Implement required methods
   - Ensure proper error handling

2. Test full trading workflow:
   - Signal generation
   - Risk management
   - Portfolio allocation
   - Order execution
   - Post-trade analysis

3. Validate with simulated trading:
   - Use paper trading mode
   - Execute multiple trades
   - Verify all components interact correctly

## 6. Post-Integration Monitoring

After integration, implement specific monitoring for Binance connectivity:

1. API health checks:
   - Regular connectivity tests
   - Response time monitoring
   - Error rate tracking

2. Order execution monitoring:
   - Order fill rate
   - Execution latency
   - Order rejection rate
   - Slippage analysis

3. Rate limit monitoring:
   - Track API usage against limits
   - Implement throttling if needed
   - Create alerts for approaching limits

## 7. Troubleshooting Guide

### Common Issues and Solutions

1. **Authentication Errors**:
   - Double-check API key and secret
   - Verify IP restrictions on API key
   - Check key permissions

2. **Order Placement Failures**:
   - Verify sufficient balance
   - Check minimum order size requirements
   - Check price precision and tick size
   - Validate order parameters against exchange rules

3. **Rate Limit Issues**:
   - Implement proper rate limiting
   - Use WebSocket for real-time data when possible
   - Batch requests where appropriate
   - Add jitter to request timing

4. **Data Consistency Issues**:
   - Implement idempotent operations
   - Use unique client order IDs
   - Implement reconciliation processes
   - Maintain robust logging for debugging