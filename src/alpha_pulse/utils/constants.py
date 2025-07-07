"""
Constants for AlphaPulse trading system.
"""

# Supported exchanges
SUPPORTED_EXCHANGES = ["binance", "bybit", "kraken", "bitfinex", "kucoin"]

# Trading constants
DEFAULT_LEVERAGE = 1.0
MAX_LEVERAGE = 10.0
MIN_POSITION_SIZE = 10.0  # USD
MAX_POSITION_SIZE = 100000.0  # USD

# Risk management constants
DEFAULT_STOP_LOSS_PCT = 0.05  # 5%
DEFAULT_TAKE_PROFIT_PCT = 0.10  # 10%
MAX_PORTFOLIO_RISK = 0.02  # 2%
MAX_DRAWDOWN = 0.20  # 20%

# Time constants
MARKET_HOURS = {
    "US": {"open": "09:30", "close": "16:00"},
    "CRYPTO": {"open": "00:00", "close": "23:59"}  # 24/7
}

# API rate limits
RATE_LIMIT_PER_MINUTE = 120
RATE_LIMIT_PER_HOUR = 3600

# Cache TTLs (seconds)
PRICE_CACHE_TTL = 10
BALANCE_CACHE_TTL = 60
ORDER_CACHE_TTL = 5