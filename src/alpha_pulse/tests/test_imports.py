"""
Test script to verify imports work correctly.
"""
try:
    # Test absolute imports
    from alpha_pulse.exchanges.interfaces import BaseExchange, ExchangeConfiguration
    from alpha_pulse.exchanges.adapters.ccxt_adapter import CCXTAdapter
    from alpha_pulse.exchanges.factories import ExchangeFactory, ExchangeType
    from alpha_pulse.exchanges.base import OHLCV, Balance
    print("Absolute imports successful!")
except Exception as e:
    print(f"Absolute imports failed: {e}")

# We don't need to test the data_pipeline import since it requires loguru