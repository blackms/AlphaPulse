"""
Simple script to verify imports work correctly.
"""
import sys
import os

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
print(f"Added {src_path} to Python path")
print(f"Python path: {sys.path}")

print("Attempting imports...")
try:
    # Test imports
    from alpha_pulse.exchanges.interfaces import BaseExchange, ExchangeConfiguration
    print("Imported interfaces")
    
    from alpha_pulse.exchanges.adapters.ccxt_adapter import CCXTAdapter
    print("Imported ccxt_adapter")
    
    from alpha_pulse.exchanges.factories import ExchangeFactory, ExchangeType
    print("Imported factories")
    
    from alpha_pulse.exchanges.base import OHLCV, Balance
    print("Imported base")
    
    print("All imports successful!")
except Exception as e:
    print(f"Imports failed: {e}")
    import traceback
    traceback.print_exc()