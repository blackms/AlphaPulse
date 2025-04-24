"""
Minimal script to verify imports work correctly.
"""
import sys
import os
import importlib.util

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
print(f"Added {src_path} to Python path")

# Try to import alpha_pulse and see what happens
print("\nTrying to import alpha_pulse...")
try:
    import alpha_pulse
    print("Successfully imported alpha_pulse")
except Exception as e:
    print(f"Error importing alpha_pulse: {e}")
    import traceback
    traceback.print_exc()

# Try to import alpha_pulse.exchanges and see what happens
print("\nTrying to import alpha_pulse.exchanges...")
try:
    import alpha_pulse.exchanges
    print("Successfully imported alpha_pulse.exchanges")
except Exception as e:
    print(f"Error importing alpha_pulse.exchanges: {e}")
    import traceback
    traceback.print_exc()

# Check if modules exist without importing them
def check_module_exists(module_path):
    try:
        spec = importlib.util.find_spec(module_path)
        exists = spec is not None
        print(f"Module {module_path}: {'EXISTS' if exists else 'DOES NOT EXIST'}")
        return exists
    except ModuleNotFoundError:
        print(f"Module {module_path}: DOES NOT EXIST")
        return False

# Check if the modules we need exist
check_module_exists('alpha_pulse')
check_module_exists('alpha_pulse.exchanges')
check_module_exists('alpha_pulse.exchanges.interfaces')
check_module_exists('alpha_pulse.exchanges.adapters')
check_module_exists('alpha_pulse.exchanges.adapters.ccxt_adapter')
check_module_exists('alpha_pulse.exchanges.factories')
check_module_exists('alpha_pulse.exchanges.base')