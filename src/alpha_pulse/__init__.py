"""
AlphaPulse package initialization.
"""

# Version of the alpha_pulse package
__version__ = "0.1.0"

# Import core modules
from . import config
from . import data_pipeline
from . import features
from . import utils

# Lazy imports to avoid circular dependencies
def get_models():
    from . import models
    return models

def get_backtesting():
    from . import backtesting
    return backtesting

def get_exchange():
    from . import exchange
    return exchange

# These imports should be moved to avoid circular dependencies
# from . import models  # This is causing the circular import
# from . import backtesting
# from . import exchange

# Instead, expose specific components that are needed
__all__ = [
    'config',
    'data_pipeline',
    'features',
    'utils',
    'get_models',
    'get_backtesting',
    'get_exchange',
]

from alpha_pulse.models import ModelTrainer