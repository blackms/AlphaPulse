"""
AlphaPulse - A powerful and efficient trading data pipeline system.
"""

__version__ = "0.1.0"

# Import order matters to avoid circular dependencies
from . import config
from . import features
from . import models
from . import data_pipeline
from . import backtesting

__all__ = [
    'config',
    'features',
    'models',
    'data_pipeline',
    'backtesting',
]