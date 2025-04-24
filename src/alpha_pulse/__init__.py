"""
AlphaPulse package initialization.
"""

# Version of the alpha_pulse package
__version__ = "0.1.0"

# Import core modules
from . import config
from . import data_pipeline
from . import features
from . import exchanges

# Instead, expose specific components that are needed
__all__ = [
    'config',
    'data_pipeline',
    'features',
    'exchanges',
]