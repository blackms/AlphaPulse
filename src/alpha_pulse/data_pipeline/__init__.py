"""
Data pipeline package.

This package provides functionality for data processing,
caching, and background tasks.
"""
# Expose key components from the new session module
from .session import async_session, get_db # Remove engine export