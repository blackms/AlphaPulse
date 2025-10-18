"""
Logging utilities for AlphaPulse.

Provides standardized logging configuration using loguru.
"""
from loguru import logger
from typing import Any


def get_logger(name: str) -> Any:
    """
    Get a logger instance with the given name.

    Uses loguru's global logger with contextual binding for the module name.

    Args:
        name: The name for the logger (typically __name__)

    Returns:
        A logger instance bound to the given name

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    return logger.bind(name=name)
