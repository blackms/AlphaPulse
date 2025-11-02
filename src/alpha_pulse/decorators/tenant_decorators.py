"""Decorators for multi-tenant context validation.

This module provides decorators to ensure tenant_id parameters are properly
validated in multi-tenant service methods.
"""
import asyncio
from functools import wraps
from typing import Callable
from loguru import logger


def require_tenant_id(func: Callable) -> Callable:
    """
    Decorator to validate tenant_id parameter is provided.

    This decorator ensures that all service methods requiring tenant context
    receive a valid tenant_id parameter. It works with both async and sync
    functions.

    Args:
        func: Function to decorate (can be async or sync)

    Returns:
        Wrapped function with tenant_id validation

    Raises:
        ValueError: If tenant_id is None, empty string, or not provided

    Example:
        >>> @require_tenant_id
        ... async def generate_signals(self, market_data, tenant_id: str):
        ...     # tenant_id is validated before this executes
        ...     return signals

        >>> # This will raise ValueError
        >>> await generate_signals(market_data)  # Missing tenant_id

        >>> # This works
        >>> await generate_signals(market_data, tenant_id="uuid-here")

    Notes:
        - The decorator expects tenant_id to be passed as a keyword argument
        - It logs the tenant_id at DEBUG level for troubleshooting
        - The error message includes the function name for clarity
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        tenant_id = kwargs.get('tenant_id')
        if not tenant_id:
            raise ValueError(
                f"{func.__qualname__} requires 'tenant_id' parameter. "
                f"Multi-tenant context is mandatory for data isolation."
            )
        logger.debug(f"{func.__qualname__} called for tenant {tenant_id}")
        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        tenant_id = kwargs.get('tenant_id')
        if not tenant_id:
            raise ValueError(
                f"{func.__qualname__} requires 'tenant_id' parameter. "
                f"Multi-tenant context is mandatory for data isolation."
            )
        logger.debug(f"{func.__qualname__} called for tenant {tenant_id}")
        return func(*args, **kwargs)

    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
