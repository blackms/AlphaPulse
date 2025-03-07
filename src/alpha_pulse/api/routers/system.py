"""
System router.

This module defines the API endpoints for system data.
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends

from ..dependencies import (
    require_view_system,
    get_system_accessor
)
from ..data import SystemDataAccessor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/system")
async def get_system_metrics(
    _: Dict[str, Any] = Depends(require_view_system),
    system_accessor: SystemDataAccessor = Depends(get_system_accessor)
) -> Dict[str, Any]:
    """
    Get system metrics.
    
    Returns:
        System metrics data
    """
    return await system_accessor.get_system_metrics()