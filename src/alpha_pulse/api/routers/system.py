"""System router."""
from typing import Dict
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_current_user, has_permission
from ..data import SystemDataAccessor

router = APIRouter()
system_accessor = SystemDataAccessor()


@router.get("/", response_model=Dict)
async def get_system_metrics(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get current system metrics.
    
    Returns:
        System metrics
    """
    # Check permissions
    if not has_permission(current_user, "view_system"):
        raise HTTPException(status_code=403, detail="Not authorized to view system metrics")
    
    # Get from data accessor
    return await system_accessor.get_system_metrics()