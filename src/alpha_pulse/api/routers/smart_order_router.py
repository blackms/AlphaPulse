"""
API endpoints for Smart Order Router.
"""
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from datetime import datetime

from alpha_pulse.services.smart_order_router_service import SmartOrderRouterService
from alpha_pulse.execution.smart_order_router import (
    SmartOrderConfig, 
    ExecutionStrategy, 
    OrderUrgency,
    SmartOrder
)
from alpha_pulse.execution.broker_interface import OrderSide
from ..dependencies import get_user


# Pydantic models for API
class SmartOrderConfigModel(BaseModel):
    """Smart order configuration model."""
    strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    urgency: OrderUrgency = OrderUrgency.MEDIUM
    max_participation_rate: float = Field(0.3, ge=0.01, le=1.0)
    max_slice_size: float = Field(0.1, ge=0.001, le=1.0)
    time_horizon_minutes: int = Field(30, ge=1, le=1440)
    min_slice_interval_seconds: int = Field(30, ge=1, le=3600)
    max_market_impact_bps: int = Field(50, ge=1, le=1000)
    enable_dark_pools: bool = True
    enable_regime_adaptation: bool = True
    slippage_tolerance_bps: int = Field(100, ge=1, le=1000)


class SubmitOrderRequest(BaseModel):
    """Request model for submitting a smart order."""
    symbol: str = Field(..., min_length=1, max_length=20)
    side: OrderSide
    quantity: float = Field(..., gt=0)
    order_type: str = Field("medium_order", regex="^(small_order|medium_order|large_order|urgent_order)$")
    custom_config: Optional[SmartOrderConfigModel] = None
    target_price: Optional[float] = Field(None, gt=0)
    stop_price: Optional[float] = Field(None, gt=0)


class SubmitOrderResponse(BaseModel):
    """Response model for submitted smart order."""
    order_id: str
    message: str
    timestamp: datetime


class OrderStatusResponse(BaseModel):
    """Response model for order status."""
    order_id: str
    symbol: str
    side: OrderSide
    total_quantity: float
    filled_quantity: float
    avg_filled_price: Optional[float]
    status: str
    created_at: datetime
    num_slices: int
    execution_stats: Dict[str, Any]


class ActiveOrdersResponse(BaseModel):
    """Response model for active orders."""
    active_orders: List[OrderStatusResponse]
    total_active: int


class PerformanceStatsResponse(BaseModel):
    """Response model for performance statistics."""
    service_stats: Dict[str, Any]
    execution_stats: Dict[str, Any]
    configuration: Dict[str, Any]


class MarketDataUpdate(BaseModel):
    """Market data update model."""
    symbol: str = Field(..., min_length=1, max_length=20)
    price: float = Field(..., gt=0)
    volume: float = Field(0.0, ge=0)
    spread: float = Field(0.001, ge=0)


# Router instance
router = APIRouter(prefix="/smart-order-router", tags=["smart-order-router"])

# Service instance - will be set during app startup
_smart_order_service: Optional[SmartOrderRouterService] = None


def get_smart_order_service():
    """Get the smart order router service instance."""
    if _smart_order_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Smart Order Router service not available"
        )
    return _smart_order_service


@router.post("/orders", response_model=SubmitOrderResponse)
async def submit_smart_order(
    request: SubmitOrderRequest,
    user: Dict[str, Any] = Depends(get_user),
    service: SmartOrderRouterService = Depends(get_smart_order_service)
):
    """
    Submit a smart order for execution.
    
    The smart order router will intelligently execute the order using
    the specified strategy and configuration.
    """
    try:
        # Convert custom config if provided
        custom_config = None
        if request.custom_config:
            from alpha_pulse.execution.smart_order_router import SmartOrderConfig
            custom_config = SmartOrderConfig(
                strategy=request.custom_config.strategy,
                urgency=request.custom_config.urgency,
                max_participation_rate=request.custom_config.max_participation_rate,
                max_slice_size=request.custom_config.max_slice_size,
                time_horizon_minutes=request.custom_config.time_horizon_minutes,
                min_slice_interval_seconds=request.custom_config.min_slice_interval_seconds,
                max_market_impact_bps=request.custom_config.max_market_impact_bps,
                enable_dark_pools=request.custom_config.enable_dark_pools,
                enable_regime_adaptation=request.custom_config.enable_regime_adaptation,
                slippage_tolerance_bps=request.custom_config.slippage_tolerance_bps
            )
        
        # Submit order
        order_id = await service.submit_smart_order(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            order_type=request.order_type,
            custom_config=custom_config,
            target_price=request.target_price,
            stop_price=request.stop_price
        )
        
        return SubmitOrderResponse(
            order_id=order_id,
            message=f"Smart order submitted successfully",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to submit smart order: {str(e)}"
        )


@router.delete("/orders/{order_id}")
async def cancel_smart_order(
    order_id: str,
    user: Dict[str, Any] = Depends(get_user),
    service: SmartOrderRouterService = Depends(get_smart_order_service)
):
    """Cancel a smart order."""
    try:
        success = await service.cancel_order(order_id)
        
        if success:
            return {"message": f"Order {order_id} cancelled successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order {order_id} not found or already completed"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel order: {str(e)}"
        )


@router.get("/orders/{order_id}", response_model=OrderStatusResponse)
async def get_order_status(
    order_id: str,
    user: Dict[str, Any] = Depends(get_user),
    service: SmartOrderRouterService = Depends(get_smart_order_service)
):
    """Get status of a specific smart order."""
    try:
        order = service.get_order_status(order_id)
        
        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order {order_id} not found"
            )
        
        return OrderStatusResponse(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            total_quantity=order.total_quantity,
            filled_quantity=order.filled_quantity,
            avg_filled_price=order.avg_filled_price,
            status=order.status.value,
            created_at=order.created_at,
            num_slices=len(order.slices),
            execution_stats=order.execution_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get order status: {str(e)}"
        )


@router.get("/orders", response_model=ActiveOrdersResponse)
async def get_active_orders(
    user: Dict[str, Any] = Depends(get_user),
    service: SmartOrderRouterService = Depends(get_smart_order_service)
):
    """Get all active smart orders."""
    try:
        active_orders = service.get_active_orders()
        
        order_responses = []
        for order in active_orders.values():
            order_responses.append(OrderStatusResponse(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                total_quantity=order.total_quantity,
                filled_quantity=order.filled_quantity,
                avg_filled_price=order.avg_filled_price,
                status=order.status.value,
                created_at=order.created_at,
                num_slices=len(order.slices),
                execution_stats=order.execution_stats
            ))
        
        return ActiveOrdersResponse(
            active_orders=order_responses,
            total_active=len(order_responses)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active orders: {str(e)}"
        )


@router.get("/performance", response_model=PerformanceStatsResponse)
async def get_performance_stats(
    user: Dict[str, Any] = Depends(get_user),
    service: SmartOrderRouterService = Depends(get_smart_order_service)
):
    """Get performance statistics for the smart order router."""
    try:
        stats = service.get_performance_stats()
        
        return PerformanceStatsResponse(
            service_stats=stats['service_stats'],
            execution_stats=stats['execution_stats'],
            configuration=stats['configuration']
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance stats: {str(e)}"
        )


@router.post("/market-data")
async def update_market_data(
    market_data: MarketDataUpdate,
    user: Dict[str, Any] = Depends(get_user),
    service: SmartOrderRouterService = Depends(get_smart_order_service)
):
    """Update market data for routing decisions."""
    try:
        service.update_market_data(
            symbol=market_data.symbol,
            price=market_data.price,
            volume=market_data.volume,
            spread=market_data.spread
        )
        
        return {"message": f"Market data updated for {market_data.symbol}"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update market data: {str(e)}"
        )


@router.get("/health")
async def health_check(
    service: SmartOrderRouterService = Depends(get_smart_order_service)
):
    """Get health status of the smart order router service."""
    try:
        health_status = await service.health_check()
        
        if health_status['status'] == 'error':
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=health_status
            )
        
        return health_status
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


# Service initialization functions
def initialize_smart_order_service(service: SmartOrderRouterService):
    """Initialize the smart order router service."""
    global _smart_order_service
    _smart_order_service = service


async def start_smart_order_service():
    """Start the smart order router service."""
    if _smart_order_service:
        await _smart_order_service.start()


async def stop_smart_order_service():
    """Stop the smart order router service."""
    if _smart_order_service:
        await _smart_order_service.stop() 