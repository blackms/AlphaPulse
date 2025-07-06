"""
GPU monitoring and metrics API endpoints.
"""
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from ..dependencies import get_gpu_service
from ...ml.gpu.gpu_service import GPUService

router = APIRouter(prefix="/gpu", tags=["gpu"])


@router.get("/metrics", response_model=Dict[str, Any])
async def get_gpu_metrics(
    gpu_service: Optional[GPUService] = Depends(get_gpu_service)
) -> Dict[str, Any]:
    """
    Get comprehensive GPU metrics and performance data.
    
    Returns:
        Dictionary containing GPU metrics, utilization, and performance data
    """
    try:
        if not gpu_service:
            return {
                "gpu_available": False,
                "status": "GPU service not available",
                "fallback_mode": "cpu_only",
                "devices": {},
                "models": {},
                "performance": {}
            }
        
        # Get comprehensive metrics from GPU service
        metrics = gpu_service.get_metrics()
        
        # Add performance summary
        performance_summary = {
            "total_devices": len(metrics.get('devices', {})),
            "devices_available": sum(1 for d in metrics.get('devices', {}).values() 
                                   if d.get('memory_usage', 1.0) < 0.95),
            "avg_utilization": 0.0,
            "avg_memory_usage": 0.0,
            "total_models": len(metrics.get('models', {}))
        }
        
        # Calculate averages
        devices = metrics.get('devices', {})
        if devices:
            performance_summary["avg_utilization"] = sum(
                d.get('utilization', 0) for d in devices.values()
            ) / len(devices)
            performance_summary["avg_memory_usage"] = sum(
                d.get('memory_usage', 0) for d in devices.values()
            ) / len(devices)
        
        # Add service status
        result = {
            **metrics,
            "performance_summary": performance_summary,
            "service_status": "running" if metrics.get('service_running') else "stopped",
            "timestamp": "now"
        }
        
        logger.debug(f"GPU metrics retrieved: {len(devices)} devices, "
                    f"{performance_summary['total_models']} models")
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving GPU metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve GPU metrics: {str(e)}"
        )


@router.get("/status", response_model=Dict[str, Any])
async def get_gpu_status(
    gpu_service: Optional[GPUService] = Depends(get_gpu_service)
) -> Dict[str, Any]:
    """
    Get basic GPU service status.
    
    Returns:
        Basic GPU service status information
    """
    try:
        if not gpu_service:
            return {
                "status": "unavailable",
                "gpu_available": False,
                "message": "GPU service not initialized",
                "fallback_mode": "cpu_only"
            }
        
        metrics = gpu_service.get_metrics()
        devices = metrics.get('devices', {})
        
        return {
            "status": "running" if metrics.get('service_running') else "stopped",
            "gpu_available": True,
            "device_count": len(devices),
            "models_loaded": len(metrics.get('models', {})),
            "memory_utilization": {
                device_id: device_info.get('memory_usage', 0)
                for device_id, device_info in devices.items()
            },
            "gpu_utilization": {
                device_id: device_info.get('utilization', 0)
                for device_id, device_info in devices.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving GPU status: {e}")
        return {
            "status": "error",
            "gpu_available": False,
            "message": str(e),
            "fallback_mode": "cpu_only"
        }


@router.get("/devices", response_model=Dict[str, Any])
async def get_gpu_devices(
    gpu_service: Optional[GPUService] = Depends(get_gpu_service)
) -> Dict[str, Any]:
    """
    Get detailed information about available GPU devices.
    
    Returns:
        Detailed information about each GPU device
    """
    try:
        if not gpu_service:
            return {
                "devices": {},
                "total_devices": 0,
                "message": "GPU service not available"
            }
        
        metrics = gpu_service.get_metrics()
        devices = metrics.get('devices', {})
        
        # Enhanced device information
        enhanced_devices = {}
        for device_id, device_info in devices.items():
            enhanced_devices[device_id] = {
                **device_info,
                "status": "active" if device_info.get('utilization', 0) > 0.1 else "idle",
                "memory_status": (
                    "high" if device_info.get('memory_usage', 0) > 0.8 else
                    "medium" if device_info.get('memory_usage', 0) > 0.5 else
                    "low"
                ),
                "performance_category": (
                    "excellent" if device_info.get('utilization', 0) > 0.8 else
                    "good" if device_info.get('utilization', 0) > 0.5 else
                    "idle"
                )
            }
        
        return {
            "devices": enhanced_devices,
            "total_devices": len(enhanced_devices),
            "available_devices": sum(1 for d in enhanced_devices.values() 
                                   if d.get('memory_usage', 1.0) < 0.95),
            "active_devices": sum(1 for d in enhanced_devices.values() 
                                if d.get('status') == 'active')
        }
        
    except Exception as e:
        logger.error(f"Error retrieving GPU devices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve GPU devices: {str(e)}"
        )


@router.get("/models", response_model=Dict[str, Any])
async def get_gpu_models(
    gpu_service: Optional[GPUService] = Depends(get_gpu_service)
) -> Dict[str, Any]:
    """
    Get information about models loaded on GPU.
    
    Returns:
        Information about GPU models and their status
    """
    try:
        if not gpu_service:
            return {
                "models": {},
                "total_models": 0,
                "message": "GPU service not available"
            }
        
        metrics = gpu_service.get_metrics()
        models = metrics.get('models', {})
        
        # Enhanced model information
        enhanced_models = {}
        for model_name, model_info in models.items():
            enhanced_models[model_name] = {
                **model_info,
                "memory_footprint": "estimated" if 'device_id' in model_info else "unknown",
                "last_used": model_info.get('created_at', 'unknown')
            }
        
        return {
            "models": enhanced_models,
            "total_models": len(enhanced_models),
            "models_by_device": {},  # Could be enhanced to group by device
            "model_types": list(set(m.get('type', 'unknown') for m in enhanced_models.values()))
        }
        
    except Exception as e:
        logger.error(f"Error retrieving GPU models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve GPU models: {str(e)}"
        )


@router.get("/performance", response_model=Dict[str, Any])
async def get_gpu_performance(
    gpu_service: Optional[GPUService] = Depends(get_gpu_service)
) -> Dict[str, Any]:
    """
    Get GPU performance analytics and batch processing stats.
    
    Returns:
        GPU performance metrics and analytics
    """
    try:
        if not gpu_service:
            return {
                "performance": {},
                "batch_processing": {},
                "message": "GPU service not available"
            }
        
        metrics = gpu_service.get_metrics()
        
        # Extract performance data
        batch_stats = metrics.get('batch_processing', {})
        devices = metrics.get('devices', {})
        memory_stats = metrics.get('memory', {})
        
        performance_data = {
            "batch_processing": {
                **batch_stats,
                "efficiency": batch_stats.get('avg_batch_size', 0) / max(batch_stats.get('max_batch_size', 1), 1),
                "throughput_score": batch_stats.get('processed_batches', 0) / max(batch_stats.get('total_time', 1), 1)
            },
            "memory_efficiency": {
                device_id: {
                    "allocated_mb": memory_info.get('allocated_mb', 0),
                    "reserved_mb": memory_info.get('reserved_mb', 0),
                    "free_mb": memory_info.get('free_mb', 0),
                    "efficiency": memory_info.get('allocated_mb', 0) / max(memory_info.get('reserved_mb', 1), 1)
                }
                for device_id, memory_info in memory_stats.items()
            },
            "device_performance": {
                device_id: {
                    "utilization": device_info.get('utilization', 0),
                    "memory_usage": device_info.get('memory_usage', 0),
                    "temperature": device_info.get('temperature', 0),
                    "performance_score": (
                        device_info.get('utilization', 0) * 0.6 +
                        (1 - device_info.get('memory_usage', 0)) * 0.4
                    )
                }
                for device_id, device_info in devices.items()
            }
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error retrieving GPU performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve GPU performance: {str(e)}"
        )