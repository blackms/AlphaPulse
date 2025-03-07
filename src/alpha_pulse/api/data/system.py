"""System data access module."""
from typing import Dict, List, Optional
from datetime import datetime
import logging
import psutil
import os


class SystemDataAccessor:
    """Access system metrics."""
    
    def __init__(self):
        """Initialize system accessor."""
        self.logger = logging.getLogger("alpha_pulse.api.data.system")
    
    async def get_system_metrics(self) -> Dict:
        """
        Get current system metrics.
        
        Returns:
            System metrics
        """
        try:
            # Get basic system metrics
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "cores": psutil.cpu_count()
                },
                "memory": {
                    "total_mb": memory.total / (1024 * 1024),
                    "used_mb": memory.used / (1024 * 1024),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": disk.total / (1024 * 1024 * 1024),
                    "used_gb": disk.used / (1024 * 1024 * 1024),
                    "percent": disk.percent
                },
                "process": {
                    "pid": os.getpid(),
                    "memory_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),
                    "threads": psutil.Process(os.getpid()).num_threads(),
                    "uptime_seconds": int(datetime.now().timestamp() - psutil.Process(os.getpid()).create_time())
                }
            }
        except Exception as e:
            self.logger.error(f"Error retrieving system metrics: {str(e)}")
            return {
                "error": str(e)
            }