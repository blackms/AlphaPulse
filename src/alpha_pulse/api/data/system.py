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
        self._last_cpu_readings = []
        self._max_readings = 12  # Store 1 minute of readings at 5-second intervals
    
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
            
            # Store CPU reading for historical tracking
            self._last_cpu_readings.append(cpu_percent)
            if len(self._last_cpu_readings) > self._max_readings:
                self._last_cpu_readings.pop(0)
            
            # Calculate if CPU has been high for extended period
            cpu_high_duration = 0
            if len(self._last_cpu_readings) >= 6:  # At least 30 seconds of data
                high_count = sum(1 for reading in self._last_cpu_readings[-6:] if reading > 80)
                cpu_high_duration = (high_count * 5) if high_count >= 6 else 0
            
            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "cores": psutil.cpu_count(),
                    "high_duration_seconds": cpu_high_duration
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