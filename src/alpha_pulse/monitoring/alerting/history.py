"""
Alert history storage for the alerting system.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import os
import logging
import aiofiles
import asyncio
from collections import deque


from .models import Alert


class AlertHistoryStorage:
    """Base class for alert history storage."""
    
    async def store_alert(self, alert: Alert) -> None:
        """Store an alert in history.
        
        Args:
            alert: The alert to store
        """
        raise NotImplementedError("Subclasses must implement store_alert")
    
    async def get_alerts(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Alert]:
        """Retrieve alerts with optional filtering.
        
        Args:
            start_time: Filter alerts after this time
            end_time: Filter alerts before this time
            filters: Additional filters (severity, acknowledged, etc.)
            
        Returns:
            List[Alert]: Filtered alert history
        """
        raise NotImplementedError("Subclasses must implement get_alerts")
    
    async def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> bool:
        """Update an alert record.
        
        Args:
            alert_id: ID of the alert to update
            updates: Dictionary of fields to update
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement update_alert")


class MemoryAlertHistory(AlertHistoryStorage):
    """In-memory alert history storage."""
    
    def __init__(self, max_alerts: int = 1000):
        """Initialize with maximum number of alerts to store.
        
        Args:
            max_alerts: Maximum number of alerts to keep in memory
        """
        self.alerts = deque(maxlen=max_alerts)
        self.logger = logging.getLogger("alpha_pulse.alerting.history.memory")
    
    async def store_alert(self, alert: Alert) -> None:
        """Store an alert in memory.
        
        Args:
            alert: The alert to store
        """
        self.alerts.append(alert)
        self.logger.debug(f"Stored alert in memory: {alert.alert_id}")
    
    async def get_alerts(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Alert]:
        """Retrieve alerts with optional filtering.
        
        Args:
            start_time: Filter alerts after this time
            end_time: Filter alerts before this time
            filters: Additional filters (severity, acknowledged, etc.)
            
        Returns:
            List[Alert]: Filtered alert history
        """
        filtered_alerts = list(self.alerts)
        
        # Apply time filters
        if start_time:
            filtered_alerts = [a for a in filtered_alerts if a.timestamp >= start_time]
        if end_time:
            filtered_alerts = [a for a in filtered_alerts if a.timestamp <= end_time]
        
        # Apply additional filters
        if filters:
            for key, value in filters.items():
                if key == "severity":
                    filtered_alerts = [a for a in filtered_alerts if a.severity.value == value]
                elif key == "acknowledged":
                    filtered_alerts = [a for a in filtered_alerts if a.acknowledged == value]
                elif key == "rule_id":
                    filtered_alerts = [a for a in filtered_alerts if a.rule_id == value]
                elif key == "metric_name":
                    filtered_alerts = [a for a in filtered_alerts if a.metric_name == value]
        
        # Sort by timestamp (newest first)
        filtered_alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return filtered_alerts
    
    async def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> bool:
        """Update an alert record.
        
        Args:
            alert_id: ID of the alert to update
            updates: Dictionary of fields to update
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                # Update fields
                for key, value in updates.items():
                    if hasattr(alert, key):
                        setattr(alert, key, value)
                
                self.logger.debug(f"Updated alert in memory: {alert_id}")
                return True
        
        return False


class FileAlertHistory(AlertHistoryStorage):
    """File-based alert history storage."""
    
    def __init__(self, file_path: str):
        """Initialize with file path.
        
        Args:
            file_path: Path to the alert history file
        """
        self.file_path = file_path
        self.logger = logging.getLogger("alpha_pulse.alerting.history.file")
        self.lock = asyncio.Lock()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    async def store_alert(self, alert: Alert) -> None:
        """Store an alert in the file.
        
        Args:
            alert: The alert to store
        """
        async with self.lock:
            try:
                # Convert alert to dict
                alert_dict = alert.to_dict()
                
                # Append to file
                async with aiofiles.open(self.file_path, "a") as f:
                    await f.write(json.dumps(alert_dict) + "\n")
                
                self.logger.debug(f"Stored alert in file: {alert.alert_id}")
            except Exception as e:
                self.logger.error(f"Failed to store alert in file: {str(e)}")
    
    async def get_alerts(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Alert]:
        """Retrieve alerts with optional filtering.
        
        Args:
            start_time: Filter alerts after this time
            end_time: Filter alerts before this time
            filters: Additional filters (severity, acknowledged, etc.)
            
        Returns:
            List[Alert]: Filtered alert history
        """
        alerts = []
        
        try:
            # Check if file exists
            if not os.path.exists(self.file_path):
                return []
            
            # Read all alerts from file
            async with aiofiles.open(self.file_path, "r") as f:
                async for line in f:
                    try:
                        alert_dict = json.loads(line.strip())
                        alert = Alert.from_dict(alert_dict)
                        alerts.append(alert)
                    except (json.JSONDecodeError, KeyError) as e:
                        self.logger.error(f"Failed to parse alert from file: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to read alerts from file: {str(e)}")
        
        # Apply time filters
        if start_time:
            alerts = [a for a in alerts if a.timestamp >= start_time]
        if end_time:
            alerts = [a for a in alerts if a.timestamp <= end_time]
        
        # Apply additional filters
        if filters:
            for key, value in filters.items():
                if key == "severity":
                    alerts = [a for a in alerts if a.severity.value == value]
                elif key == "acknowledged":
                    alerts = [a for a in alerts if a.acknowledged == value]
                elif key == "rule_id":
                    alerts = [a for a in alerts if a.rule_id == value]
                elif key == "metric_name":
                    alerts = [a for a in alerts if a.metric_name == value]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts
    
    async def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> bool:
        """Update an alert record.
        
        Args:
            alert_id: ID of the alert to update
            updates: Dictionary of fields to update
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        async with self.lock:
            try:
                # Read all alerts
                alerts = await self.get_alerts()
                
                # Find and update the alert
                updated = False
                for alert in alerts:
                    if alert.alert_id == alert_id:
                        # Update fields
                        for key, value in updates.items():
                            if hasattr(alert, key):
                                setattr(alert, key, value)
                        updated = True
                        break
                
                if not updated:
                    return False
                
                # Write all alerts back to file
                async with aiofiles.open(self.file_path, "w") as f:
                    for alert in alerts:
                        await f.write(json.dumps(alert.to_dict()) + "\n")
                
                self.logger.debug(f"Updated alert in file: {alert_id}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to update alert in file: {str(e)}")
                return False


def create_alert_history(config: Dict[str, Any]) -> AlertHistoryStorage:
    """Create alert history storage based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AlertHistoryStorage: Alert history storage instance
    """
    storage_type = config.get("type", "memory")
    
    if storage_type == "memory":
        max_alerts = config.get("max_alerts", 1000)
        return MemoryAlertHistory(max_alerts=max_alerts)
    elif storage_type == "file":
        file_path = config.get("file_path", "alerts.jsonl")
        return FileAlertHistory(file_path=file_path)
    else:
        raise ValueError(f"Unsupported alert history storage type: {storage_type}")