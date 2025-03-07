"""Alerts data access module."""
from typing import Dict, List, Optional
from datetime import datetime
import logging

from alpha_pulse.monitoring.alerting.manager import AlertManager


class AlertDataAccessor:
    """Access alert data."""
    
    def __init__(self):
        """Initialize alert accessor."""
        self.logger = logging.getLogger("alpha_pulse.api.data.alerts")
        self.alert_manager = AlertManager.get_instance()
    
    async def get_alerts(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Get alert history with optional filtering.
        
        Args:
            start_time: Filter alerts after this time
            end_time: Filter alerts before this time
            filters: Additional filters (severity, acknowledged, etc.)
            
        Returns:
            List of alert data
        """
        try:
            # Get alerts from alert manager
            alerts = await self.alert_manager.get_alert_history(
                start_time=start_time,
                end_time=end_time,
                filters=filters
            )
            
            # Transform to API format
            result = []
            for alert in alerts:
                result.append(alert.to_dict())
                
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving alerts: {str(e)}")
            return []
    
    async def acknowledge_alert(self, alert_id: str, user: str) -> Dict:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            user: User acknowledging the alert
            
        Returns:
            Updated alert data
        """
        try:
            # Acknowledge alert
            success = await self.alert_manager.acknowledge_alert(alert_id, user)
            
            if not success:
                return {"success": False, "error": "Alert not found or already acknowledged"}
            
            # Get updated alert
            for alert in await self.alert_manager.get_alert_history(
                filters={"alert_id": alert_id}
            ):
                return {
                    "success": True,
                    "alert": alert.to_dict()
                }
                
            return {"success": False, "error": "Alert not found after acknowledgment"}
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {str(e)}")
            return {"success": False, "error": str(e)}