"""Alerts data access module."""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from alpha_pulse.monitoring.alerting import AlertManager
from alpha_pulse.monitoring.alerting.models import Alert


class AlertDataAccessor:
    """Access alert data from the alerting system."""
    
    def __init__(self, alert_manager: AlertManager):
        """Initialize alert accessor with alert manager.
        
        Args:
            alert_manager: The alerting system manager
        """
        self.logger = logging.getLogger("alpha_pulse.api.data.alerts")
        self.alert_manager = alert_manager
    
    async def get_alerts(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
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
            # Set default start time if not provided (24 hours ago)
            if start_time is None:
                start_time = datetime.now() - timedelta(days=1)
            
            # Get alerts from the alert manager
            alerts = await self.alert_manager.get_alert_history(
                start_time=start_time,
                end_time=end_time,
                filters=filters
            )
            
            # Convert Alert objects to dictionaries for API response
            return [self._format_alert_for_api(alert) for alert in alerts]
            
        except Exception as e:
            self.logger.error(f"Error retrieving alerts: {str(e)}")
            return []
    
    async def acknowledge_alert(self, alert_id: str, user: str) -> Dict[str, Any]:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            user: User acknowledging the alert
            
        Returns:
            Updated alert data
        """
        try:
            # Acknowledge the alert through the alert manager
            success = await self.alert_manager.acknowledge_alert(alert_id, user)
            
            if success:
                # Get the updated alert
                alerts = await self.alert_manager.get_alert_history(
                    filters={"alert_id": alert_id}
                )
                
                if alerts:
                    return {
                        "success": True,
                        "alert": self._format_alert_for_api(alerts[0])
                    }
                else:
                    return {
                        "success": True,
                        "alert": {"alert_id": alert_id, "acknowledged": True}
                    }
            
            return {
                "success": False,
                "error": "Alert not found or already acknowledged"
            }
            
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _format_alert_for_api(self, alert: Alert) -> Dict[str, Any]:
        """
        Format an Alert object for API response.
        
        Args:
            alert: The Alert object
            
        Returns:
            Dictionary representation of the alert
        """
        # Convert Alert object to a dictionary
        alert_dict = alert.to_dict()
        
        # Rename and restructure fields to match API format
        return {
            "id": alert_dict["alert_id"],
            "title": f"{alert_dict['metric_name']} Alert",
            "message": alert_dict["message"],
            "severity": alert_dict["severity"],
            "source": alert_dict["metric_name"],
            "created_at": alert_dict["timestamp"],
            "acknowledged": alert_dict["acknowledged"],
            "acknowledged_by": alert_dict["acknowledged_by"],
            "acknowledged_at": alert_dict["acknowledged_at"],
            "tags": [alert_dict["metric_name"], alert_dict["severity"]]
        }