"""
Database-based alert history storage for the alerting system.

This module provides PostgreSQL-based alert history storage for the alerting system.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging
import json
import asyncio
import os
import asyncpg

from .models import Alert, AlertSeverity
from .history import AlertHistoryStorage


class DatabaseAlertHistory(AlertHistoryStorage):
    """PostgreSQL-based alert history storage."""
    
    def __init__(self, connection_params: Dict[str, Any]):
        """Initialize with PostgreSQL connection parameters.
        
        Args:
            connection_params: PostgreSQL connection parameters
        """
        self.connection_params = connection_params
        self.logger = logging.getLogger("alpha_pulse.alerting.history.database")
        self.lock = asyncio.Lock()
        self.initialized = False
        self.pool = None
    
    async def initialize(self) -> bool:
        """Initialize the database.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            async with self.lock:
                # Create a connection pool
                self.pool = await asyncpg.create_pool(**self.connection_params)
                
                # Create alerts table if it doesn't exist
                acquire_result = await self.pool.acquire() # Added await
                self.logger.debug(f"pool.acquire() returned: {acquire_result}")
                self.logger.debug(f"Type of acquire_result: {type(acquire_result)}")
                self.logger.debug(f"Has __aenter__: {hasattr(acquire_result, '__aenter__')}")
                self.logger.debug(f"Has __aexit__: {hasattr(acquire_result, '__aexit__')}")
                async with acquire_result as conn:
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS alerts (
                            alert_id TEXT PRIMARY KEY,
                            rule_id TEXT NOT NULL,
                            metric_name TEXT NOT NULL,
                            metric_value TEXT NOT NULL,
                            severity TEXT NOT NULL,
                            message TEXT NOT NULL,
                            timestamp TIMESTAMP NOT NULL,
                            acknowledged BOOLEAN NOT NULL DEFAULT FALSE,
                            acknowledged_by TEXT,
                            acknowledged_at TIMESTAMP
                        )
                    ''')
            
            self.initialized = True
            self.logger.info("PostgreSQL database alert history initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database alert history: {str(e)}")
            return False
    
    async def store_alert(self, alert: Alert) -> None:
        """Store an alert in the database.
        
        Args:
            alert: The alert to store
        """
        if not self.initialized:
            await self.initialize()
            
        async with self.lock:
            try:
                acquire_result = await self.pool.acquire() # Added await
                self.logger.debug(f"store_alert: pool.acquire() returned: {acquire_result}")
                self.logger.debug(f"store_alert: Type of acquire_result: {type(acquire_result)}")
                self.logger.debug(f"store_alert: Has __aenter__: {hasattr(acquire_result, '__aenter__')}")
                self.logger.debug(f"store_alert: Has __aexit__: {hasattr(acquire_result, '__aexit__')}")
                async with acquire_result as conn:
                    await conn.execute('''
                        INSERT INTO alerts (
                            alert_id, rule_id, metric_name, metric_value, severity,
                            message, timestamp, acknowledged, acknowledged_by, acknowledged_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ''',
                        alert.alert_id,
                        alert.rule_id,
                        alert.metric_name,
                        json.dumps(alert.metric_value),
                        alert.severity.value,
                        alert.message,
                        alert.timestamp,
                        alert.acknowledged,
                        alert.acknowledged_by,
                        alert.acknowledged_at
                    )
                    
                self.logger.debug(f"Stored alert in database: {alert.alert_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to store alert in database: {str(e)}")
    
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
        if not self.initialized:
            await self.initialize()
            
        alerts = []
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []
        param_index = 1
        
        # Apply time filters
        if start_time:
            query += f" AND timestamp >= ${param_index}"
            params.append(start_time)
            param_index += 1
        if end_time:
            query += f" AND timestamp <= ${param_index}"
            params.append(end_time)
            param_index += 1
        
        # Apply additional filters
        if filters:
            if "severity" in filters:
                query += f" AND severity = ${param_index}"
                params.append(filters["severity"])
                param_index += 1
            if "acknowledged" in filters:
                query += f" AND acknowledged = ${param_index}"
                params.append(filters["acknowledged"])
                param_index += 1
            if "rule_id" in filters:
                query += f" AND rule_id = ${param_index}"
                params.append(filters["rule_id"])
                param_index += 1
            if "metric_name" in filters:
                query += f" AND metric_name = ${param_index}"
                params.append(filters["metric_name"])
                param_index += 1
        
        # Order by timestamp (newest first)
        query += " ORDER BY timestamp DESC"
        
        try:
            acquire_result = await self.pool.acquire() # Added await
            self.logger.debug(f"get_alerts: pool.acquire() returned: {acquire_result}")
            self.logger.debug(f"get_alerts: Type of acquire_result: {type(acquire_result)}")
            self.logger.debug(f"get_alerts: Has __aenter__: {hasattr(acquire_result, '__aenter__')}")
            self.logger.debug(f"get_alerts: Has __aexit__: {hasattr(acquire_result, '__aexit__')}")
            async with acquire_result as conn:
                rows = await conn.fetch(query, *params)
                for row in rows:
                    try:
                        # Convert row to Alert object
                        alert = Alert(
                            alert_id=row["alert_id"],
                            rule_id=row["rule_id"],
                            metric_name=row["metric_name"],
                            metric_value=json.loads(row["metric_value"]),
                            severity=row["severity"],
                            message=row["message"],
                            timestamp=row["timestamp"],
                            acknowledged=row["acknowledged"],
                            acknowledged_by=row["acknowledged_by"],
                            acknowledged_at=row["acknowledged_at"]
                        )
                        alerts.append(alert)
                    except (json.JSONDecodeError, ValueError) as e:
                        self.logger.error(f"Failed to parse alert from database: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve alerts from database: {str(e)}")
        
        return alerts
    
    async def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> bool:
        """Update an alert record.
        
        Args:
            alert_id: ID of the alert to update
            updates: Dictionary of fields to update
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if not self.initialized:
            await self.initialize()
            
        async with self.lock:
            try:
                # Build update query
                set_clauses = []
                params = []
                param_index = 1
                
                for key, value in updates.items():
                    if key == "acknowledged":
                        set_clauses.append(f"acknowledged = ${param_index}")
                        params.append(value)
                        param_index += 1
                    elif key == "acknowledged_by":
                        set_clauses.append(f"acknowledged_by = ${param_index}")
                        params.append(value)
                        param_index += 1
                    elif key == "acknowledged_at":
                        set_clauses.append(f"acknowledged_at = ${param_index}")
                        params.append(value)
                        param_index += 1
                    # Add other fields as needed
                
                if not set_clauses:
                    return False
                
                # Add alert_id to params
                params.append(alert_id)
                
                # Execute update
                acquire_result = await self.pool.acquire() # Added await
                self.logger.debug(f"update_alert: pool.acquire() returned: {acquire_result}")
                self.logger.debug(f"update_alert: Type of acquire_result: {type(acquire_result)}")
                self.logger.debug(f"update_alert: Has __aenter__: {hasattr(acquire_result, '__aenter__')}")
                self.logger.debug(f"update_alert: Has __aexit__: {hasattr(acquire_result, '__aexit__')}")
                async with acquire_result as conn:
                    status = await conn.execute(
                        f"UPDATE alerts SET {', '.join(set_clauses)} WHERE alert_id = ${param_index}",
                        *params
                    )
                    
                    if status.endswith("0"):
                        self.logger.warning(f"Alert not found for update: {alert_id}")
                        return False
                        # Note: The original code returned False here, which is correct for update failure.
                    else:
                        self.logger.debug(f"Updated alert in database: {alert_id}")
                        return True
            except Exception as e:
                self.logger.error(f"Failed to update alert in database: {str(e)}")
                return False


# Update the create_alert_history function in history.py to include the database option
def create_database_alert_history(config: Dict[str, Any]) -> AlertHistoryStorage:
    """Create a database-based alert history storage.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AlertHistoryStorage: Database alert history storage instance
    """
    # Build PostgreSQL connection params from config
    connection_params = {
        "host": config.get("host", "localhost"),
        "port": config.get("port", 5432),
        "user": config.get("user", "postgres"),
        "password": config.get("password", "postgres"),
        "database": config.get("database", "alphapulse"),
    }
    
    storage = DatabaseAlertHistory(connection_params=connection_params)
    asyncio.create_task(storage.initialize())
    return storage