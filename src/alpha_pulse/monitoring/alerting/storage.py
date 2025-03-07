"""
Database-based alert history storage for the alerting system.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging
import json
import asyncio
import aiosqlite

from .models import Alert, AlertSeverity
from .history import AlertHistoryStorage


class DatabaseAlertHistory(AlertHistoryStorage):
    """SQLite-based alert history storage."""
    
    def __init__(self, db_path: str):
        """Initialize with database path.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger("alpha_pulse.alerting.history.database")
        self.lock = asyncio.Lock()
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the database.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            async with self.lock:
                async with aiosqlite.connect(self.db_path) as db:
                    # Create alerts table if it doesn't exist
                    await db.execute('''
                        CREATE TABLE IF NOT EXISTS alerts (
                            alert_id TEXT PRIMARY KEY,
                            rule_id TEXT NOT NULL,
                            metric_name TEXT NOT NULL,
                            metric_value TEXT NOT NULL,
                            severity TEXT NOT NULL,
                            message TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            acknowledged INTEGER NOT NULL DEFAULT 0,
                            acknowledged_by TEXT,
                            acknowledged_at TEXT
                        )
                    ''')
                    await db.commit()
            
            self.initialized = True
            self.logger.info(f"Database alert history initialized: {self.db_path}")
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
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute('''
                        INSERT INTO alerts (
                            alert_id, rule_id, metric_name, metric_value, severity,
                            message, timestamp, acknowledged, acknowledged_by, acknowledged_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        alert.alert_id,
                        alert.rule_id,
                        alert.metric_name,
                        json.dumps(alert.metric_value),
                        alert.severity.value,
                        alert.message,
                        alert.timestamp.isoformat(),
                        1 if alert.acknowledged else 0,
                        alert.acknowledged_by,
                        alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
                    ))
                    await db.commit()
                    
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
        
        # Apply time filters
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        # Apply additional filters
        if filters:
            if "severity" in filters:
                query += " AND severity = ?"
                params.append(filters["severity"])
            if "acknowledged" in filters:
                query += " AND acknowledged = ?"
                params.append(1 if filters["acknowledged"] else 0)
            if "rule_id" in filters:
                query += " AND rule_id = ?"
                params.append(filters["rule_id"])
            if "metric_name" in filters:
                query += " AND metric_name = ?"
                params.append(filters["metric_name"])
        
        # Order by timestamp (newest first)
        query += " ORDER BY timestamp DESC"
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, params) as cursor:
                    async for row in cursor:
                        try:
                            # Convert row to Alert object
                            alert = Alert(
                                alert_id=row["alert_id"],
                                rule_id=row["rule_id"],
                                metric_name=row["metric_name"],
                                metric_value=json.loads(row["metric_value"]),
                                severity=row["severity"],
                                message=row["message"],
                                timestamp=datetime.fromisoformat(row["timestamp"]),
                                acknowledged=bool(row["acknowledged"]),
                                acknowledged_by=row["acknowledged_by"],
                                acknowledged_at=datetime.fromisoformat(row["acknowledged_at"]) if row["acknowledged_at"] else None
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
                
                for key, value in updates.items():
                    if key == "acknowledged":
                        set_clauses.append("acknowledged = ?")
                        params.append(1 if value else 0)
                    elif key == "acknowledged_by":
                        set_clauses.append("acknowledged_by = ?")
                        params.append(value)
                    elif key == "acknowledged_at":
                        set_clauses.append("acknowledged_at = ?")
                        params.append(value.isoformat() if value else None)
                    # Add other fields as needed
                
                if not set_clauses:
                    return False
                
                # Add alert_id to params
                params.append(alert_id)
                
                # Execute update
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute(
                        f"UPDATE alerts SET {', '.join(set_clauses)} WHERE alert_id = ?",
                        params
                    )
                    await db.commit()
                    
                    if cursor.rowcount > 0:
                        self.logger.debug(f"Updated alert in database: {alert_id}")
                        return True
                    else:
                        self.logger.warning(f"Alert not found for update: {alert_id}")
                        return False
                
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
    db_path = config.get("db_path", "alerts.db")
    storage = DatabaseAlertHistory(db_path=db_path)
    asyncio.create_task(storage.initialize())
    return storage