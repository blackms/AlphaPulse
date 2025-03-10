"""
Repository classes for database access.

This module provides repository classes for accessing database entities.
"""
import json
import logging
from typing import List, Dict, Any, Optional, TypeVar, Generic, Type, Union
from datetime import datetime

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete, func, and_, or_

from .connection import get_pg_connection, _execute_with_retry as execute_with_retry
from .models import (
    User, ApiKey, Portfolio, Position, Trade, Alert, Metric, BaseModel
)

# Configure logging
logger = logging.getLogger('db_repository')

# Type variable for generic repository
T = TypeVar('T', bound=BaseModel)


class Repository(Generic[T]):
    """
    Generic repository for database entities.
    
    This class provides common CRUD operations for database entities.
    """
    
    def __init__(self, model_class: Type[T]):
        """Initialize the repository with a model class."""
        self.model_class = model_class
        self.table_name = f"alphapulse.{model_class.__tablename__}"
    
    async def find_by_id(self, id: int) -> Optional[Dict[str, Any]]:
        """Find an entity by ID."""
        async with get_pg_connection() as conn:
            query = f"SELECT * FROM {self.table_name} WHERE id = $1"
            row = await conn.fetchrow(query, id)
            return dict(row) if row else None
    
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Find all entities with pagination."""
        async with get_pg_connection() as conn:
            query = f"SELECT * FROM {self.table_name} ORDER BY id LIMIT $1 OFFSET $2"
            rows = await conn.fetch(query, limit, offset)
            return [dict(row) for row in rows]
    
    async def find_by_criteria(self, criteria: Dict[str, Any], limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Find entities by criteria."""
        if not criteria:
            return await self.find_all(limit, offset)
        
        # Build WHERE clause
        where_clauses = []
        params = []
        param_index = 1
        
        for key, value in criteria.items():
            where_clauses.append(f"{key} = ${param_index}")
            params.append(value)
            param_index += 1
        
        where_clause = " AND ".join(where_clauses)
        
        async with get_pg_connection() as conn:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE {where_clause}
                ORDER BY id
                LIMIT ${param_index} OFFSET ${param_index + 1}
            """
            params.extend([limit, offset])
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def count(self, criteria: Dict[str, Any] = None) -> int:
        """Count entities, optionally filtered by criteria."""
        if not criteria:
            async with get_pg_connection() as conn:
                query = f"SELECT COUNT(*) FROM {self.table_name}"
                result = await conn.fetchval(query)
                return result
        
        # Build WHERE clause
        where_clauses = []
        params = []
        param_index = 1
        
        for key, value in criteria.items():
            where_clauses.append(f"{key} = ${param_index}")
            params.append(value)
            param_index += 1
        
        where_clause = " AND ".join(where_clauses)
        
        async with get_pg_connection() as conn:
            query = f"SELECT COUNT(*) FROM {self.table_name} WHERE {where_clause}"
            result = await conn.fetchval(query, *params)
            return result
    
    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new entity."""
        # Remove any None values
        data = {k: v for k, v in data.items() if v is not None}
        
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f"${i+1}" for i in range(len(data)))
        values = list(data.values())
        
        async with get_pg_connection() as conn:
            query = f"""
                INSERT INTO {self.table_name} ({columns})
                VALUES ({placeholders})
                RETURNING *
            """
            row = await conn.fetchrow(query, *values)
            return dict(row)
    
    async def update(self, id: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an entity by ID."""
        # Remove any None values
        data = {k: v for k, v in data.items() if v is not None}
        
        if not data:
            return await self.find_by_id(id)
        
        # Add updated_at timestamp
        if 'updated_at' not in data:
            data['updated_at'] = datetime.now()
        
        set_clauses = []
        values = []
        param_index = 1
        
        for key, value in data.items():
            set_clauses.append(f"{key} = ${param_index}")
            values.append(value)
            param_index += 1
        
        set_clause = ", ".join(set_clauses)
        values.append(id)  # For the WHERE clause
        
        async with get_pg_connection() as conn:
            query = f"""
                UPDATE {self.table_name}
                SET {set_clause}
                WHERE id = ${param_index}
                RETURNING *
            """
            row = await conn.fetchrow(query, *values)
            return dict(row) if row else None
    
    async def delete(self, id: int) -> bool:
        """Delete an entity by ID."""
        async with get_pg_connection() as conn:
            query = f"DELETE FROM {self.table_name} WHERE id = $1 RETURNING id"
            row = await conn.fetchrow(query, id)
            return row is not None
    
    async def delete_by_criteria(self, criteria: Dict[str, Any]) -> int:
        """Delete entities by criteria."""
        if not criteria:
            return 0
        
        # Build WHERE clause
        where_clauses = []
        params = []
        param_index = 1
        
        for key, value in criteria.items():
            where_clauses.append(f"{key} = ${param_index}")
            params.append(value)
            param_index += 1
        
        where_clause = " AND ".join(where_clauses)
        
        async with get_pg_connection() as conn:
            query = f"""
                DELETE FROM {self.table_name}
                WHERE {where_clause}
                RETURNING id
            """
            rows = await conn.fetch(query, *params)
            return len(rows)


class UserRepository(Repository[User]):
    """Repository for User entities."""
    
    def __init__(self):
        """Initialize the repository."""
        super().__init__(User)
    
    async def find_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Find a user by username."""
        async with get_pg_connection() as conn:
            query = f"SELECT * FROM {self.table_name} WHERE username = $1"
            row = await conn.fetchrow(query, username)
            return dict(row) if row else None
    
    async def find_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Find a user by email."""
        async with get_pg_connection() as conn:
            query = f"SELECT * FROM {self.table_name} WHERE email = $1"
            row = await conn.fetchrow(query, email)
            return dict(row) if row else None
    
    async def update_last_login(self, user_id: int) -> None:
        """Update the last login timestamp for a user."""
        async with get_pg_connection() as conn:
            query = f"""
                UPDATE {self.table_name}
                SET last_login = now()
                WHERE id = $1
            """
            await conn.execute(query, user_id)


class ApiKeyRepository(Repository[ApiKey]):
    """Repository for ApiKey entities."""
    
    def __init__(self):
        """Initialize the repository."""
        super().__init__(ApiKey)
    
    async def find_by_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Find an API key by its value."""
        async with get_pg_connection() as conn:
            query = f"""
                SELECT ak.*, u.username, u.role
                FROM {self.table_name} ak
                JOIN alphapulse.users u ON ak.user_id = u.id
                WHERE ak.api_key = $1
            """
            row = await conn.fetchrow(query, api_key)
            return dict(row) if row else None
    
    async def find_by_user_id(self, user_id: int) -> List[Dict[str, Any]]:
        """Find all API keys for a user."""
        async with get_pg_connection() as conn:
            query = f"SELECT * FROM {self.table_name} WHERE user_id = $1"
            rows = await conn.fetch(query, user_id)
            return [dict(row) for row in rows]
    
    async def update_last_used(self, api_key_id: int) -> None:
        """Update the last used timestamp for an API key."""
        async with get_pg_connection() as conn:
            query = f"""
                UPDATE {self.table_name}
                SET last_used = now()
                WHERE id = $1
            """
            await conn.execute(query, api_key_id)


class PortfolioRepository(Repository[Portfolio]):
    """Repository for Portfolio entities."""
    
    def __init__(self):
        """Initialize the repository."""
        super().__init__(Portfolio)
    
    async def find_with_positions(self, portfolio_id: int) -> Optional[Dict[str, Any]]:
        """Find a portfolio with its positions."""
        async with get_pg_connection() as conn:
            # Get portfolio
            portfolio_query = f"SELECT * FROM {self.table_name} WHERE id = $1"
            portfolio_row = await conn.fetchrow(portfolio_query, portfolio_id)
            
            if not portfolio_row:
                return None
            
            portfolio = dict(portfolio_row)
            
            # Get positions
            positions_query = """
                SELECT * FROM alphapulse.positions
                WHERE portfolio_id = $1
            """
            position_rows = await conn.fetch(positions_query, portfolio_id)
            portfolio['positions'] = [dict(row) for row in position_rows]
            
            return portfolio
    
    async def find_all_with_positions(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Find all portfolios with their positions."""
        async with get_pg_connection() as conn:
            # Get portfolios
            portfolios_query = f"""
                SELECT * FROM {self.table_name}
                ORDER BY id
                LIMIT $1 OFFSET $2
            """
            portfolio_rows = await conn.fetch(portfolios_query, limit, offset)
            
            if not portfolio_rows:
                return []
            
            portfolios = [dict(row) for row in portfolio_rows]
            
            # Get positions for all portfolios
            portfolio_ids = [p['id'] for p in portfolios]
            positions_query = """
                SELECT * FROM alphapulse.positions
                WHERE portfolio_id = ANY($1::int[])
            """
            position_rows = await conn.fetch(positions_query, portfolio_ids)
            
            # Group positions by portfolio_id
            positions_by_portfolio = {}
            for row in position_rows:
                position = dict(row)
                portfolio_id = position['portfolio_id']
                if portfolio_id not in positions_by_portfolio:
                    positions_by_portfolio[portfolio_id] = []
                positions_by_portfolio[portfolio_id].append(position)
            
            # Add positions to portfolios
            for portfolio in portfolios:
                portfolio['positions'] = positions_by_portfolio.get(portfolio['id'], [])
            
            return portfolios


class PositionRepository(Repository[Position]):
    """Repository for Position entities."""
    
    def __init__(self):
        """Initialize the repository."""
        super().__init__(Position)
    
    async def find_by_portfolio_and_symbol(self, portfolio_id: int, symbol: str) -> Optional[Dict[str, Any]]:
        """Find a position by portfolio ID and symbol."""
        async with get_pg_connection() as conn:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE portfolio_id = $1 AND symbol = $2
            """
            row = await conn.fetchrow(query, portfolio_id, symbol)
            return dict(row) if row else None
    
    async def update_current_price(self, position_id: int, current_price: float) -> Optional[Dict[str, Any]]:
        """Update the current price of a position."""
        async with get_pg_connection() as conn:
            query = f"""
                UPDATE {self.table_name}
                SET current_price = $1, updated_at = now()
                WHERE id = $2
                RETURNING *
            """
            row = await conn.fetchrow(query, current_price, position_id)
            return dict(row) if row else None
    
    async def update_current_prices(self, symbol_prices: Dict[str, float]) -> int:
        """Update current prices for multiple positions by symbol."""
        if not symbol_prices:
            return 0
        
        # Build case statement for the update
        case_statements = []
        symbols = []
        
        for symbol, price in symbol_prices.items():
            case_statements.append(f"WHEN symbol = ${len(symbols) + 1} THEN ${len(symbols) + 2}::numeric")
            symbols.extend([symbol, price])
        
        case_statement = " ".join(case_statements)
        
        async with get_pg_connection() as conn:
            query = f"""
                UPDATE {self.table_name}
                SET current_price = CASE {case_statement} END,
                    updated_at = now()
                WHERE symbol = ANY(${ len(symbols) + 1 }::varchar[])
                RETURNING id
            """
            symbol_list = [s for s, _ in symbol_prices.items()]
            rows = await conn.fetch(query, *symbols, symbol_list)
            return len(rows)


class TradeRepository(Repository[Trade]):
    """Repository for Trade entities."""
    
    def __init__(self):
        """Initialize the repository."""
        super().__init__(Trade)
    
    async def find_by_portfolio(self, portfolio_id: int, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Find trades by portfolio ID."""
        async with get_pg_connection() as conn:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE portfolio_id = $1
                ORDER BY executed_at DESC
                LIMIT $2 OFFSET $3
            """
            rows = await conn.fetch(query, portfolio_id, limit, offset)
            return [dict(row) for row in rows]
    
    async def find_by_symbol(self, symbol: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Find trades by symbol."""
        async with get_pg_connection() as conn:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE symbol = $1
                ORDER BY executed_at DESC
                LIMIT $2 OFFSET $3
            """
            rows = await conn.fetch(query, symbol, limit, offset)
            return [dict(row) for row in rows]
    
    async def find_by_time_range(self, start_time: datetime, end_time: datetime, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Find trades within a time range."""
        async with get_pg_connection() as conn:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE executed_at BETWEEN $1 AND $2
                ORDER BY executed_at DESC
                LIMIT $3 OFFSET $4
            """
            rows = await conn.fetch(query, start_time, end_time, limit, offset)
            return [dict(row) for row in rows]


class AlertRepository(Repository[Alert]):
    """Repository for Alert entities."""
    
    def __init__(self):
        """Initialize the repository."""
        super().__init__(Alert)
    
    async def find_unacknowledged(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Find unacknowledged alerts."""
        async with get_pg_connection() as conn:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE acknowledged = false
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
            """
            rows = await conn.fetch(query, limit, offset)
            return [dict(row) for row in rows]
    
    async def find_by_severity(self, severity: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Find alerts by severity."""
        async with get_pg_connection() as conn:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE severity = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
            """
            rows = await conn.fetch(query, severity, limit, offset)
            return [dict(row) for row in rows]
    
    async def acknowledge(self, alert_id: int, acknowledged_by: str) -> Optional[Dict[str, Any]]:
        """Acknowledge an alert."""
        async with get_pg_connection() as conn:
            query = f"""
                UPDATE {self.table_name}
                SET acknowledged = true,
                    acknowledged_by = $1,
                    acknowledged_at = now()
                WHERE id = $2 AND acknowledged = false
                RETURNING *
            """
            row = await conn.fetchrow(query, acknowledged_by, alert_id)
            return dict(row) if row else None


class MetricRepository:
    """Repository for time-series metrics."""
    
    def __init__(self):
        """Initialize the repository."""
        self.table_name = "alphapulse.metrics"
    
    async def insert(self, metric: Metric) -> None:
        """Insert a new metric."""
        async with get_pg_connection() as conn:
            query = f"""
                INSERT INTO {self.table_name} (time, metric_name, value, labels)
                VALUES ($1, $2, $3, $4)
            """
            await conn.execute(
                query,
                metric.timestamp,
                metric.metric_name,
                metric.value,
                json.dumps(metric.labels)
            )
    
    async def insert_batch(self, metrics: List[Metric]) -> None:
        """Insert multiple metrics in a batch."""
        if not metrics:
            return
        
        async with get_pg_connection() as conn:
            # Prepare values for batch insert
            values = []
            for metric in metrics:
                values.append((
                    metric.timestamp,
                    metric.metric_name,
                    metric.value,
                    json.dumps(metric.labels)
                ))
            
            # Use copy_records_to_table for efficient batch insert
            await conn.copy_records_to_table(
                self.table_name,
                records=values,
                columns=["time", "metric_name", "value", "labels"]
            )
    
    async def find_by_name(
        self, 
        metric_name: str, 
        start_time: datetime, 
        end_time: datetime,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Find metrics by name within a time range."""
        async with get_pg_connection() as conn:
            query = f"""
                SELECT time, value, labels
                FROM {self.table_name}
                WHERE metric_name = $1 AND time BETWEEN $2 AND $3
                ORDER BY time DESC
                LIMIT $4
            """
            rows = await conn.fetch(query, metric_name, start_time, end_time, limit)
            return [
                {
                    'timestamp': row['time'],
                    'value': row['value'],
                    'labels': row['labels']
                }
                for row in rows
            ]
    
    async def find_latest_by_name(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Find the latest metric by name."""
        async with get_pg_connection() as conn:
            query = f"""
                SELECT time, value, labels
                FROM {self.table_name}
                WHERE metric_name = $1
                ORDER BY time DESC
                LIMIT 1
            """
            row = await conn.fetchrow(query, metric_name)
            if row:
                return {
                    'timestamp': row['time'],
                    'value': row['value'],
                    'labels': row['labels']
                }
            return None
    
    async def aggregate_by_time(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        interval: str,
        aggregation: str = 'avg'
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics by time interval."""
        valid_aggregations = ['avg', 'min', 'max', 'sum', 'count']
        if aggregation not in valid_aggregations:
            raise ValueError(f"Invalid aggregation: {aggregation}. Must be one of {valid_aggregations}")
        
        async with get_pg_connection() as conn:
            # Use date_trunc for time bucketing instead of TimescaleDB's time_bucket
            query = f"""
                SELECT
                    date_trunc($1, time) AS bucket,
                    {aggregation}(value) AS value
                FROM {self.table_name}
                WHERE metric_name = $2 AND time BETWEEN $3 AND $4
                GROUP BY bucket
                ORDER BY bucket
            """
            rows = await conn.fetch(query, interval, metric_name, start_time, end_time)
            return [
                {
                    'timestamp': row['bucket'],
                    'value': row['value']
                }
                for row in rows
            ]