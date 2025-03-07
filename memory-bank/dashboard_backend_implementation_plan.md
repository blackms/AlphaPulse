**File**: `src/alpha_pulse/api/data/portfolio.py`
```python
"""Portfolio data access module."""
from typing import Dict, Optional
import logging

from alpha_pulse.portfolio.portfolio_manager import PortfolioManager


class PortfolioDataAccessor:
    """Access portfolio data."""
    
    def __init__(self):
        """Initialize portfolio accessor."""
        self.logger = logging.getLogger("alpha_pulse.api.data.portfolio")
        self.portfolio_manager = PortfolioManager.get_instance()
    
    async def get_portfolio(self, include_history: bool = False) -> Dict:
        """
        Get current portfolio data.
        
        Args:
            include_history: Whether to include historical data
            
        Returns:
            Portfolio data
        """
        try:
            # Get current portfolio
            portfolio = self.portfolio_manager.get_portfolio_data()
            
            # Transform to API format
            result = {
                "total_value": portfolio.total_value,
                "cash": portfolio.cash,
                "positions": []
            }
            
            # Add positions
            for position in portfolio.positions:
                result["positions"].append({
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "entry_price": position.entry_price,
                    "current_price": position.current_price,
                    "value": position.value,
                    "pnl": position.pnl,
                    "pnl_percentage": position.pnl_percentage
                })
                
            # Add performance metrics
            result["metrics"] = {
                "sharpe_ratio": portfolio.metrics.sharpe_ratio,
                "sortino_ratio": portfolio.metrics.sortino_ratio,
                "max_drawdown": portfolio.metrics.max_drawdown,
                "volatility": portfolio.metrics.volatility,
                "return_since_inception": portfolio.metrics.return_since_inception
            }
            
            # Add historical data if requested
            if include_history:
                history = self.portfolio_manager.get_portfolio_history()
                result["history"] = []
                
                for entry in history:
                    result["history"].append({
                        "timestamp": entry.timestamp.isoformat(),
                        "total_value": entry.total_value,
                        "cash": entry.cash,
                        "positions_value": entry.positions_value
                    })
            
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving portfolio data: {str(e)}")
            return {
                "total_value": 0,
                "cash": 0,
                "positions": [],
                "metrics": {},
                "error": str(e)
            }
```

**File**: `src/alpha_pulse/api/data/trades.py`
```python
"""Trade data access module."""
from typing import Dict, List, Optional
from datetime import datetime
import logging

from alpha_pulse.execution.broker_interface import BrokerInterface


class TradeDataAccessor:
    """Access trade data."""
    
    def __init__(self):
        """Initialize trade accessor."""
        self.logger = logging.getLogger("alpha_pulse.api.data.trades")
        self.broker = BrokerInterface.get_instance()
    
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get trade history.
        
        Args:
            symbol: Filter by symbol
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of trade data
        """
        try:
            # Get trade history from broker
            trades = await self.broker.get_trade_history(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time
            )
            
            # Transform to API format
            result = []
            for trade in trades:
                result.append({
                    "id": trade.id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "timestamp": trade.timestamp.isoformat(),
                    "status": trade.status,
                    "order_type": trade.order_type,
                    "fees": trade.fees
                })
                
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving trade history: {str(e)}")
            return []
```

**File**: `src/alpha_pulse/api/data/system.py`
```python
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
```

### Phase 3: REST API Endpoints (Day 3)

**Objective**: Implement the REST API endpoints for all data sources.

#### Step 3.1: Implement Router Modules

**File**: `src/alpha_pulse/api/routers/__init__.py`
```python
"""API routers."""
```

**File**: `src/alpha_pulse/api/routers/metrics.py`
```python
"""Metrics router."""
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException

from ..dependencies import get_current_user, has_permission
from ..data import MetricsDataAccessor
from ..cache import get_cache

router = APIRouter()
metrics_accessor = MetricsDataAccessor()


@router.get("/{metric_type}", response_model=List[Dict])
async def get_metrics(
    metric_type: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    aggregation: Optional[str] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get metrics data.
    
    Args:
        metric_type: Type of metric to retrieve
        start_time: Start time for query
        end_time: End time for query
        aggregation: Aggregation method (e.g., "mean", "sum")
        
    Returns:
        List of metric data points
    """
    # Check permissions
    if not has_permission(current_user, "view_metrics"):
        raise HTTPException(status_code=403, detail="Not authorized to view metrics")
    
    # Use cache for common queries
    cache = await get_cache()
    cache_key = f"metrics:{metric_type}:{start_time}:{end_time}:{aggregation}"
    
    # Try to get from cache
    cached_data = await cache.get(cache_key)
    if cached_data:
        return cached_data
    
    # Get from data accessor
    data = await metrics_accessor.get_metrics(
        metric_type=metric_type,
        start_time=start_time,
        end_time=end_time,
        aggregation=aggregation
    )
    
    # Cache for future requests
    await cache.set(cache_key, data, expiry=300)  # 5 minutes
    
    return data


@router.get("/{metric_type}/latest", response_model=Dict)
async def get_latest_metrics(
    metric_type: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get latest metrics of a specific type.
    
    Args:
        metric_type: Type of metric to retrieve
        
    Returns:
        Dictionary of latest metrics
    """
    # Check permissions
    if not has_permission(current_user, "view_metrics"):
        raise HTTPException(status_code=403, detail="Not authorized to view metrics")
    
    # Use cache with short TTL for latest metrics
    cache = await get_cache()
    cache_key = f"metrics:{metric_type}:latest"
    
    # Try to get from cache
    cached_data = await cache.get(cache_key)
    if cached_data:
        return cached_data
    
    # Get from data accessor
    data = await metrics_accessor.get_latest_metrics(metric_type)
    
    # Cache for future requests (short TTL for latest data)
    await cache.set(cache_key, data, expiry=60)  # 1 minute
    
    return data
```

**File**: `src/alpha_pulse/api/routers/alerts.py`
```python
"""Alerts router."""
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException

from ..dependencies import get_current_user, has_permission
from ..data import AlertDataAccessor

router = APIRouter()
alert_accessor = AlertDataAccessor()


@router.get("/", response_model=List[Dict])
async def get_alerts(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    severity: Optional[str] = Query(None),
    acknowledged: Optional[bool] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get alert history.
    
    Args:
        start_time: Filter alerts after this time
        end_time: Filter alerts before this time
        severity: Filter by severity
        acknowledged: Filter by acknowledgment status
        
    Returns:
        List of alert data
    """
    # Check permissions
    if not has_permission(current_user, "view_alerts"):
        raise HTTPException(status_code=403, detail="Not authorized to view alerts")
    
    # Build filters
    filters = {}
    if severity:
        filters["severity"] = severity
    if acknowledged is not None:
        filters["acknowledged"] = acknowledged
    
    # Get from data accessor
    return await alert_accessor.get_alerts(
        start_time=start_time,
        end_time=end_time,
        filters=filters
    )


@router.post("/{alert_id}/acknowledge", response_model=Dict)
async def acknowledge_alert(
    alert_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Acknowledge an alert.
    
    Args:
        alert_id: ID of the alert to acknowledge
        
    Returns:
        Updated alert data
    """
    # Check permissions
    if not has_permission(current_user, "acknowledge_alerts"):
        raise HTTPException(status_code=403, detail="Not authorized to acknowledge alerts")
    
    # Acknowledge alert
    result = await alert_accessor.acknowledge_alert(
        alert_id=alert_id,
        user=current_user["username"]
    )
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result
```

**File**: `src/alpha_pulse/api/routers/portfolio.py`
```python
"""Portfolio router."""
from typing import Dict
from fastapi import APIRouter, Depends, Query, HTTPException

from ..dependencies import get_current_user, has_permission
from ..data import PortfolioDataAccessor

router = APIRouter()
portfolio_accessor = PortfolioDataAccessor()


@router.get("/", response_model=Dict)
async def get_portfolio(
    include_history: bool = Query(False),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get current portfolio data.
    
    Args:
        include_history: Whether to include historical data
        
    Returns:
        Portfolio data
    """
    # Check permissions
    if not has_permission(current_user, "view_portfolio"):
        raise HTTPException(status_code=403, detail="Not authorized to view portfolio")
    
    # Get from data accessor
    return await portfolio_accessor.get_portfolio(include_history=include_history)
```

**File**: `src/alpha_pulse/api/routers/trades.py`
```python
"""Trades router."""
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException

from ..dependencies import get_current_user, has_permission
from ..data import TradeDataAccessor

router = APIRouter()
trade_accessor = TradeDataAccessor()


@router.get("/", response_model=List[Dict])
async def get_trades(
    symbol: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get trade history.
    
    Args:
        symbol: Filter by symbol
        start_time: Filter by start time
        end_time: Filter by end time
        
    Returns:
        List of trade data
    """
    # Check permissions
    if not has_permission(current_user, "view_trades"):
        raise HTTPException(status_code=403, detail="Not authorized to view trades")
    
    # Get from data accessor
    return await trade_accessor.get_trades(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time
    )
```

**File**: `src/alpha_pulse/api/routers/system.py`
```python
"""System router."""
from typing import Dict
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_current_user, has_permission
from ..data import SystemDataAccessor

router = APIRouter()
system_accessor = SystemDataAccessor()


@router.get("/", response_model=Dict)
async def get_system_metrics(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get current system metrics.
    
    Returns:
        System metrics
    """
    # Check permissions
    if not has_permission(current_user, "view_system"):
        raise HTTPException(status_code=403, detail="Not authorized to view system metrics")
    
    # Get from data accessor
    return await system_accessor.get_system_metrics()
```

#### Step 3.2: Update Main Application

**Update File**: `src/alpha_pulse/api/main.py`
```python
# Import routers
from .routers import metrics, portfolio, alerts, trades, system

# Add routers to API router
api_router.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
api_router.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["alerts"])
api_router.include_router(trades.router, prefix="/trades", tags=["trades"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
```

### Phase 4: WebSocket Implementation (Day 4-5)

**Objective**: Implement WebSocket server for real-time updates.

#### Step 4.1: WebSocket Manager Module

**File**: `src/alpha_pulse/api/websockets/__init__.py`
```python
"""WebSocket module."""
```

**File**: `src/alpha_pulse/api/websockets/manager.py`
```python
"""WebSocket connection manager."""
from typing import Dict, List, Set
import logging
from fastapi import WebSocket
import json
import asyncio


class ConnectionManager:
    """Manage WebSocket connections."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ConnectionManager()
        return cls._instance
    
    def __init__(self):
        """Initialize connection manager."""
        self.logger = logging.getLogger("alpha_pulse.api.websockets")
        self.active_connections: Dict[str, List[WebSocket]] = {
            "metrics": [],
            "alerts": [],
            "trades": [],
            "portfolio": []
        }
        self.client_subscriptions: Dict[WebSocket, Set[str]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Connect a WebSocket client."""
        await websocket.accept()
        self.client_subscriptions[websocket] = set()
        self.logger.info(f"Client {client_id} connected")
        
    async def disconnect(self, websocket: WebSocket, client_id: str) -> None:
        """Disconnect a WebSocket client."""
        # Remove from all channels
        for channel in self.active_connections.values():
            if websocket in channel:
                channel.remove(websocket)
                
        # Remove subscriptions
        if websocket in self.client_subscriptions:
            del self.client_subscriptions[websocket]
            
        self.logger.info(f"Client {client_id} disconnected")
        
    async def subscribe(self, websocket: WebSocket, channel: str) -> None:
        """Subscribe client to a channel."""
        if channel in self.active_connections:
            if websocket not in self.active_connections[channel]:
                self.active_connections[channel].append(websocket)
                self.client_subscriptions[websocket].add(channel)
                
    async def unsubscribe(self, websocket: WebSocket, channel: str) -> None:
        """Unsubscribe client from a channel."""
        if channel in self.active_connections:
            if websocket in self.active_connections[channel]:
                self.active_connections[channel].remove(websocket)
                
            if websocket in self.client_subscriptions and channel in self.client_subscriptions[websocket]:
                self.client_subscriptions[websocket].remove(channel)
                
    async def broadcast(self, channel: str, message: Dict) -> None:
        """Broadcast message to all subscribed clients."""
        if channel not in self.active_connections:
            return
            
        disconnected = []
        for websocket in self.active_connections[channel]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                self.logger.error(f"Error sending message: {str(e)}")
                disconnected.append(websocket)
                
        # Clean up disconnected clients
        for websocket in disconnected:
            for channel in self.active_connections:
                if websocket in self.active_connections[channel]:
                    self.active_connections[channel].remove(websocket)
                    
            if websocket in self.client_subscriptions:
                del self.client_subscriptions[websocket]
```

#### Step 4.2: WebSocket Auth and Subscription Modules

**File**: `src/alpha_pulse/api/websockets/auth.py`
```python
"""WebSocket authentication."""
from typing import Dict, Optional
import json
import logging
from fastapi import WebSocket

from ..dependencies import authenticate_token, has_permission


class WebSocketAuthenticator:
    """Authenticate WebSocket connections."""
    
    def __init__(self):
        """Initialize WebSocket authenticator."""
        self.logger = logging.getLogger("alpha_pulse.api.websockets.auth")
    
    async def authenticate(self, websocket: WebSocket) -> Optional[Dict]:
        """
        Authenticate WebSocket client.
        
        Args:
            websocket: WebSocket connection
            
        Returns:
            User data if authenticated, None otherwise
        """
        try:
            # Get authentication message
            auth_message = await websocket.receive_json()
            
            # Check for token
            token = auth_message.get("token")
            if not token:
                await websocket.send_json({"error": "Missing authentication token"})
                return None
                
            # Authenticate token
            user = await authenticate_token(token)
            if not user:
                await websocket.send_json({"error": "Invalid authentication token"})
                return None
                
            # Send success message
            await websocket.send_json({"message": "Authenticated successfully"})
            return user
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            try:
                await websocket.send_json({"error": "Authentication error"})
            except:
                pass
            return None
```

**File**: `src/alpha_pulse/api/websockets/subscription.py`
```python
"""WebSocket subscription management."""
from typing import Dict, List, Optional, Set
import asyncio
import logging
from datetime import datetime

from .manager import ConnectionManager
from alpha_pulse.monitoring.metrics_calculations import calculate_derived_metrics


class SubscriptionManager:
    """Manage subscriptions and updates."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = SubscriptionManager()
        return cls._instance
    
    def __init__(self):
        """Initialize subscription manager."""
        self.logger = logging.getLogger("alpha_pulse.api.websockets.subscription")
        self.connection_manager = ConnectionManager.get_instance()
        self.running = False
        self.update_tasks = []
        
    async def start(self) -> None:
        """Start subscription manager."""
        if self.running:
            return
            
        self.running = True
        
        # Start update tasks
        self.update_tasks = [
            asyncio.create_task(self._update_metrics()),
            asyncio.create_task(self._update_portfolio()),
            asyncio.create_task(self._listen_for_alerts()),
            asyncio.create_task(self._listen_for_trades())
        ]
        
        self.logger.info("Subscription manager started")
        
    async def stop(self) -> None:
        """Stop subscription manager."""
        if not self.running:
            return
            
        self.running = False
        
        # Cancel all tasks
        for task in self.update_tasks:
            task.cancel()
            
        self.update_tasks = []
        self.logger.info("Subscription manager stopped")
    
    async def _update_metrics(self) -> None:
        """Update metrics periodically."""
        from alpha_pulse.monitoring.collector import MetricsCollector
        
        collector = MetricsCollector.get_instance()
        
        try:
            while self.running:
                # Get latest metrics
                latest_metrics = await collector.collect_latest_metrics()
                
                # Add derived metrics
                derived = calculate_derived_metrics(latest_metrics)
                
                # Create update message
                message = {
                    "type": "metrics",
                    "timestamp": datetime.now().isoformat(),
                    "data": {}
                }
                
                # Add metrics to message
                for metric in latest_metrics:
                    message["data"][metric.name] = {
                        "value": metric.value,
                        "timestamp": metric.timestamp.isoformat(),
                        "labels": metric.labels
                    }
                    
                # Add derived metrics
                for name, value in derived.items():
                    message["data"][name] = {
                        "value": value,
                        "timestamp": datetime.now().isoformat(),
                        "labels": {"derived": "true"}
                    }
                
                # Broadcast to subscribers
                await self.connection_manager.broadcast("metrics", message)
                
                # Wait for next update
                await asyncio.sleep(5)  # Update every 5 seconds
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
            
    async def _update_portfolio(self) -> None:
        """Update portfolio periodically."""
        from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
        
        portfolio_manager = PortfolioManager.get_instance()
        
        try:
            while self.running:
                # Get current portfolio
                portfolio = portfolio_manager.get_portfolio_data()
                
                # Create update message
                message = {
                    "type": "portfolio",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "total_value": portfolio.total_value,
                        "cash": portfolio.cash,
                        "positions": []
                    }
                }
                
                # Add positions
                for position in portfolio.positions:
                    message["data"]["positions"].append({
                        "symbol": position.symbol,
                        "quantity": position.quantity,
                        "entry_price": position.entry_price,
                        "current_price": position.current_price,
                        "value": position.value,
                        "pnl": position.pnl,
                        "pnl_percentage": position.pnl_percentage
                    })
                
                # Broadcast to subscribers
                await self.connection_manager.broadcast("portfolio", message)
                
                # Wait for next update
                await asyncio.sleep(10)  # Update every 10 seconds
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {str(e)}")
            
    async def _listen_for_alerts(self) -> None:
        """Listen for alert events."""
        from alpha_pulse.monitoring.alerting.manager import AlertManager
        
        alert_manager = AlertManager.get_instance()
        
        try:
            # Register callback for new alerts
            async def handle_alert(alert):
                # Create update message
                message = {
                    "type": "alert",
                    "timestamp": datetime.now().isoformat(),
                    "data": alert.to_dict()
                }
                
                # Broadcast to subscribers
                await self.connection_manager.broadcast("alerts", message)
            
            # Register callback
            alert_manager.register_alert_callback(handle_alert)
            
            # Keep task alive
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Task was cancelled
            alert_manager.unregister_alert_callback(handle_alert)
        except Exception as e:
            self.logger.error(f"Error listening for alerts: {str(e)}")
            
    async def _listen_for_trades(self) -> None:
        """Listen for trade events."""
        from alpha_pulse.execution.broker_interface import BrokerInterface
        
        broker = BrokerInterface.get_instance()
        
        try:
            # Register callback for new trades
            async def handle_trade(trade):
                # Create update message
                message = {
                    "type": "trade",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "id": trade.id,
                        "symbol": trade.symbol,
                        "side": trade.side,
                        "quantity": trade.quantity,
                        "price": trade.price,
                        "timestamp": trade.timestamp.isoformat(),
                        "status": trade.status,
                        "order_type": trade.order_type,
                        "fees": trade.fees
                    }
                }
                
                # Broadcast to subscribers
                await self.connection_manager.broadcast("trades", message)
            
            # Register callback
            broker.register_trade_callback(handle_trade)
            
            # Keep task alive
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Task was cancelled
            broker.unregister_trade_callback(handle_trade)
        except Exception as e:
            self.logger.error(f"Error listening for trades: {str(e)}")
```

#### Step 4.3: WebSocket Endpoints in Main Application

**Update File**: `src/alpha_pulse/api/main.py` with the following additions:

```python
# Import WebSocket dependencies
from fastapi import WebSocket, WebSocketDisconnect
from .websockets.manager import ConnectionManager
from .websockets.auth import WebSocketAuthenticator
from .websockets.subscription import SubscriptionManager

# Initialize websocket components
connection_manager = ConnectionManager.get_instance()
websocket_auth = WebSocketAuthenticator()
subscription_manager = SubscriptionManager.get_instance()

# Add startup and shutdown handlers
@app.on_event("startup")
async def startup_event():
    """Initialize API components on startup."""
    await subscription_manager.start()
    logger.info("API server started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up API components on shutdown."""
    await subscription_manager.stop()
    logger.info("API server stopped")

# WebSocket endpoints
@app.websocket("/ws/metrics")
async def metrics_websocket(websocket: WebSocket):
    """WebSocket for real-time metrics updates."""
    # Accept connection
    await websocket.accept()
    client_id = f"client-{id(websocket)}"
    
    try:
        # Authenticate user
        user = await websocket_auth.authenticate(websocket)
        if not user:
            await websocket.close(code=1008, reason="Authentication failed")
            return
            
        # Check permissions
        if not has_permission(user, "view_metrics"):
            await websocket.close(code=1008, reason="Not authorized")
            return
            
        # Register connection
        await connection_manager.connect(websocket, client_id)
        
        # Subscribe to metrics channel
        await connection_manager.subscribe(websocket, "metrics")
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Ping-pong for keepalive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        # Handle disconnect
        logger.info(f"Client {client_id} disconnected")
    finally:
        await connection_manager.disconnect(websocket, client_id)

@app.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket):
    """WebSocket for real-time alerts updates."""
    # Accept connection
    await websocket.accept()
    client_id = f"client-{id(websocket)}"
    
    try:
        # Authenticate user
        user = await websocket_auth.authenticate(websocket)
        if not user:
            await websocket.close(code=1008, reason="Authentication failed")
            return
            
        # Check permissions
        if not has_permission(user, "view_alerts"):
            await websocket.close(code=1008, reason="Not authorized")
            return
            
        # Register connection
        await connection_manager.connect(websocket, client_id)
        
        # Subscribe to alerts channel
        await connection_manager.subscribe(websocket, "alerts")
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Ping-pong for keepalive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        # Handle disconnect
        logger.info(f"Client {client_id} disconnected")
    finally:
        await connection_manager.disconnect(websocket, client_id)

@app.websocket("/ws/portfolio")
async def portfolio_websocket(websocket: WebSocket):
    """WebSocket for real-time portfolio updates."""
    # Accept connection
    await websocket.accept()
    client_id = f"client-{id(websocket)}"
    
    try:
        # Authenticate user
        user = await websocket_auth.authenticate(websocket)
        if not user:
            await websocket.close(code=1008, reason="Authentication failed")
            return
            
        # Check permissions
        if not has_permission(user, "view_portfolio"):
            await websocket.close(code=1008, reason="Not authorized")
            return
            
        # Register connection
        await connection_manager.connect(websocket, client_id)
        
        # Subscribe to portfolio channel
        await connection_manager.subscribe(websocket, "portfolio")
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Ping-pong for keepalive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        # Handle disconnect
        logger.info(f"Client {client_id} disconnected")
    finally:
        await connection_manager.disconnect(websocket, client_id)

@app.websocket("/ws/trades")
async def trades_websocket(websocket: WebSocket):
    """WebSocket for real-time trade updates."""
    # Accept connection
    await websocket.accept()
    client_id = f"client-{id(websocket)}"
    
    try:
        # Authenticate user
        user = await websocket_auth.authenticate(websocket)
        if not user:
            await websocket.close(code=1008, reason="Authentication failed")
            return
            
        # Check permissions
        if not has_permission(user, "view_trades"):
            await websocket.close(code=1008, reason="Not authorized")
            return
            
        # Register connection
        await connection_manager.connect(websocket, client_id)
        
        # Subscribe to trades channel
        await connection_manager.subscribe(websocket, "trades")
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Ping-pong for keepalive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        # Handle disconnect
        logger.info(f"Client {client_id} disconnected")
    finally:
        await connection_manager.disconnect(websocket, client_id)
```

### Phase 5: Testing and Launch Script (Day 6-7)

**Objective**: Implement tests and create launch script.

#### Step 5.1: Create Basic Tests

**File**: `src/alpha_pulse/tests/api/test_auth.py`
```python
"""Test authentication."""
import pytest
import jwt
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

from alpha_pulse.api.main import app
from alpha_pulse.api.dependencies import create_token


@pytest.fixture
def client():
    """Get test client."""
    return TestClient(app)


@pytest.fixture
def valid_token():
    """Get valid token."""
    return create_token(
        data={"sub": "testuser", "username": "testuser", "role": "admin"},
        expires_delta=timedelta(minutes=15)
    )


def test_missing_token(client):
    """Test request without token."""
    response = client.get("/api/metrics/performance")
    assert response.status_code == 401
    

def test_invalid_token(client):
    """Test request with invalid token."""
    response = client.get(
        "/api/metrics/performance",
        headers={"Authorization": "Bearer invalid-token"}
    )
    assert response.status_code == 401


def test_valid_token(client, valid_token):
    """Test request with valid token."""
    response = client.get(
        "/api/metrics/performance",
        headers={"Authorization": f"Bearer {valid_token}"}
    )
    # Since no actual data, expect 200 with empty list
    assert response.status_code == 200
    assert isinstance(response.json(), list)
```

**File**: `src/alpha_pulse/tests/api/test_endpoints.py`
```python
"""Test API endpoints."""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from alpha_pulse.api.main import app
from alpha_pulse.api.dependencies import create_token


@pytest.fixture
def client():
    """Get test client."""
    return TestClient(app)


@pytest.fixture
def valid_token():
    """Get valid token."""
    return create_token(
        data={"sub": "testuser", "username": "testuser", "role": "admin"},
        expires_delta=timedelta(minutes=15)
    )


@patch("alpha_pulse.api.data.metrics.MetricsDataAccessor.get_metrics")
def test_get_metrics(mock_get_metrics, client, valid_token):
    """Test GET /api/metrics/{metric_type}."""
    # Mock return value
    mock_get_metrics.return_value = [
        {
            "name": "portfolio_value",
            "value": 10000.0,
            "timestamp": datetime.now().isoformat(),
            "labels": {"portfolio": "main"}
        }
    ]
    
    # Test endpoint
    response = client.get(
        "/api/metrics/portfolio",
        headers={"Authorization": f"Bearer {valid_token}"}
    )
    
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["name"] == "portfolio_value"
    assert response.json()[0]["value"] == 10000.0


@patch("alpha_pulse.api.data.alerts.AlertDataAccessor.get_alerts")
def test_get_alerts(mock_get_alerts, client, valid_token):
    """Test GET /api/alerts."""
    # Mock return value
    mock_get_alerts.return_value = [
        {
            "alert_id": "test-alert-1",
            "rule_id": "test-rule-1",
            "metric_name": "drawdown",
            "metric_value": 0.15,
            "severity": "warning",
            "message": "Drawdown exceeds 10%",
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False,
            "acknowledged_by": None,
            "acknowledged_at": None
        }
    ]
    
    # Test endpoint
    response = client.get(
        "/api/alerts",
        headers={"Authorization": f"Bearer {valid_token}"}
    )
    
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["alert_id"] == "test-alert-1"
    assert response.json()[0]["severity"] == "warning"


@patch("alpha_pulse.api.data.portfolio.PortfolioDataAccessor.get_portfolio")
def test_get_portfolio(mock_get_portfolio, client, valid_token):
    """Test GET /api/portfolio."""
    # Mock return value
    mock_get_portfolio.return_value = {
        "total_value": 10000.0,
        "cash": 5000.0,
        "positions": [
            {
                "symbol": "BTC-USD",
                "quantity": 0.1,
                "entry_price": 50000.0,
                "current_price": 55000.0,
                "value": 5500.0,
                "pnl": 500.0,
                "pnl_percentage": 10.0
            }
        ],
        "metrics": {
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "max_drawdown": 0.05,
            "volatility": 0.2,
            "return_since_inception": 0.15
        }
    }
    
    # Test endpoint
    response = client.get(
        "/api/portfolio",
        headers={"Authorization": f"Bearer {valid_token}"}
    )
    
    assert response.status_code == 200
    assert response.json()["total_value"] == 10000.0
    assert response.json()["cash"] == 5000.0
    assert len(response.json()["positions"]) == 1
    assert response.json()["positions"][0]["symbol"] == "BTC-USD"
```

#### Step 5.2: Create Launch Script

**File**: `src/scripts/run_api.py`
```python
"""Launch the API server."""
import os
import argparse
import uvicorn
import logging
from alpha_pulse.api.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("alpha_pulse.api.launcher")


def main():
    """Run the API server."""
    parser = argparse.ArgumentParser(description="Launch the AI Hedge Fund Dashboard API")
    parser.add_argument("--host", help="Host to bind to", default=None)
    parser.add_argument("--port", help="Port to bind to", type=int, default=None)
    parser.add_argument("--reload", help="Enable auto-reload", action="store_true")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Get host and port
    host = args.host or config.host
    port = args.port or config.port
    
    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"Swagger UI available at http://{host}:{port}/docs")
    
    # Run server
    uvicorn.run(
        "alpha_pulse.api.main:app",
        host=host,
        port=port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
```

#### Step 5.3: Configuration File

**File**: `config/api_config.yaml`
```yaml
dashboard_api:
  host: "0.0.0.0"
  port: 8080
  
  # Authentication
  auth:
    jwt_secret: "${AP_JWT_SECRET}"
    token_expiry: 3600  # seconds
    api_keys_enabled: true
    
  # Rate limiting
  rate_limit:
    enabled: true
    requests_per_minute: 120
    
  # CORS settings
  cors:
    allowed_origins:
      - "http://localhost:3000"
      - "https://dashboard.example.com"
    allow_credentials: true
    
  # Caching
  cache:
    type: "memory"  # "memory", "redis"
    redis_url: "${AP_REDIS_URL}"
    default_ttl: 300  # seconds
    
  # Logging
  logging:
    level: "info"
    format: "json"
    
  # WebSocket
  websocket:
    max_connections: 1000
    ping_interval: 30  # seconds
```

### Testing and Demonstration

To test and run the API:

```bash
# Set JWT secret for testing
export AP_JWT_SECRET="your-secret-key"

# Run API server
python -m src.scripts.run_api --reload
```

The server will be available at:
- API Documentation: http://localhost:8080/docs
- API Endpoints: http://localhost:8080/api/*
- WebSocket Endpoints: ws://localhost:8080/ws/*

## Implementation Timeline

| Phase | Description | Timeline | Status |
|-------|-------------|----------|--------|
| 1 | Project Setup and Core Infrastructure | Day 1 | Pending |
| 2 | Cache and Data Access Layer | Day 2 | Pending |
| 3 | REST API Endpoints | Day 3 | Pending |
| 4 | WebSocket Implementation | Day 4-5 | Pending |
| 5 | Testing and Launch Script | Day 6-7 | Pending |

## Next Steps After Implementation

1. Begin implementing the Dashboard Frontend (Task 1.5)
2. Create API documentation for frontend developers
3. Implement additional data sources as needed
4. Add performance optimizations for production deployment