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