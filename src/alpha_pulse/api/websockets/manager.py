"""
WebSocket connection manager.

This module provides a connection manager for WebSocket connections.
"""
import logging
import json
from typing import Dict, Set, Any, List
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    WebSocket connection manager.
    
    This class manages WebSocket connections and provides methods
    for broadcasting messages to connected clients.
    """
    
    def __init__(self):
        """Initialize the connection manager."""
        # Map of channel -> set of connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Map of connection -> set of channels
        self.connection_channels: Dict[WebSocket, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket):
        """
        Connect a WebSocket client.
        
        Args:
            websocket: The WebSocket connection
        """
        await websocket.accept()
        self.connection_channels[websocket] = set()
        logger.info(f"WebSocket client connected: {websocket}")
    
    async def disconnect(self, websocket: WebSocket):
        """
        Disconnect a WebSocket client.
        
        Args:
            websocket: The WebSocket connection
        """
        # Remove from all channels
        channels = self.connection_channels.get(websocket, set())
        for channel in channels:
            if channel in self.active_connections:
                self.active_connections[channel].discard(websocket)
        
        # Remove from connection map
        self.connection_channels.pop(websocket, None)
        logger.info(f"WebSocket client disconnected: {websocket}")
    
    async def subscribe(self, websocket: WebSocket, channel: str):
        """
        Subscribe a WebSocket client to a channel.
        
        Args:
            websocket: The WebSocket connection
            channel: The channel to subscribe to
        """
        # Add to channel map
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        self.active_connections[channel].add(websocket)
        
        # Add to connection map
        if websocket not in self.connection_channels:
            self.connection_channels[websocket] = set()
        self.connection_channels[websocket].add(channel)
        
        logger.info(f"WebSocket client subscribed to channel {channel}: {websocket}")
    
    async def unsubscribe(self, websocket: WebSocket, channel: str):
        """
        Unsubscribe a WebSocket client from a channel.
        
        Args:
            websocket: The WebSocket connection
            channel: The channel to unsubscribe from
        """
        # Remove from channel map
        if channel in self.active_connections:
            self.active_connections[channel].discard(websocket)
        
        # Remove from connection map
        if websocket in self.connection_channels:
            self.connection_channels[websocket].discard(channel)
        
        logger.info(f"WebSocket client unsubscribed from channel {channel}: {websocket}")
    
    async def broadcast(self, channel: str, message: Any):
        """
        Broadcast a message to all clients subscribed to a channel.
        
        Args:
            channel: The channel to broadcast to
            message: The message to broadcast
        """
        if channel not in self.active_connections:
            return
        
        # Convert message to JSON if it's not a string
        if not isinstance(message, str):
            message = json.dumps(message)
        
        # Send to all connections in the channel
        for connection in self.active_connections[channel]:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket client: {e}")
                # Remove the connection if it's closed
                await self.disconnect(connection)
    
    def get_connection_count(self, channel: str = None) -> int:
        """
        Get the number of active connections.
        
        Args:
            channel: The channel to count connections for (optional)
        
        Returns:
            The number of active connections
        """
        if channel:
            return len(self.active_connections.get(channel, set()))
        return len(self.connection_channels)
    
    def get_channels(self) -> List[str]:
        """
        Get the list of active channels.
        
        Returns:
            The list of active channels
        """
        return list(self.active_connections.keys())


# Create a singleton instance
connection_manager = ConnectionManager()