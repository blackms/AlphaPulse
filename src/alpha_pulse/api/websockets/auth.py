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