#!/usr/bin/env python3
"""
Script to run the AlphaPulse API server with uvicorn.
"""
import uvicorn
import os
from uvicorn.config import Config

if __name__ == "__main__":
    # Set longer timeout for uvicorn
    config = Config(
        "src.alpha_pulse.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        timeout_keep_alive=120,  # Increase keep-alive timeout to 120 seconds
    )
    
    # Run the application with uvicorn
    server = uvicorn.Server(config)
    server.run()