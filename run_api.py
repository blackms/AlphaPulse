#!/usr/bin/env python3
"""
Script to run the AlphaPulse API server with uvicorn.
"""
import uvicorn
import os
import sys
from uvicorn.config import Config
from loguru import logger

# Configure Loguru for DEBUG level
def setup_logging():
    """Configure logging to show DEBUG level messages."""
    # Remove default logger
    logger.remove()
    
    # Add console handler with DEBUG level
    logger.add(
        sys.stderr,
        level="DEBUG",  # Set to DEBUG to see all debug messages
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Also log to file for analysis
    logger.add(
        "debug_credentials.log",
        level="DEBUG",
        rotation="20 MB",
        retention="1 week"
    )
    
    logger.debug("Logging configured at DEBUG level")

if __name__ == "__main__":
    # Configure logging first
    setup_logging()
    logger.info("Starting AlphaPulse API with DEBUG logging enabled")
    
    # Set longer timeout for uvicorn
    config = Config(
        "src.alpha_pulse.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        timeout_keep_alive=120,  # Increase keep-alive timeout to 120 seconds
        log_level="debug",  # Set uvicorn log level to debug too
    )
    
    # Run the application with uvicorn
    logger.info("Starting uvicorn server")
    server = uvicorn.Server(config)
    server.run()