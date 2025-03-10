#!/usr/bin/env python
"""
Run the AlphaPulse API server.

This script starts the API server using uvicorn.
"""
import os
import sys
import argparse
import logging
import uvicorn
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('run_api')


def main():
    """Run the API server."""
    parser = argparse.ArgumentParser(description='Run the AlphaPulse API server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload', default=True)
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--log-level', default='info', help='Log level')
    
    args = parser.parse_args()
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    
    # Run the server
    uvicorn.run(
        "alpha_pulse.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level
    )


if __name__ == '__main__':
    main()
