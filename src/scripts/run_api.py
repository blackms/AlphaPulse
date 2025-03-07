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