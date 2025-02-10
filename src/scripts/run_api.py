"""
Development server script for AlphaPulse API.
"""
import uvicorn
from loguru import logger

def main():
    """Run the FastAPI development server."""
    logger.info("Starting AlphaPulse API development server")
    
    # Configure uvicorn server
    uvicorn.run(
        "alpha_pulse.api.main:app",
        host="0.0.0.0",
        port=18001,  # Use a different port
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )

if __name__ == "__main__":
    main()