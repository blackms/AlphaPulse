#!/usr/bin/env python3
"""
Script to run the AlphaPulse API server with uvicorn.
"""
import uvicorn

if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(
        "src.alpha_pulse.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )