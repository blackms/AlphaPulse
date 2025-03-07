#!/usr/bin/env python3
"""
Run the demo with the patched PortfolioData class.
"""
import sys
import os
from pathlib import Path
import importlib.util
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and patch the PortfolioData class
from alpha_pulse.portfolio.data_models import PortfolioPosition

# Create a patched version of PortfolioData
@dataclass
class PortfolioData:
    """Portfolio data for analysis."""
    total_value: Decimal
    cash_balance: Decimal
    positions: List[PortfolioPosition]
    risk_metrics: Optional[Dict[str, str]] = None
    timestamp: Optional[datetime] = None

# Replace the original class
import alpha_pulse.portfolio.data_models
alpha_pulse.portfolio.data_models.PortfolioData = PortfolioData

print("PortfolioData class patched successfully!")

# Now run the demo script
os.system("SKIP_API_CHECK=true ./run_demo.sh")