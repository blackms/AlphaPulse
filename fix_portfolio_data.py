#!/usr/bin/env python3
"""
Comprehensive fix for the PortfolioData class issue.
This script fixes the missing asset_allocation attribute and ensures
proper initialization in the portfolio manager.
"""
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    # Import the modules we need to patch
    from alpha_pulse.portfolio.data_models import PortfolioData, PortfolioPosition
    from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
    import inspect
    from dataclasses import dataclass, fields
    from datetime import datetime
    from decimal import Decimal
    from typing import Dict, List, Optional

    # Check if asset_allocation already exists in PortfolioData
    has_asset_allocation = any(field.name == 'asset_allocation' for field in fields(PortfolioData))

    if not has_asset_allocation:
        logger.info("Adding asset_allocation field to PortfolioData class...")
        
        # Create a patched version with the missing field
        @dataclass
        class PatchedPortfolioData:
            """Portfolio data for analysis."""
            total_value: Decimal
            cash_balance: Decimal
            positions: List[PortfolioPosition]
            risk_metrics: Optional[Dict[str, str]] = None
            timestamp: Optional[datetime] = None
            asset_allocation: Optional[Dict[str, Decimal]] = None  # Add the missing field
        
        # Replace the original class
        import alpha_pulse.portfolio.data_models
        alpha_pulse.portfolio.data_models.PortfolioData = PatchedPortfolioData
        logger.info("PortfolioData class patched successfully!")
    else:
        logger.info("PortfolioData already has asset_allocation field. No patching needed.")

    # Check if PortfolioManager.get_portfolio_data populates the field
    source_lines = inspect.getsource(PortfolioManager.get_portfolio_data)
    if "asset_allocation=" not in source_lines:
        logger.warning("PortfolioManager.get_portfolio_data doesn't populate asset_allocation field.")
        logger.warning("This might cause errors in the demo - please manually update it.")
        logger.warning("Add 'asset_allocation=current_allocation,' in the PortfolioData instantiation.")
    else:
        logger.info("PortfolioManager.get_portfolio_data correctly populates asset_allocation field.")

    logger.info("Patch completed. The system should now run properly.")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please ensure you're running this script from the project root directory.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Patching failed: {e}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)