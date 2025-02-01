"""
Demo script for LLM-based portfolio analysis using real exchange data.
"""

import asyncio
import os
import webbrowser
from loguru import logger
from dotenv import load_dotenv

from alpha_pulse.exchanges import ExchangeType, ExchangeFactory
from alpha_pulse.portfolio.llm_analysis import OpenAILLMAnalyzer
from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from alpha_pulse.portfolio.html_report import HTMLReportGenerator

# Load environment variables from .env file
load_dotenv()

async def main():
    """Run the LLM portfolio analysis demo."""
    try:
        # Set up Bybit credentials
        os.environ["ALPHA_PULSE_BYBIT_API_KEY"] = os.environ.get("BYBIT_API_KEY", "")
        os.environ["ALPHA_PULSE_BYBIT_API_SECRET"] = os.environ.get("BYBIT_API_SECRET", "")
        os.environ["ALPHA_PULSE_BYBIT_TESTNET"] = "false"

        # Initialize exchange
        exchange = await ExchangeFactory.create_exchange(
            exchange_type=ExchangeType.BYBIT,
            testnet=False
        )
        logger.info("Connected to Bybit exchange")

        # Initialize portfolio manager
        manager = PortfolioManager("src/alpha_pulse/portfolio/portfolio_config.yaml")
        logger.info("Initialized portfolio manager")

        # Initialize LLM analyzer
        # Get OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        analyzer = OpenAILLMAnalyzer(
            api_key=openai_api_key,
            model_name="o3-mini"
        )
        logger.info("Initialized LLM analyzer")

        # Get portfolio analysis
        logger.info("Analyzing portfolio...")
        portfolio_data = await manager.get_portfolio_data(exchange)
        analysis = await manager.analyze_portfolio_with_llm(analyzer, exchange)
        
        # Generate HTML report
        logger.info("Generating HTML report...")
        report_path = HTMLReportGenerator.generate_report(
            portfolio_data=portfolio_data,
            analysis_result=analysis,
            output_dir="reports"
        )
        
        # Open report in browser
        logger.info(f"Opening report: {report_path}")
        webbrowser.open(f"file://{os.path.abspath(report_path)}")
        
        logger.info("Analysis complete! Check the HTML report for detailed results.")

    except Exception as e:
        logger.error(f"Error during demo: {e}")
        raise
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())