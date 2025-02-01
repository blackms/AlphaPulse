"""
Demo script for LLM-based portfolio analysis using real exchange data.
"""

import asyncio
import os
from loguru import logger
from dotenv import load_dotenv

from alpha_pulse.exchanges import ExchangeType, ExchangeFactory
from alpha_pulse.portfolio.llm_analysis import OpenAILLMAnalyzer
from alpha_pulse.portfolio.portfolio_manager import PortfolioManager

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
            model_name="gpt-4"
        )
        logger.info("Initialized LLM analyzer")

        # Get portfolio analysis
        analysis = await manager.analyze_portfolio_with_llm(analyzer, exchange)
        
        # Print results
        logger.info("\n=== Portfolio Analysis Results ===")
        logger.info("\nRecommendations:")
        for i, rec in enumerate(analysis.recommendations, 1):
            logger.info(f"{i}. {rec}")
            
        logger.info(f"\nRisk Assessment:\n{analysis.risk_assessment}")
        
        if analysis.rebalancing_suggestions:
            logger.info("\nRebalancing Suggestions:")
            for suggestion in analysis.rebalancing_suggestions:
                logger.info(f"- {suggestion.asset}: {suggestion.target_allocation:.2%}")
                
        logger.info(f"\nConfidence Score: {analysis.confidence_score:.2%}")
        logger.info(f"\nReasoning:\n{analysis.reasoning}")

    except Exception as e:
        logger.error(f"Error during demo: {e}")
        raise
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())