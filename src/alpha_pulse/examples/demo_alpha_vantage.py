"""
Example script demonstrating Alpha Vantage data fetching.
"""
import asyncio
import os
from datetime import datetime
import logging
from dotenv import load_dotenv

from alpha_pulse.data_pipeline.providers.fundamental.alpha_vantage_provider import (
    AlphaVantageProvider
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main execution function."""
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment")

        # Initialize provider
        provider = AlphaVantageProvider(api_key=api_key)

        # Test symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]

        for symbol in symbols:
            logger.info(f"\nFetching data for {symbol}...")

            try:
                # Get financial statements
                financial_data = await provider.get_financial_statements(symbol)
                
                # Print financial ratios
                logger.info("\nFinancial Ratios:")
                for name, value in financial_data.financial_ratios.items():
                    logger.info(f"{name}: {value:.2f}")

                # Print key balance sheet items
                logger.info("\nBalance Sheet:")
                logger.info(f"Total Assets: ${financial_data.balance_sheet['total_assets']:,.2f}")
                logger.info(f"Total Liabilities: ${financial_data.balance_sheet['total_liabilities']:,.2f}")
                logger.info(f"Total Equity: ${financial_data.balance_sheet['total_equity']:,.2f}")

                # Print income statement highlights
                logger.info("\nIncome Statement:")
                logger.info(f"Revenue: ${financial_data.income_statement['revenue']:,.2f}")
                logger.info(f"Net Income: ${financial_data.income_statement['net_income']:,.2f}")
                logger.info(f"EPS: ${financial_data.income_statement['eps']:.2f}")

                # Print cash flow information
                logger.info("\nCash Flow:")
                logger.info(
                    f"Operating Cash Flow: "
                    f"${financial_data.cash_flow['operating_cash_flow']:,.2f}"
                )
                logger.info(
                    f"Free Cash Flow: "
                    f"${financial_data.cash_flow['free_cash_flow']:,.2f}"
                )

                # Get company profile
                profile = await provider.get_company_profile(symbol)
                
                logger.info("\nCompany Profile:")
                logger.info(f"Sector: {profile['sector']}")
                logger.info(f"Industry: {profile['industry']}")
                logger.info(f"Market Cap: ${profile['market_cap']:,.2f}")
                logger.info(f"Beta: {profile['beta']:.2f}")
                logger.info(f"52-Week Range: ${profile['52_week_low']:.2f} - "
                          f"${profile['52_week_high']:.2f}")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue

            # Add delay to respect rate limits
            await asyncio.sleep(12)  # Alpha Vantage free tier: 5 calls per minute

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up
        await provider.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())