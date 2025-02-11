"""
Alpha Vantage fundamental data provider implementation.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from ...interfaces import FundamentalData
from ..base import BaseDataProvider, retry_on_error, CacheMixin


class AlphaVantageProvider(BaseDataProvider, CacheMixin):
    """
    Alpha Vantage fundamental data provider implementation.
    
    Features:
    - Financial statements
    - Company overview
    - Earnings data
    - Key metrics
    """

    def __init__(
        self,
        api_key: str,
        cache_ttl: int = 3600  # 1 hour cache for fundamental data
    ):
        """
        Initialize Alpha Vantage provider.

        Args:
            api_key: Alpha Vantage API key
            cache_ttl: Cache time-to-live in seconds
        """
        BaseDataProvider.__init__(
            self,
            provider_name="alpha_vantage",
            provider_type="fundamental",
            base_url="https://www.alphavantage.co/query",
            default_headers=None
        )
        CacheMixin.__init__(self, cache_ttl=cache_ttl)
        
        self._api_key = api_key

    def _parse_financial_data(self, data: Dict[str, Any]) -> FundamentalData:
        """Parse Alpha Vantage financial data into FundamentalData object."""
        try:
            # Extract overview data
            overview = data.get("Overview", {})
            
            # Extract income statement
            income_stmt = data.get("Income Statement", {}).get("annualReports", [{}])[0]
            
            # Extract balance sheet
            balance_sheet = data.get("Balance Sheet", {}).get("annualReports", [{}])[0]
            
            # Extract cash flow
            cash_flow = data.get("Cash Flow", {}).get("annualReports", [{}])[0]
            
            return FundamentalData(
                symbol=overview.get("Symbol"),
                timestamp=datetime.now(),
                metadata={
                    "market_cap": float(overview.get("MarketCapitalization", 0)),
                    "sector": overview.get("Sector"),
                    "industry": overview.get("Industry"),
                    "exchange": overview.get("Exchange")
                },
                financial_ratios={
                    "pe_ratio": float(overview.get("PERatio", 0)),
                    "peg_ratio": float(overview.get("PEGRatio", 0)),
                    "beta": float(overview.get("Beta", 0)),
                    "dividend_yield": float(overview.get("DividendYield", 0))
                },
                income_statement={
                    "revenue": float(income_stmt.get("totalRevenue", 0)),
                    "gross_profit": float(income_stmt.get("grossProfit", 0)),
                    "operating_income": float(income_stmt.get("operatingIncome", 0)),
                    "net_income": float(income_stmt.get("netIncome", 0))
                },
                balance_sheet={
                    "total_assets": float(balance_sheet.get("totalAssets", 0)),
                    "total_liabilities": float(balance_sheet.get("totalLiabilities", 0)),
                    "total_equity": float(balance_sheet.get("totalShareholderEquity", 0))
                },
                cash_flow={
                    "operating_cash_flow": float(cash_flow.get("operatingCashflow", 0)),
                    "investing_cash_flow": float(cash_flow.get("cashflowFromInvestment", 0)),
                    "financing_cash_flow": float(cash_flow.get("cashflowFromFinancing", 0)),
                    "free_cash_flow": float(cash_flow.get("freeCashFlow", 0))
                }
            )
            
        except Exception as e:
            logger.exception(f"Error parsing financial data: {str(e)}")
            raise

    @retry_on_error(retries=3, delay=1.0)
    async def get_financial_statements(self, symbol: str) -> FundamentalData:
        """
        Get financial statements for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            FundamentalData object
        """
        cache_key = f"financials_{symbol}"
        
        # Try to get from cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            logger.debug(f"Fetching financial data for {symbol}")
            
            # Get company overview
            overview_response = await self._execute_request(
                endpoint="",
                params={
                    "function": "OVERVIEW",
                    "symbol": symbol,
                    "apikey": self._api_key
                }
            )
            overview_data = await self._process_response(overview_response)
            
            # Get income statement
            income_response = await self._execute_request(
                endpoint="",
                params={
                    "function": "INCOME_STATEMENT",
                    "symbol": symbol,
                    "apikey": self._api_key
                }
            )
            income_data = await self._process_response(income_response)
            
            # Get balance sheet
            balance_response = await self._execute_request(
                endpoint="",
                params={
                    "function": "BALANCE_SHEET",
                    "symbol": symbol,
                    "apikey": self._api_key
                }
            )
            balance_data = await self._process_response(balance_response)
            
            # Get cash flow
            cash_flow_response = await self._execute_request(
                endpoint="",
                params={
                    "function": "CASH_FLOW",
                    "symbol": symbol,
                    "apikey": self._api_key
                }
            )
            cash_flow_data = await self._process_response(cash_flow_response)
            
            # Combine all data
            combined_data = {
                "Overview": overview_data,
                "Income Statement": income_data,
                "Balance Sheet": balance_data,
                "Cash Flow": cash_flow_data
            }
            
            # Parse into FundamentalData object
            fundamental_data = self._parse_financial_data(combined_data)
            
            # Cache the results
            self._set_in_cache(cache_key, fundamental_data)
            
            return fundamental_data
            
        except Exception as e:
            logger.exception(f"Error fetching financial data for {symbol}: {str(e)}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await super().__aexit__(exc_type, exc_val, exc_tb)