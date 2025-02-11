"""
Alpha Vantage fundamental data provider implementation.
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
from urllib.parse import urljoin

from ...interfaces import FundamentalData
from ..base import BaseDataProvider, retry_on_error, CacheMixin


class AlphaVantageProvider(BaseDataProvider, CacheMixin):
    """
    Fundamental data provider implementation using Alpha Vantage API.
    
    Features:
    - Financial statements retrieval
    - Company overview and metrics
    - Global market status
    - Earnings data
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: str,
        cache_ttl: int = 3600,  # 1 hour cache for fundamental data
        session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize Alpha Vantage provider.

        Args:
            api_key: Alpha Vantage API key
            cache_ttl: Cache time-to-live in seconds
            session: Optional aiohttp session
        """
        super().__init__("alpha_vantage", "fundamental", api_key)
        CacheMixin.__init__(self, cache_ttl)
        self._session = session
        self._base_params = {"apikey": api_key}

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if not self._session:
            self._session = aiohttp.ClientSession()

    async def _execute_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute Alpha Vantage API request."""
        await self._ensure_session()
        
        # Merge base params with request params
        request_params = {**self._base_params}
        if params:
            request_params.update(params)

        async with self._session.request(
            method=method,
            url=self.BASE_URL,
            params=request_params,
            headers=headers,
            json=data
        ) as response:
            response.raise_for_status()
            return await response.json()

    @retry_on_error(max_retries=3)
    async def get_financial_statements(
        self,
        symbol: str,
        start_time: Optional[datetime] = None
    ) -> FundamentalData:
        """
        Get financial statements data from Alpha Vantage.

        Args:
            symbol: Company symbol
            start_time: Optional start time for historical data

        Returns:
            FundamentalData object
        """
        cache_key = f"financials_{symbol}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        # Fetch all financial data concurrently
        tasks = [
            self._make_request("", params={
                "function": "INCOME_STATEMENT",
                "symbol": symbol
            }),
            self._make_request("", params={
                "function": "BALANCE_SHEET",
                "symbol": symbol
            }),
            self._make_request("", params={
                "function": "CASH_FLOW",
                "symbol": symbol
            }),
            self._make_request("", params={
                "function": "OVERVIEW",
                "symbol": symbol
            })
        ]

        income_stmt, balance_sheet, cash_flow, overview = \
            await asyncio.gather(*tasks)

        # Process and validate the data
        if not all([income_stmt, balance_sheet, cash_flow, overview]):
            raise ValueError(f"Missing fundamental data for {symbol}")

        # Get the most recent data
        latest_income = income_stmt.get("annualReports", [{}])[0]
        latest_balance = balance_sheet.get("annualReports", [{}])[0]
        latest_cash_flow = cash_flow.get("annualReports", [{}])[0]

        fundamental_data = FundamentalData(
            symbol=symbol,
            timestamp=datetime.now(),
            financial_ratios={
                "pe_ratio": float(overview.get("PERatio", 0)),
                "pb_ratio": float(overview.get("PriceToBookRatio", 0)),
                "ps_ratio": float(overview.get("PriceToSalesRatio", 0)),
                "roe": float(overview.get("ReturnOnEquityTTM", 0)),
                "roa": float(overview.get("ReturnOnAssetsTTM", 0)),
                "current_ratio": float(overview.get("CurrentRatio", 0)),
                "debt_to_equity": float(overview.get("DebtToEquityRatio", 0)),
                "gross_margin": float(overview.get("GrossProfitTTM", 0)) / 
                               float(overview.get("RevenueTTM", 1)),
                "operating_margin": float(overview.get("OperatingMarginTTM", 0)),
                "net_margin": float(overview.get("ProfitMargin", 0)),
                "dividend_yield": float(overview.get("DividendYield", 0))
            },
            balance_sheet={
                "total_assets": float(latest_balance.get("totalAssets", 0)),
                "total_liabilities": float(latest_balance.get("totalLiabilities", 0)),
                "total_equity": float(latest_balance.get("totalShareholderEquity", 0)),
                "cash": float(latest_balance.get("cashAndShortTermInvestments", 0)),
                "debt": float(latest_balance.get("shortLongTermDebtTotal", 0)),
                "working_capital": float(latest_balance.get("totalCurrentAssets", 0)) -
                                 float(latest_balance.get("totalCurrentLiabilities", 0))
            },
            income_statement={
                "revenue": float(latest_income.get("totalRevenue", 0)),
                "gross_profit": float(latest_income.get("grossProfit", 0)),
                "operating_income": float(latest_income.get("operatingIncome", 0)),
                "net_income": float(latest_income.get("netIncome", 0)),
                "eps": float(overview.get("EPS", 0)),
                "ebitda": float(latest_income.get("ebitda", 0))
            },
            cash_flow={
                "operating_cash_flow": float(latest_cash_flow.get("operatingCashflow", 0)),
                "investing_cash_flow": float(latest_cash_flow.get("cashflowFromInvestment", 0)),
                "financing_cash_flow": float(latest_cash_flow.get("cashflowFromFinancing", 0)),
                "free_cash_flow": float(latest_cash_flow.get("operatingCashflow", 0)) -
                                float(latest_cash_flow.get("capitalExpenditures", 0)),
                "capex": float(latest_cash_flow.get("capitalExpenditures", 0))
            },
            metadata={
                "sector": overview.get("Sector", ""),
                "industry": overview.get("Industry", ""),
                "market_cap": float(overview.get("MarketCapitalization", 0)),
                "beta": float(overview.get("Beta", 0)),
                "volume": float(overview.get("Volume", 0)),
                "exchange": overview.get("Exchange", ""),
                "currency": overview.get("Currency", ""),
                "country": overview.get("Country", ""),
                "last_updated": datetime.now().isoformat()
            }
        )

        # Cache the processed data
        self._store_in_cache(cache_key, fundamental_data)
        return fundamental_data

    @retry_on_error(max_retries=3)
    async def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Get company profile data from Alpha Vantage.

        Args:
            symbol: Company symbol

        Returns:
            Dictionary containing company profile data
        """
        cache_key = f"profile_{symbol}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        overview = await self._make_request("", params={
            "function": "OVERVIEW",
            "symbol": symbol
        })

        if not overview:
            raise ValueError(f"Invalid profile data for {symbol}")

        processed_data = {
            "name": overview.get("Name", ""),
            "description": overview.get("Description", ""),
            "sector": overview.get("Sector", ""),
            "industry": overview.get("Industry", ""),
            "country": overview.get("Country", ""),
            "exchange": overview.get("Exchange", ""),
            "currency": overview.get("Currency", ""),
            "market_cap": float(overview.get("MarketCapitalization", 0)),
            "pe_ratio": float(overview.get("PERatio", 0)),
            "dividend_yield": float(overview.get("DividendYield", 0)),
            "beta": float(overview.get("Beta", 0)),
            "52_week_high": float(overview.get("52WeekHigh", 0)),
            "52_week_low": float(overview.get("52WeekLow", 0)),
            "shares_outstanding": float(overview.get("SharesOutstanding", 0)),
            "updated_at": datetime.now().isoformat()
        }

        # Cache the processed data
        self._store_in_cache(cache_key, processed_data)
        return processed_data

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources on exit."""
        if self._session:
            await self._session.close()