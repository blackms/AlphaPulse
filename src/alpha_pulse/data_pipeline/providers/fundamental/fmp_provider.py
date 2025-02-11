"""
Financial Modeling Prep (FMP) fundamental data provider implementation.
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
from urllib.parse import urljoin

from ...interfaces import FundamentalData
from ..base import BaseDataProvider, retry_on_error, CacheMixin


class FMPProvider(BaseDataProvider, CacheMixin):
    """
    Fundamental data provider implementation using Financial Modeling Prep API.
    
    Features:
    - Financial statements retrieval
    - Company profiles and metrics
    - Industry analysis
    - Real-time fundamental data updates
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3/"

    def __init__(
        self,
        api_key: str,
        cache_ttl: int = 3600,  # 1 hour cache for fundamental data
        session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize FMP provider.

        Args:
            api_key: FMP API key
            cache_ttl: Cache time-to-live in seconds
            session: Optional aiohttp session
        """
        super().__init__("fmp", "fundamental", api_key)
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
        """Execute FMP API request."""
        await self._ensure_session()
        url = urljoin(self.BASE_URL, endpoint)
        
        # Merge base params with request params
        request_params = {**self._base_params}
        if params:
            request_params.update(params)

        async with self._session.request(
            method=method,
            url=url,
            params=request_params,
            headers=headers,
            json=data
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def _process_response(self, response: Any) -> Any:
        """Process FMP API response."""
        if not response:
            return None
        return response

    @retry_on_error(max_retries=3)
    async def get_financial_statements(
        self,
        symbol: str,
        start_time: Optional[datetime] = None
    ) -> FundamentalData:
        """
        Get financial statements data from FMP.

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
            self._make_request(f"income-statement/{symbol}"),
            self._make_request(f"balance-sheet-statement/{symbol}"),
            self._make_request(f"cash-flow-statement/{symbol}"),
            self._make_request(f"ratios/{symbol}"),
            self._make_request(f"company-key-metrics/{symbol}")
        ]

        income_stmt, balance_sheet, cash_flow, ratios, metrics = \
            await asyncio.gather(*tasks)

        # Process and validate the data
        if not all([income_stmt, balance_sheet, cash_flow, ratios, metrics]):
            raise ValueError(f"Missing fundamental data for {symbol}")

        # Get the most recent data
        latest_income = income_stmt[0] if income_stmt else {}
        latest_balance = balance_sheet[0] if balance_sheet else {}
        latest_cash_flow = cash_flow[0] if cash_flow else {}
        latest_ratios = ratios[0] if ratios else {}
        latest_metrics = metrics[0] if metrics else {}

        fundamental_data = FundamentalData(
            symbol=symbol,
            timestamp=datetime.now(),
            financial_ratios={
                "pe_ratio": latest_ratios.get("priceEarningsRatio", 0),
                "pb_ratio": latest_ratios.get("priceToBookRatio", 0),
                "ps_ratio": latest_ratios.get("priceToSalesRatio", 0),
                "roe": latest_ratios.get("returnOnEquity", 0),
                "roa": latest_ratios.get("returnOnAssets", 0),
                "current_ratio": latest_ratios.get("currentRatio", 0),
                "debt_to_equity": latest_ratios.get("debtToEquity", 0),
                "gross_margin": latest_ratios.get("grossProfitMargin", 0),
                "operating_margin": latest_ratios.get("operatingProfitMargin", 0),
                "net_margin": latest_ratios.get("netProfitMargin", 0),
                "dividend_yield": latest_ratios.get("dividendYield", 0)
            },
            balance_sheet={
                "total_assets": latest_balance.get("totalAssets", 0),
                "total_liabilities": latest_balance.get("totalLiabilities", 0),
                "total_equity": latest_balance.get("totalStockholdersEquity", 0),
                "cash": latest_balance.get("cashAndCashEquivalents", 0),
                "debt": latest_balance.get("totalDebt", 0),
                "working_capital": latest_balance.get("netWorkingCapital", 0)
            },
            income_statement={
                "revenue": latest_income.get("revenue", 0),
                "gross_profit": latest_income.get("grossProfit", 0),
                "operating_income": latest_income.get("operatingIncome", 0),
                "net_income": latest_income.get("netIncome", 0),
                "eps": latest_income.get("eps", 0),
                "ebitda": latest_income.get("ebitda", 0)
            },
            cash_flow={
                "operating_cash_flow": latest_cash_flow.get("operatingCashFlow", 0),
                "investing_cash_flow": latest_cash_flow.get("investingCashFlow", 0),
                "financing_cash_flow": latest_cash_flow.get("financingCashFlow", 0),
                "free_cash_flow": latest_cash_flow.get("freeCashFlow", 0),
                "capex": latest_cash_flow.get("capitalExpenditure", 0)
            },
            metadata={
                "sector": latest_metrics.get("sector", ""),
                "industry": latest_metrics.get("industry", ""),
                "market_cap": latest_metrics.get("marketCap", 0),
                "beta": latest_metrics.get("beta", 0),
                "volume": latest_metrics.get("volume", 0),
                "last_updated": datetime.now().isoformat()
            }
        )

        # Cache the processed data
        self._store_in_cache(cache_key, fundamental_data)
        return fundamental_data

    @retry_on_error(max_retries=3)
    async def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Get company profile data from FMP.

        Args:
            symbol: Company symbol

        Returns:
            Dictionary containing company profile data
        """
        cache_key = f"profile_{symbol}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        profile = await self._make_request(f"profile/{symbol}")
        if not profile or not isinstance(profile, list):
            raise ValueError(f"Invalid profile data for {symbol}")

        company_data = profile[0]
        processed_data = {
            "name": company_data.get("companyName", ""),
            "sector": company_data.get("sector", ""),
            "industry": company_data.get("industry", ""),
            "country": company_data.get("country", ""),
            "exchange": company_data.get("exchange", ""),
            "currency": company_data.get("currency", ""),
            "website": company_data.get("website", ""),
            "description": company_data.get("description", ""),
            "ceo": company_data.get("ceo", ""),
            "employees": company_data.get("fullTimeEmployees", 0),
            "address": company_data.get("address", ""),
            "phone": company_data.get("phone", ""),
            "market_cap": company_data.get("mktCap", 0),
            "price": company_data.get("price", 0),
            "beta": company_data.get("beta", 0),
            "volume_avg": company_data.get("volAvg", 0),
            "last_dividend": company_data.get("lastDiv", 0),
            "ipo_date": company_data.get("ipoDate", ""),
            "updated_at": datetime.now().isoformat()
        }

        # Cache the processed data
        self._store_in_cache(cache_key, processed_data)
        return processed_data

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources on exit."""
        if self._session:
            await self._session.close()