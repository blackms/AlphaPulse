"""Tests for LLM-based portfolio analysis."""

import pytest
from datetime import datetime, UTC
from decimal import Decimal
from unittest.mock import patch, MagicMock
import json
import os

from langchain.schema import AIMessage
from alpha_pulse.portfolio.data_models import Position, PortfolioData, LLMAnalysisResult
from alpha_pulse.portfolio.llm_analysis import OpenAILLMAnalyzer, PortfolioAnalysisOutput
from alpha_pulse.portfolio.portfolio_manager import PortfolioManager

# Use test API key if not set in environment
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "fake-test-key-not-real-1234"

@pytest.fixture
def mock_exchange():
    """Create a mock exchange for testing."""
    exchange = MagicMock()
    
    # Create async mock methods
    async def mock_coro(return_value):
        return return_value
    
    # Set up mock returns
    prices = {
        "BTC/USDT": Decimal("50000"),
        "ETH/USDT": Decimal("3000"),
        "SOL/USDT": Decimal("100")
    }
    balances = {
        "BTC": MagicMock(total=Decimal("0.1")),
        "ETH": MagicMock(total=Decimal("2")),
        "SOL": MagicMock(total=Decimal("50")),
        "USDT": MockBalance(total=Decimal("10000"))
    }
    
    # Configure mock methods
    exchange.get_ticker_price.side_effect = lambda symbol: mock_coro(prices.get(symbol, Decimal("1")))
    exchange.get_balances.side_effect = lambda: mock_coro(balances)
    exchange.get_portfolio_value.side_effect = lambda: mock_coro(Decimal("20000"))
    exchange.get_average_entry_price.side_effect = lambda symbol: mock_coro(prices.get(symbol, Decimal("1")))
    
    return exchange

class MockBalance:
    def __init__(self, total):
        self.total = total

@pytest.fixture
def sample_portfolio_data():
    """Create sample portfolio data for testing."""
    return PortfolioData(
        positions=[
            Position(
                asset_id="BTC",
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("50000"),
                timestamp=datetime.now(UTC)
            ),
            Position(
                asset_id="ETH",
                quantity=Decimal("2"),
                entry_price=Decimal("3000"),
                current_price=Decimal("3000"),
                timestamp=datetime.now(UTC)
            )
        ],
        total_value=Decimal("20000"),
        cash_balance=Decimal("10000"),
        timestamp=datetime.now(UTC),
        risk_metrics={
            "volatility_target": 0.15,
            "max_drawdown_limit": 0.25
        }
    )

@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return AIMessage(content="""
{
    "recommendations": [
        "Reduce BTC exposure",
        "Increase stablecoin allocation"
    ],
    "risk_assessment": "Portfolio is overexposed to volatile assets",
    "rebalancing_suggestions": [
        {
            "asset": "BTC",
            "target_allocation": 0.2
        },
        {
            "asset": "USDT",
            "target_allocation": 0.5
        }
    ],
    "confidence_score": 0.85,
    "reasoning": "High market volatility suggests defensive positioning"
}""".strip())

@pytest.mark.asyncio
async def test_portfolio_llm_analysis(mock_exchange, sample_portfolio_data, mock_llm_response):
    """Test the full portfolio analysis flow with LLM."""
    
    # Initialize analyzer with mock LLM
    with patch("langchain_openai.chat_models.ChatOpenAI.ainvoke", return_value=mock_llm_response):
        analyzer = OpenAILLMAnalyzer(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4-test"
        )
        
        # Test direct analysis
        result = await analyzer.analyze_portfolio(sample_portfolio_data)
        assert isinstance(result, LLMAnalysisResult)
        assert len(result.recommendations) == 2
        assert result.confidence_score == 0.85
        assert len(result.rebalancing_suggestions) == 2
        
        # Test integration with portfolio manager
        config_path = "src/alpha_pulse/portfolio/portfolio_config.yaml"
        manager = PortfolioManager(config_path)
        
        analysis_result = await manager.analyze_portfolio_with_llm(analyzer, mock_exchange)
        assert isinstance(analysis_result, LLMAnalysisResult)
        assert analysis_result.recommendations
        assert analysis_result.risk_assessment
        assert analysis_result.timestamp is not None

@pytest.mark.asyncio
async def test_error_handling(sample_portfolio_data):
    """Test error handling in LLM analysis."""
    
    # Test API error handling
    with patch("langchain_openai.chat_models.ChatOpenAI.ainvoke", side_effect=Exception("API Error")):
        analyzer = OpenAILLMAnalyzer(api_key=os.getenv("OPENAI_API_KEY"))
        with pytest.raises(RuntimeError):
            await analyzer.analyze_portfolio(sample_portfolio_data)
    
    # Test invalid response format handling
    mock_response = AIMessage(content="Invalid JSON")
    with patch("langchain_openai.chat_models.ChatOpenAI.ainvoke", return_value=mock_response):
        analyzer = OpenAILLMAnalyzer(api_key=os.getenv("OPENAI_API_KEY"))
        with pytest.raises(ValueError):
            await analyzer.analyze_portfolio(sample_portfolio_data)

def test_custom_prompt_template():
    """Test custom prompt template functionality."""
    custom_template = "Custom prompt: {total_value}"
    analyzer = OpenAILLMAnalyzer(
        api_key=os.getenv("OPENAI_API_KEY"),
        prompt_template=custom_template
    )
    assert analyzer.prompt_template == custom_template