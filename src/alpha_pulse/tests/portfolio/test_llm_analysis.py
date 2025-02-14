"""
Tests for the portfolio LLM analysis module.
"""
import pytest
import pytest_asyncio
from datetime import datetime, UTC
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import json

from alpha_pulse.portfolio.llm_analysis import (
    RebalancingSuggestion,
    PortfolioAnalysisOutput,
    OpenAILLMAnalyzer,
    IPortfolioLLMAnalyzer
)
from alpha_pulse.portfolio.data_models import (
    PortfolioData,
    PortfolioPosition,
    LLMAnalysisResult
)


@pytest.fixture
def sample_portfolio_data():
    """Create sample portfolio data for testing."""
    return PortfolioData(
        total_value=Decimal("100000.00"),
        cash_balance=Decimal("20000.00"),
        positions=[
            PortfolioPosition(
                asset_id="BTC",
                quantity=Decimal("1.5"),
                current_price=Decimal("50000.00"),
                market_value=Decimal("75000.00"),
                profit_loss=Decimal("5000.00")
            ),
            PortfolioPosition(
                asset_id="ETH",
                quantity=Decimal("3.0"),
                current_price=Decimal("2000.00"),
                market_value=Decimal("6000.00"),
                profit_loss=Decimal("-1000.00")
            )
        ],
        risk_metrics={
            "volatility": "0.25",
            "sharpe_ratio": "1.5",
            "max_drawdown": "-0.15"
        }
    )


@pytest.fixture
def sample_llm_response():
    """Create sample LLM response for testing."""
    return {
        "recommendations": [
            "Reduce BTC exposure by 10%",
            "Increase ETH allocation"
        ],
        "risk_assessment": "Portfolio shows moderate risk with high BTC concentration",
        "rebalancing_suggestions": [
            {"asset": "BTC", "target_allocation": 0.6},
            {"asset": "ETH", "target_allocation": 0.3}
        ],
        "confidence_score": 0.85,
        "reasoning": "Market conditions suggest rebalancing towards ETH"
    }


@pytest.fixture
def mock_openai_response(sample_llm_response):
    """Create mock OpenAI response."""
    return MagicMock(content=json.dumps(sample_llm_response))


@pytest_asyncio.fixture
async def llm_analyzer():
    """Create LLM analyzer instance."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        analyzer = OpenAILLMAnalyzer(
            api_key="test_key",
            model_name="test-model",
            temperature=0.5
        )
        # Mock the ChatOpenAI instance
        analyzer.model = AsyncMock()
        return analyzer


def test_rebalancing_suggestion_model():
    """Test RebalancingSuggestion model."""
    suggestion = RebalancingSuggestion(
        asset="BTC",
        target_allocation=0.6
    )
    assert suggestion.asset == "BTC"
    assert suggestion.target_allocation == 0.6


def test_portfolio_analysis_output_model():
    """Test PortfolioAnalysisOutput model."""
    output = PortfolioAnalysisOutput(
        recommendations=["Reduce BTC exposure"],
        risk_assessment="Moderate risk",
        rebalancing_suggestions=[
            RebalancingSuggestion(asset="BTC", target_allocation=0.6)
        ],
        confidence_score=0.85,
        reasoning="Market conditions favorable"
    )
    
    assert len(output.recommendations) == 1
    assert output.risk_assessment == "Moderate risk"
    assert len(output.rebalancing_suggestions) == 1
    assert output.confidence_score == 0.85
    assert output.reasoning == "Market conditions favorable"


@pytest.mark.asyncio
async def test_format_portfolio_data(llm_analyzer, sample_portfolio_data):
    """Test portfolio data formatting."""
    formatted_data = llm_analyzer._format_portfolio_data(sample_portfolio_data)
    
    # Check key components are present
    assert str(sample_portfolio_data.total_value) in formatted_data
    assert str(sample_portfolio_data.cash_balance) in formatted_data
    assert "BTC" in formatted_data
    assert "ETH" in formatted_data
    assert "volatility" in formatted_data
    assert "sharpe_ratio" in formatted_data


@pytest.mark.asyncio
async def test_analyze_portfolio_success(
    llm_analyzer,
    sample_portfolio_data,
    mock_openai_response
):
    """Test successful portfolio analysis."""
    llm_analyzer.model.ainvoke.return_value = mock_openai_response
    
    result = await llm_analyzer.analyze_portfolio(sample_portfolio_data)
    
    assert isinstance(result, LLMAnalysisResult)
    assert len(result.recommendations) == 2
    assert result.confidence_score == 0.85
    assert len(result.rebalancing_suggestions) == 2
    assert isinstance(result.timestamp, datetime)
    assert result.timestamp.tzinfo == UTC


@pytest.mark.asyncio
async def test_analyze_portfolio_invalid_response(llm_analyzer, sample_portfolio_data):
    """Test handling of invalid LLM response."""
    llm_analyzer.model.ainvoke.return_value = MagicMock(content="Invalid JSON")
    
    with pytest.raises(ValueError, match="Invalid response format from LLM"):
        await llm_analyzer.analyze_portfolio(sample_portfolio_data)


@pytest.mark.asyncio
async def test_analyze_portfolio_api_error(llm_analyzer, sample_portfolio_data):
    """Test handling of API error."""
    llm_analyzer.model.ainvoke.side_effect = Exception("API Error")
    
    with pytest.raises(RuntimeError, match="Failed to get analysis from OpenAI"):
        await llm_analyzer.analyze_portfolio(sample_portfolio_data)


def test_custom_prompt_template():
    """Test custom prompt template initialization."""
    custom_template = "Custom template {total_value}"
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        analyzer = OpenAILLMAnalyzer(
            api_key="test_key",
            prompt_template=custom_template
        )
        assert analyzer.prompt_template == custom_template


def test_default_prompt_template():
    """Test default prompt template content."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        analyzer = OpenAILLMAnalyzer(api_key="test_key")
        template = analyzer._default_prompt_template
        
        # Check key sections are present
        assert "Portfolio Summary:" in template
        assert "Current Positions:" in template
        assert "Risk Metrics:" in template
        assert "Portfolio Optimization Recommendations:" in template
        assert "Risk Assessment:" in template
        assert "Rebalancing Strategy:" in template


def test_model_temperature_handling():
    """Test model temperature handling for different model types."""
    with patch('alpha_pulse.portfolio.llm_analysis.ChatOpenAI') as mock_chat:
        # Test o3 model (should not have temperature)
        OpenAILLMAnalyzer(
            api_key="test_key",
            model_name="o3-mini",
            temperature=0.5
        )
        mock_chat.assert_called_with(model_name="o3-mini")
        
        mock_chat.reset_mock()
        
        # Test non-o3 model (should have temperature)
        OpenAILLMAnalyzer(
            api_key="test_key",
            model_name="gpt-4",
            temperature=0.5
        )
        mock_chat.assert_called_with(model_name="gpt-4", temperature=0.5)


@pytest.mark.asyncio
async def test_portfolio_data_validation(llm_analyzer):
    """Test portfolio data validation."""
    # Test with invalid portfolio data
    invalid_data = PortfolioData(
        total_value=Decimal("-100.00"),  # Invalid negative value
        cash_balance=Decimal("0"),
        positions=[]
    )
    
    with pytest.raises(ValueError, match="Portfolio total value must be positive"):
        llm_analyzer._validate_portfolio_data(invalid_data)
        llm_analyzer._format_portfolio_data(invalid_data)


@pytest.mark.asyncio
async def test_small_number_formatting(llm_analyzer):
    """Test formatting of very small numbers in portfolio data."""
    portfolio_data = PortfolioData(
        total_value=Decimal("100.00"),
        cash_balance=Decimal("50.00"),
        positions=[
            PortfolioPosition(
                asset_id="SHIB",
                quantity=Decimal("1000000"),
                current_price=Decimal("0.00001"),  # Very small price
                market_value=Decimal("10.00"),
                profit_loss=Decimal("1.00")
            )
        ]
    )
    
    formatted_data = llm_analyzer._format_portfolio_data(portfolio_data)
    assert "1.00e-5" in formatted_data  # Scientific notation for small numbers