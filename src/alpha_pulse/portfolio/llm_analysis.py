from abc import ABC, abstractmethod
from datetime import datetime, UTC
from typing import Dict, Optional
import json
import os

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from loguru import logger

from alpha_pulse.portfolio.data_models import PortfolioData, LLMAnalysisResult

class RebalancingSuggestion(BaseModel):
    """Schema for rebalancing suggestion."""
    asset: str = Field(description="Asset identifier")
    target_allocation: float = Field(description="Target allocation for the asset")

class PortfolioAnalysisOutput(BaseModel):
    """Schema for LLM output."""
    recommendations: list[str] = Field(description="List of portfolio recommendations")
    risk_assessment: str = Field(description="Detailed risk analysis")
    rebalancing_suggestions: Optional[list[RebalancingSuggestion]] = Field(
        description="Optional list of rebalancing suggestions with asset and target allocation"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Detailed reasoning behind recommendations")

class IPortfolioLLMAnalyzer(ABC):
    """Interface for portfolio analysis using LLM."""
    
    @abstractmethod
    async def analyze_portfolio(self, portfolio_data: PortfolioData) -> LLMAnalysisResult:
        """Analyze the portfolio and provide recommendations."""
        pass

class OpenAILLMAnalyzer(IPortfolioLLMAnalyzer):
    """OpenAI-based implementation of portfolio analyzer using langchain."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "o3-mini",
        prompt_template: Optional[str] = None,
        temperature: float = 0.7
    ):
        """Initialize the analyzer with OpenAI credentials and configuration.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the model to use (default: gpt-4)
            prompt_template: Custom prompt template (optional)
            temperature: Model temperature (default: 0.7)
        """
        os.environ["OPENAI_API_KEY"] = api_key
        model_kwargs = {"model_name": model_name}
        if "o3-" not in model_name:  # Only add temperature for non-o3 models
            model_kwargs["temperature"] = temperature
        self.model = ChatOpenAI(**model_kwargs)
        self.prompt_template = prompt_template or self._default_prompt_template
        self.output_parser = PydanticOutputParser(pydantic_object=PortfolioAnalysisOutput)
        logger.info(f"Initialized OpenAILLMAnalyzer with model: {model_name}")

    @property
    def _default_prompt_template(self) -> str:
        return """Analyze the following cryptocurrency portfolio data and provide comprehensive recommendations:

Portfolio Summary:
- Total Value: ${total_value}
- Cash Balance: ${cash_balance}
- Number of Positions: {num_positions}

Current Positions:
{positions}

Risk Metrics:
{risk_metrics}

Please provide a detailed analysis including:

1. Portfolio Optimization Recommendations:
   - Asset allocation adjustments
   - Risk management strategies
   - Entry/exit suggestions based on P/L
   - Diversification opportunities

2. Risk Assessment:
   - Portfolio concentration risk
   - Market risk exposure
   - Volatility analysis
   - Correlation between assets
   - Maximum drawdown potential

3. Fundamental Analysis:
   - Market trends for major holdings
   - Network metrics (where applicable)
   - Development activity
   - Adoption metrics
   - Recent news impact

4. Technical Analysis:
   - Current market positioning
   - Support/resistance levels
   - Volume analysis
   - Trend analysis

5. Rebalancing Strategy:
   - Target allocations with rationale
   - Suggested entry/exit points
   - Risk-adjusted position sizing
   - Market timing considerations

6. Detailed Reasoning:
   - Market context
   - Risk-reward rationale
   - Timeline considerations
   - Alternative scenarios

{format_instructions}
"""

    def _format_portfolio_data(self, portfolio_data: PortfolioData) -> str:
        """Format portfolio data into a prompt string."""
        # Add headers
        positions_str = (
            "Asset      Quantity      Price ($)        Value ($)    P/L ($)\n"
            "-------------------------------------------------------------\n"
        )
        
        # Add positions with consistent formatting
        positions_str += "\n".join([
            f"{p.asset_id:<5}  {p.quantity:>12,.4f}  {p.current_price:>12,.4f}  {p.market_value:>12,.2f}  {p.profit_loss:>9,.2f}"
            if p.current_price >= 0.01 else
            f"{p.asset_id:<5}  {p.quantity:>12,.4f}  {p.current_price:>12.2e}  {p.market_value:>12,.2f}  {p.profit_loss:>9,.2f}"
            for p in portfolio_data.positions
        ])
        
        risk_metrics_str = "\n".join([
            f"- {key}: {value}"
            for key, value in (portfolio_data.risk_metrics or {}).items()
        ])

        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        
        return prompt.format(
            total_value=portfolio_data.total_value,
            cash_balance=portfolio_data.cash_balance,
            num_positions=len(portfolio_data.positions),
            positions=positions_str,
            risk_metrics=risk_metrics_str,
            format_instructions=self.output_parser.get_format_instructions()
        )

    async def analyze_portfolio(self, portfolio_data: PortfolioData) -> LLMAnalysisResult:
        """Analyze the portfolio using OpenAI's API via langchain."""
        try:
            prompt = self._format_portfolio_data(portfolio_data)
            logger.debug(f"Sending prompt to OpenAI:\n{prompt}")

            # Get response from LLM
            response = (await self.model.ainvoke(prompt)).content
            logger.debug(f"Received response from OpenAI:\n{response}")

            try:
                # Parse the response
                parsed_response = self.output_parser.parse(response)

                result = LLMAnalysisResult(
                    recommendations=parsed_response.recommendations,
                    risk_assessment=parsed_response.risk_assessment,
                    confidence_score=parsed_response.confidence_score,
                    reasoning=parsed_response.reasoning,
                    timestamp=datetime.now(UTC),
                    rebalancing_suggestions=parsed_response.rebalancing_suggestions,
                    raw_response=response
                )

                logger.info("Successfully analyzed portfolio")
                return result

            except Exception as e:
                logger.error(f"Failed to parse LLM response: {e}")
                raise ValueError(f"Invalid response format from LLM: {e}")

        except ValueError as e:
            # Re-raise ValueError for invalid response format
            raise e
        except Exception as e:
            logger.error(f"Error during portfolio analysis: {e}")
            raise RuntimeError(f"Failed to get analysis from OpenAI: {e}")