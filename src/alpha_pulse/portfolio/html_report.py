"""HTML report generator for portfolio analysis."""
import os
from datetime import datetime
from typing import List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from alpha_pulse.portfolio.data_models import LLMAnalysisResult, PortfolioData

class HTMLReportGenerator:
    """Generate HTML reports for portfolio analysis."""

    @staticmethod
    def generate_report(
        portfolio_data: PortfolioData,
        analysis_result: LLMAnalysisResult,
        output_dir: str = "reports"
    ) -> str:
        """Generate an HTML report from portfolio analysis results.
        
        Args:
            portfolio_data: Portfolio data used for analysis
            analysis_result: LLM analysis results
            output_dir: Directory to save the report (default: reports)
            
        Returns:
            str: Path to the generated HTML report
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create report filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_analysis_{timestamp}.html"
        filepath = os.path.join(output_dir, filename)
        
        # Generate plots
        allocation_plot = HTMLReportGenerator._create_allocation_plot(portfolio_data)
        pnl_plot = HTMLReportGenerator._create_pnl_plot(portfolio_data)
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Portfolio Analysis Report - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2 {{
                    color: #2c3e50;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                .recommendations li {{
                    margin-bottom: 10px;
                    color: #34495e;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background-color: #fff;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2980b9;
                }}
                .metric-label {{
                    color: #7f8c8d;
                    font-size: 14px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f8f9fa;
                    color: #2c3e50;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .positive {{
                    color: #27ae60;
                }}
                .negative {{
                    color: #c0392b;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Portfolio Analysis Report</h1>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="section">
                    <h2>Portfolio Overview</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">${float(portfolio_data.total_value):,.2f}</div>
                            <div class="metric-label">Total Value</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${float(portfolio_data.cash_balance):,.2f}</div>
                            <div class="metric-label">Cash Balance</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{len(portfolio_data.positions)}</div>
                            <div class="metric-label">Number of Positions</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{analysis_result.confidence_score:.0%}</div>
                            <div class="metric-label">Analysis Confidence</div>
                        </div>
                    </div>
                    
                    <div id="allocation_plot">
                        {allocation_plot}
                    </div>
                    
                    <div id="pnl_plot">
                        {pnl_plot}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Current Positions</h2>
                    <table>
                        <tr>
                            <th>Asset</th>
                            <th>Quantity</th>
                            <th>Price ($)</th>
                            <th>Value ($)</th>
                            <th>P/L ($)</th>
                        </tr>
                        {"".join([f'''
                        <tr>
                            <td>{p.asset_id}</td>
                            <td>{float(p.quantity):,.4f}</td>
                            <td>{f"{float(p.current_price):,.4f}" if float(p.current_price) >= 0.01 else f"{float(p.current_price):.2e}"}</td>
                            <td>{float(p.market_value):,.2f}</td>
                            <td class="{'positive' if float(p.profit_loss) > 0 else 'negative' if float(p.profit_loss) < 0 else ''}">
                                {float(p.profit_loss):,.2f}
                            </td>
                        </tr>
                        ''' for p in portfolio_data.positions])}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Analysis Results</h2>
                    <h3>Recommendations</h3>
                    <ul class="recommendations">
                        {"".join([f"<li>{rec}</li>" for rec in analysis_result.recommendations])}
                    </ul>
                    
                    <h3>Risk Assessment</h3>
                    <p>{analysis_result.risk_assessment}</p>
                    
                    <h3>Rebalancing Suggestions</h3>
                    {"<table><tr><th>Asset</th><th>Target Allocation</th></tr>" + 
                    "".join([f'''
                        <tr>
                            <td>{sug.asset}</td>
                            <td>{sug.target_allocation:.1%}</td>
                        </tr>
                    ''' for sug in (analysis_result.rebalancing_suggestions or [])]) + "</table>"
                    if analysis_result.rebalancing_suggestions else "<p>No rebalancing suggestions provided.</p>"}
                    
                    <h3>Analysis Reasoning</h3>
                    <p>{analysis_result.reasoning}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(filepath, "w") as f:
            f.write(html_content)
        
        return filepath

    @staticmethod
    def _create_allocation_plot(portfolio_data: PortfolioData) -> str:
        """Create portfolio allocation plot."""
        # Calculate allocations
        allocations = [
            {
                'asset': p.asset_id,
                'value': p.market_value,
                'percentage': p.market_value / portfolio_data.total_value * 100
            }
            for p in portfolio_data.positions
        ]
        
        # Sort by value
        allocations.sort(key=lambda x: x['value'], reverse=True)
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=[a['asset'] for a in allocations],
            values=[a['value'] for a in allocations],
            hovertemplate="<b>%{label}</b><br>" +
                         "Value: $%{value:,.2f}<br>" +
                         "Allocation: %{percent:.1%}<extra></extra>"
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            height=500
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def _create_pnl_plot(portfolio_data: PortfolioData) -> str:
        """Create P/L visualization plot."""
        # Prepare data
        pnl_data = [
            {
                'asset': p.asset_id,
                'pnl': p.profit_loss
            }
            for p in portfolio_data.positions
            if p.profit_loss != 0  # Only include positions with non-zero P/L
        ]
        
        # Sort by absolute P/L
        pnl_data.sort(key=lambda x: abs(x['pnl']), reverse=True)
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="P/L",
            orientation="v",
            measure=["relative"] * len(pnl_data),
            x=[d['asset'] for d in pnl_data],
            y=[d['pnl'] for d in pnl_data],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#c0392b"}},
            increasing={"marker": {"color": "#27ae60"}},
            hovertemplate="<b>%{x}</b><br>P/L: $%{y:,.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Profit/Loss by Position",
            height=400,
            showlegend=False,
            yaxis_title="Profit/Loss ($)"
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)