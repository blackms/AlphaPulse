"""HTML report generator for portfolio analysis."""
import os
from datetime import datetime
from typing import Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from .data_models import PortfolioData


def generate_portfolio_report(
    portfolio_data: PortfolioData,
    metrics: Dict[str, float],
    output_path: Optional[str] = None
) -> str:
    """
    Generate an HTML report from portfolio data and metrics.
    
    Args:
        portfolio_data: Portfolio data to report on
        metrics: Dictionary of performance metrics
        output_path: Optional specific output path
        
    Returns:
        str: Path to the generated HTML report
    """
    # Create output directory if needed
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/portfolio_analysis_{timestamp}.html"

    # Generate plots
    allocation_plot = _create_allocation_plot(portfolio_data)
    pnl_plot = _create_pnl_plot(portfolio_data)
    metrics_plot = _create_metrics_plot(metrics)

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
                        <div class="metric-value">{metrics.get('sharpe_ratio', 0):.2f}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                </div>
                
                <div id="metrics_plot">
                    {metrics_plot}
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
                        <td>{float(p.current_price):,.2f}</td>
                        <td>{float(p.quantity * p.current_price):,.2f}</td>
                        <td class="{'positive' if p.profit_loss > 0 else 'negative' if p.profit_loss < 0 else ''}">
                            {float(p.profit_loss):,.2f}
                        </td>
                    </tr>
                    ''' for p in portfolio_data.positions])}
                </table>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    {"".join([f'''
                    <tr>
                        <td>{metric.replace('_', ' ').title()}</td>
                        <td>{value:.4f}</td>
                    </tr>
                    ''' for metric, value in metrics.items()])}
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_path, "w") as f:
        f.write(html_content)
    
    return output_path


def _create_allocation_plot(portfolio_data: PortfolioData) -> str:
    """Create portfolio allocation plot."""
    # Calculate allocations
    allocations = [
        {
            'asset': p.asset_id,
            'value': float(p.quantity * p.current_price),
            'percentage': float(p.quantity * p.current_price) / float(portfolio_data.total_value) * 100
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


def _create_pnl_plot(portfolio_data: PortfolioData) -> str:
    """Create P/L visualization plot."""
    # Prepare data
    pnl_data = [
        {
            'asset': p.asset_id,
            'pnl': float(p.profit_loss)
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


def _create_metrics_plot(metrics: Dict[str, float]) -> str:
    """Create performance metrics visualization."""
    # Select key metrics to display
    key_metrics = {
        'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
        'Sortino Ratio': metrics.get('sortino_ratio', 0),
        'Information Ratio': metrics.get('information_ratio', 0),
        'Calmar Ratio': metrics.get('calmar_ratio', 0)
    }
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(key_metrics.keys()),
            y=list(key_metrics.values()),
            marker_color=['#2980b9', '#27ae60', '#8e44ad', '#d35400'],
            hovertemplate="<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title="Key Performance Metrics",
        height=300,
        showlegend=False,
        yaxis_title="Value"
    )
    
    return fig.to_html(full_html=False, include_plotlyjs=False)