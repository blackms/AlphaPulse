# AlphaPulse Examples

This directory contains example scripts demonstrating various features and capabilities of AlphaPulse.

## Categories

- **data/** - Examples related to data fetching, processing, and pipeline usage
  - Data fetching from different sources
  - Historical data handling
  - Real-time data processing

- **trading/** - Trading system demonstrations
  - AI Hedge Fund complete flow
  - Paper trading
  - Backtesting strategies
  - Reinforcement Learning trading

- **portfolio/** - Portfolio management examples
  - Portfolio rebalancing
  - Risk management
  - Performance analysis
  - LLM-based portfolio analysis

- **analysis/** - Analysis and feature engineering
  - Feature engineering demonstrations
  - Model training workflows
  - Technical analysis

## Running Examples

Each example is a standalone script that can be run directly. Make sure you have installed AlphaPulse and its dependencies:

```bash
pip install -e .
```

Then you can run any example:

```bash
python examples/trading/demo_ai_hedge_fund.py
```

## Configuration

Most examples require configuration files located in the `config/` directory. Make sure to review and adjust the configuration files before running the examples.