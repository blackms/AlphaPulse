# AlphaPulse Project Context

## Project Overview

AlphaPulse is a comprehensive cryptocurrency and stock trading system with AI-powered portfolio management. The system implements a sophisticated architecture for automated trading, risk management, and portfolio optimization.

## Core Components

### 1. AI Hedge Fund System
- Multi-agent trading system with specialized strategies
- Intelligent signal aggregation
- Risk-aware position sizing
- Performance tracking and adaptation
- Real-time strategy adjustment

### 2. Portfolio Management
- Black-Litterman portfolio optimization
- Hierarchical Risk Parity (HRP) strategy
- Modern Portfolio Theory (MPT) implementation
- LLM-assisted portfolio analysis
- Dynamic rebalancing

### 3. Risk Management
- Multi-asset risk analysis
- Position sizing optimization
- Portfolio-level risk controls
- Real-time monitoring
- Stop-loss management

### 4. Hedging Strategies
- Grid-based hedging with risk management
- Basic futures hedging
- Position tracking and rebalancing
- Multiple trading modes (Real/Paper/Recommendation)
- Delta-neutral strategies

### 5. Data Pipeline
- Real-time market data integration
- Historical data management
- Feature engineering
- Database integration
- Automated data cleaning
- Exchange data synchronization
- Robust error handling and graceful degradation

### 6. Execution
- Multi-exchange support (Binance, Bybit)
- Paper trading simulation
- Real-time order management
- Risk-aware execution
- Smart order routing

### 7. Machine Learning
- Feature generation
- Model training pipeline
- Reinforcement learning integration
- LLM-powered analysis
- Hyperparameter optimization

### 8. Dashboard and Monitoring
- Real-time performance visualization
- Portfolio analytics
- Alert management
- System health monitoring
- WebSocket-based real-time updates

## System Architecture

The AI Hedge Fund system implements a multi-agent trading architecture with four main layers:
1. Input Layer (Specialized Agents)
2. Risk Management Layer
3. Portfolio Management Layer
4. Output Layer (Trading Actions)

## Error Handling Approach

The system implements robust error handling with several key patterns:

1. **Graceful Degradation**: When non-critical functionality is unavailable, the system continues to operate with reduced capabilities rather than failing completely.

2. **Specific Exception Handling**: We catch specific exceptions rather than generic ones to avoid masking unrelated errors and ensure appropriate handling for each error type.

3. **Informative Logging**: All errors are logged with detailed information about what happened, the context, and any recovery actions taken.

4. **Circuit Breaker Pattern**: For operations that might fail repeatedly, we implement circuit breakers to prevent cascading failures and maintain system stability.

## Current Focus Areas

The current focus areas include:

1. **Dashboard Frontend Implementation**
   - Real-time data visualization
   - Portfolio and trade management interface
   - Alert monitoring and management
   - System configuration and control

2. **Data Pipeline Robustness**
   - Improving error handling and recovery
   - Implementing version compatibility layers
   - Enhancing logging and monitoring
   - Adding comprehensive testing

3. **Exchange Integration**
   - Completing Binance integration
   - Adding support for additional exchanges
   - Implementing robust error handling for API interactions
   - Creating comprehensive testing suite

## Memory Bank Structure

This Memory Bank contains the following core files:

1. **productContext.md** (this file): Project overview, goals, and key components
2. **activeContext.md**: Current session context and what we're working on
3. **progress.md**: Track progress and next steps
4. **decisionLog.md**: Document key architectural decisions
5. **systemPatterns.md**: Document system patterns and design principles

Additional files may be created as needed to document specific aspects of the project.