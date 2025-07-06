# Current Integration Points Analysis

## Overview
This document maps where Sprint 3-4 features are currently connected (or disconnected) in the AlphaPulse system.

## Sprint 3: Risk Management Features

### Tail Risk Hedging
**Current Integrations:**
- ✅ API Router: `/src/alpha_pulse/api/routers/hedging.py`
  - `POST /api/v1/hedging/analyze` - Analyze hedging opportunities
  - `POST /api/v1/hedging/execute` - Execute hedge trades
  - `POST /api/v1/hedging/close` - Close hedge positions
- ✅ Service Layer: Standalone service with LLM integration

**Missing Integrations:**
- ❌ Not called by portfolio optimizer during rebalancing
- ❌ Not triggered by risk manager on threshold breaches
- ❌ No automatic hedging based on tail risk metrics
- ❌ No integration with position sizing logic

### Correlation Analysis
**Current Integrations:**
- ✅ Used by `DynamicRiskBudgetManager.calculate_correlation_adjustments()`
- ✅ Internal use in risk calculations

**Missing Integrations:**
- ❌ No API endpoints for correlation data
- ❌ Not displayed in any UI components
- ❌ Not used by portfolio optimizer for diversification
- ❌ No alerts on correlation breakdowns

### Dynamic Risk Budgeting
**Current Integrations:**
- ✅ Service runs background monitoring loops
- ✅ Uses correlation analysis internally
- ✅ Integrates with regime detection (if it were running)

**Missing Integrations:**
- ❌ Not connected to position sizing in execution
- ❌ Risk budgets not enforced by trading engine
- ❌ No API endpoints for budget status
- ❌ Portfolio rebalancing ignores budget changes

### Liquidity Management
**Current Integrations:**
- ✅ Standalone service implementation
- ✅ Comprehensive liquidity models

**Missing Integrations:**
- ❌ Order router doesn't use liquidity analysis
- ❌ Position sizing ignores liquidity constraints
- ❌ No API endpoints for liquidity metrics
- ❌ Execution algorithms don't optimize for impact

### Monte Carlo Simulation
**Current Integrations:**
- ✅ GPU acceleration support built-in
- ✅ Used by some risk calculations internally

**Missing Integrations:**
- ❌ Not exposed through API
- ❌ Risk reports don't include simulation results
- ❌ Portfolio optimization doesn't use scenarios
- ❌ No user-facing scenario analysis tools

## Sprint 4: ML Enhancement Features

### Ensemble Methods
**Current Integrations:**
- ✅ Database models created and migrated
- ✅ Comprehensive service implementation

**Missing Integrations:**
- ❌ Agent manager doesn't use ensemble aggregation
- ❌ Signal router bypasses ensemble logic
- ❌ No API endpoints for ensemble management
- ❌ Trading decisions ignore ensemble confidence

### Online Learning
**Current Integrations:**
- ✅ Standalone service with session management
- ✅ Model persistence and checkpointing

**Missing Integrations:**
- ❌ Trading models don't update online
- ❌ No integration with model serving pipeline
- ❌ Agents don't adapt to market changes
- ❌ No API endpoints for adaptation monitoring

### GPU Acceleration
**Current Integrations:**
- ✅ Monte Carlo engine can use GPU
- ✅ Some portfolio optimization GPU support

**Missing Integrations:**
- ❌ Trading agents don't use GPU models
- ❌ Model training doesn't leverage GPU
- ❌ Real-time inference not GPU-optimized
- ❌ No API endpoints for GPU management

### Explainable AI
**Current Integrations:**
- ✅ Service layer with multiple explanation methods
- ✅ Database persistence for explanations

**Missing Integrations:**
- ❌ Trading decisions lack explanations in UI
- ❌ No API endpoints for explanations
- ❌ Compliance reporting doesn't use explainability
- ❌ Users can't understand model decisions

## Critical Integration Paths

### 1. Signal Generation Flow
```
Market Data → Agents → [MISSING: Ensemble Aggregation] → Signal Router → Portfolio Manager
                ↓
         [MISSING: Online Learning Updates]
```

### 2. Risk Management Flow
```
Positions → Risk Manager → [MISSING: Correlation Analysis Display]
              ↓
       [MISSING: Dynamic Risk Budget Enforcement]
              ↓
       [MISSING: Tail Risk Hedging Triggers]
```

### 3. Order Execution Flow
```
Trading Signal → [MISSING: Liquidity Analysis] → Order Router → Exchange
                            ↓
                 [MISSING: Impact Optimization]
```

### 4. Model Serving Flow
```
Trained Models → [MISSING: GPU Optimization] → Model Server → Agents
       ↓
[MISSING: Online Updates]
       ↓
[MISSING: Explainability]
```

## Service Initialization Gaps

### In API Main (`src/alpha_pulse/api/main.py`):
**Currently Initialized:**
- Basic services (monitoring, exchange, portfolio)
- WebSocket manager

**Not Initialized:**
- ❌ Risk budgeting service
- ❌ Liquidity risk service  
- ❌ Simulation service
- ❌ Ensemble service
- ❌ Online learning service
- ❌ GPU service
- ❌ Explainability service

## Configuration Gaps

### Missing Configuration Integration:
- No environment variables for ML features
- No config files for risk management features
- Services use hardcoded defaults
- No user-controllable parameters

## Testing Gaps

### Integration Tests Missing:
- No end-to-end tests using Sprint 3-4 features
- No performance impact tests
- No user journey tests
- No business metric validation

## Conclusion

The Sprint 3-4 features exist as isolated islands of functionality. They are well-implemented but disconnected from the main trading flow, making them effectively invisible and unused. This is a systemic integration problem similar to the regime detection issue.