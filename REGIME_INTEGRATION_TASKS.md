# Regime Detection Integration Task List

## Critical Issues to Fix

### 🔴 P0 - Critical (Must Fix Immediately)

#### 1. Start RegimeDetectionService in API
- **File**: `src/alpha_pulse/api/main.py`
- **Status**: ❌ NOT STARTED
- **Impact**: Service exists but never runs
- **Effort**: 1 hour
- **Code**: See `src/alpha_pulse/integration/api_regime_startup.py`

#### 2. Create RegimeIntegrationHub
- **File**: NEW - `src/alpha_pulse/integration/regime_integration.py`
- **Status**: ✅ CREATED (ready to integrate)
- **Impact**: Central distribution of regime information
- **Effort**: Already done

### 🟡 P1 - High Priority (Fix This Week)

#### 3. Update All Trading Agents
- **Files**: All files in `src/alpha_pulse/agents/`
- **Status**: ❌ Only Technical agent has basic regime detection
- **Impact**: Agents operate without market context
- **Effort**: 4 hours
- **Tasks**:
  - [ ] Modify agent factory to pass regime hub
  - [ ] Update Technical agent to use full HMM
  - [ ] Add regime awareness to Fundamental agent
  - [ ] Add regime awareness to Sentiment agent
  - [ ] Add regime awareness to Value agent
  - [ ] Add regime awareness to Activist agent

#### 4. Integrate Risk Manager
- **File**: `src/alpha_pulse/risk_management/risk_manager.py`
- **Status**: ❌ NOT INTEGRATED
- **Impact**: Risk limits don't adapt to market conditions
- **Effort**: 2 hours
- **Code**: See `src/alpha_pulse/integration/portfolio_risk_regime_integration.py`

#### 5. Integrate Portfolio Optimizer
- **Files**: `src/alpha_pulse/portfolio/*.py`
- **Status**: ❌ NOT INTEGRATED
- **Impact**: Portfolio allocation ignores market regime
- **Effort**: 3 hours
- **Tasks**:
  - [ ] Update MeanVarianceOptimizer
  - [ ] Update HierarchicalRiskParity
  - [ ] Update BlackLittermanOptimizer

### 🟢 P2 - Medium Priority (Fix This Month)

#### 6. Add Regime API Endpoints
- **File**: NEW - `src/alpha_pulse/api/routes/regime.py`
- **Status**: ❌ NOT CREATED
- **Impact**: No visibility into regime detection
- **Effort**: 2 hours
- **Endpoints**:
  - [ ] GET /regime/current
  - [ ] GET /regime/history
  - [ ] POST /regime/analyze
  - [ ] GET /regime/parameters

#### 7. Update Backtesting Framework
- **File**: `src/alpha_pulse/backtesting/backtest_engine.py`
- **Status**: ❌ NOT INTEGRATED
- **Impact**: Can't analyze historical regime performance
- **Effort**: 4 hours
- **Tasks**:
  - [ ] Track regime during backtest
  - [ ] Generate regime-specific metrics
  - [ ] Add regime transition analysis

#### 8. Create Regime Dashboard
- **Files**: `dashboard/src/components/RegimePanel.tsx`
- **Status**: ❌ NOT CREATED
- **Impact**: No real-time regime visibility
- **Effort**: 6 hours

### 🔵 P3 - Nice to Have

#### 9. Advanced Regime Features
- Multi-timeframe regime detection
- Cross-asset regime correlation
- Regime prediction (not just detection)
- Custom regime definitions

#### 10. Performance Optimizations
- Regime calculation caching
- Distributed regime computation
- GPU acceleration for HMM

## Implementation Checklist

### Week 1 Sprint
- [ ] Start RegimeDetectionService in API startup
- [ ] Test service is running and detecting regimes
- [ ] Update agent factory to pass regime hub
- [ ] Modify at least 2 agents to use regime

### Week 2 Sprint
- [ ] Complete all agent integrations
- [ ] Integrate risk manager
- [ ] Start portfolio optimizer integration
- [ ] Add basic regime endpoints

### Week 3 Sprint
- [ ] Complete portfolio optimizer integration
- [ ] Add regime tracking to backtesting
- [ ] Create regime monitoring dashboard
- [ ] Run integration tests

### Week 4 Sprint
- [ ] Performance testing with regime
- [ ] Documentation updates
- [ ] Production deployment preparation
- [ ] Monitor regime detection accuracy

## Code Snippets for Quick Implementation

### 1. Minimal API Startup Change
```python
# In api/main.py startup_event()

from alpha_pulse.services.regime_detection_service import RegimeDetectionService

# After other service initialization
regime_service = RegimeDetectionService(
    config=config.regime_detection,
    metrics_collector=metrics_collector
)
await regime_service.start()
app.state.regime_service = regime_service
```

### 2. Quick Agent Update
```python
# In any agent's analyze method

# Get current regime
regime = await app.state.regime_service.get_current_regime()
if regime == MarketRegime.CRISIS:
    # Reduce signal strength or skip
    return []
```

### 3. Risk Manager Quick Fix
```python
# In risk_manager.py calculate_position_size()

regime = await self.regime_service.get_current_regime()
regime_multipliers = {
    MarketRegime.BULL: 1.2,
    MarketRegime.BEAR: 0.6,
    MarketRegime.CRISIS: 0.3
}
size *= regime_multipliers.get(regime, 1.0)
```

## Validation Tests

### Test 1: Service Running
```bash
curl http://localhost:8000/regime/current
# Should return current regime, not 404
```

### Test 2: Agents Using Regime
```python
# Check agent signals include regime metadata
signal = agent.analyze(data)
assert 'regime' in signal.metadata
```

### Test 3: Risk Adjustment
```python
# Verify position sizes change with regime
bull_size = risk_manager.calculate_position_size(signal, MarketRegime.BULL)
crisis_size = risk_manager.calculate_position_size(signal, MarketRegime.CRISIS)
assert crisis_size < bull_size
```

## Success Metrics

- ✅ RegimeDetectionService runs continuously
- ✅ All 6 agents use regime information
- ✅ Risk limits adjust based on regime
- ✅ Portfolio allocation considers regime
- ✅ Regime visible in monitoring dashboard
- ✅ Backtest results include regime analysis

## Estimated Timeline

- **Total Effort**: ~25 hours
- **Critical Path**: 3-5 hours (just starting the service)
- **Full Integration**: 2-3 weeks
- **With Testing**: 4 weeks

## Risk Mitigation

1. **Gradual Rollout**: Start with just service running, add components gradually
2. **Feature Flags**: Use config to enable/disable regime integration
3. **Monitoring**: Track regime detection accuracy before full integration
4. **Fallback**: Keep non-regime code paths as fallback

## Questions to Answer

1. Should regime detection block trading if confidence is low?
2. How often should regime be recalculated? (Currently 5 min)
3. Should we alert on every regime change?
4. What's the minimum historical data for regime detection?

## Next Action

**START HERE**: Open `src/alpha_pulse/api/main.py` and add the regime service initialization in the startup event. This single change will activate the entire regime detection system that's currently sitting idle.