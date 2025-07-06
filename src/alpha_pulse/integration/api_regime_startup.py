"""API startup modifications for regime detection integration."""

# This file contains the code that should be added to src/alpha_pulse/api/main.py

import logging

logger = logging.getLogger(__name__)

async def initialize_regime_detection(app):
    """Initialize regime detection service and integration hub."""
    from ..services.regime_detection_service import RegimeDetectionService
    from ..integration.regime_integration import RegimeIntegrationHub
    from ..config.regime_config import RegimeDetectionConfig
    
    logger.info("Initializing regime detection system...")
    
    # Create regime detection configuration
    regime_config = RegimeDetectionConfig(
        n_states=5,
        update_interval=300,  # 5 minutes
        min_confidence=0.6,
        features={
            'volatility_windows': [5, 10, 20, 60],
            'return_windows': [1, 5, 20, 60],
            'use_vix': True,
            'use_sentiment': True,
            'use_volume': True,
            'use_market_breadth': True
        },
        cache_ttl=300,
        enable_alerts=True,
        alert_on_regime_change=True,
        alert_on_low_confidence=True
    )
    
    # Initialize regime detection service
    regime_service = RegimeDetectionService(
        config=regime_config,
        metrics_collector=app.state.metrics_collector,
        alert_manager=app.state.alert_manager,
        cache_manager=app.state.cache_manager
    )
    
    # Start the service
    await regime_service.start()
    
    # Create integration hub
    regime_hub = RegimeIntegrationHub(regime_service)
    await regime_hub.initialize()
    
    # Store in app state for access by other components
    app.state.regime_service = regime_service
    app.state.regime_hub = regime_hub
    
    logger.info("Regime detection system initialized successfully")
    
    return regime_hub


# Add to startup event:
"""
@app.on_event("startup")
async def startup_event():
    # ... existing initialization code ...
    
    # Initialize regime detection
    regime_hub = await initialize_regime_detection(app)
    
    # Subscribe existing components to regime updates
    if hasattr(app.state, 'risk_manager'):
        regime_hub.subscribe(app.state.risk_manager)
    
    if hasattr(app.state, 'portfolio_manager'):
        regime_hub.subscribe(app.state.portfolio_manager)
    
    # Subscribe all agents
    for agent in app.state.agents.values():
        regime_hub.subscribe(agent)
    
    logger.info("All components subscribed to regime updates")
"""

# Add to shutdown event:
"""
@app.on_event("shutdown")
async def shutdown_event():
    # ... existing shutdown code ...
    
    # Stop regime detection service
    if hasattr(app.state, 'regime_service'):
        await app.state.regime_service.stop()
        logger.info("Regime detection service stopped")
"""

# Add new endpoint for regime information:
"""
@app.get("/regime/current", response_model=RegimeStatusResponse)
async def get_current_regime(
    regime_hub: RegimeIntegrationHub = Depends(get_regime_hub)
):
    '''Get current market regime information.'''
    regime = regime_hub.get_current_regime()
    confidence = regime_hub.get_regime_confidence()
    params = regime_hub.get_regime_params()
    
    return RegimeStatusResponse(
        regime=regime.value if regime else "unknown",
        confidence=confidence,
        parameters=params,
        last_update=regime_hub._last_update
    )

@app.get("/regime/history", response_model=List[RegimeHistoryResponse])
async def get_regime_history(
    hours: int = 24,
    regime_service: RegimeDetectionService = Depends(get_regime_service)
):
    '''Get regime history for the specified time period.'''
    history = await regime_service.get_regime_history(hours=hours)
    return [
        RegimeHistoryResponse(
            timestamp=entry['timestamp'],
            regime=entry['regime'].value,
            confidence=entry['confidence'],
            transition=entry.get('transition', False)
        )
        for entry in history
    ]

@app.post("/regime/analyze")
async def analyze_regime(
    data: MarketDataRequest,
    regime_service: RegimeDetectionService = Depends(get_regime_service)
):
    '''Analyze market regime for given data.'''
    result = await regime_service.analyze_market_data(
        data.market_data,
        data.timestamp
    )
    
    return {
        "regime": result['regime'].value,
        "confidence": result['confidence'],
        "probabilities": result['probabilities'],
        "features": result['features'],
        "transition_probability": result.get('transition_probability', 0)
    }
"""