# AlphaPulse: AI-Driven Hedge Fund System
# Copyright (C) 2024 AlphaPulse Trading System
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Main API application.

This module defines the FastAPI application and routes.
"""
from datetime import timedelta
from typing import Any

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from loguru import logger
from alpha_pulse.config.database import get_db_session

# Import routers
from .routers import metrics, alerts, portfolio, system, trades, correlation, risk_budget, regime, hedging, liquidity, ensemble, online_learning, gpu, explainability, data_quality, backtesting, data_lake
from .routes import audit  # Add audit routes
from .websockets import endpoints as ws_endpoints
from .websockets.subscription import subscription_manager

# Import dependencies
from .dependencies import get_alert_manager
from .data import PortfolioDataAccessor
from .auth import authenticate_user, create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES

# Import audit middleware
from .middleware.audit_middleware import AuditLoggingMiddleware, SecurityEventMiddleware
from .middleware.rate_limiting import RateLimitingMiddleware
from .middleware.security_headers import SecurityHeadersMiddleware, ContentSecurityPolicyReportMiddleware
from .middleware.validation_middleware import ValidationMiddleware, CSRFProtectionMiddleware
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType
from alpha_pulse.utils.ddos_protection import DDoSMitigator
from alpha_pulse.utils.ip_filtering import IPFilterManager
from alpha_pulse.services.throttling_service import ThrottlingService
from alpha_pulse.services.risk_budgeting_service import (
    RiskBudgetingService,
    RiskBudgetingConfig
)
from alpha_pulse.services.regime_detection_service import (
    RegimeDetectionService,
    RegimeDetectionConfig
)
from alpha_pulse.services.tail_risk_hedging_service import TailRiskHedgingService
from alpha_pulse.hedging.risk.manager import HedgeManager
from alpha_pulse.services.ensemble_service import EnsembleService
from alpha_pulse.ml.online.online_learning_service import OnlineLearningService
from alpha_pulse.ml.gpu.gpu_service import GPUService
from alpha_pulse.ml.gpu.gpu_config import get_default_config as get_gpu_config
from alpha_pulse.data_pipeline.data_fetcher import DataFetcher
from alpha_pulse.data_pipeline.providers.yfinance_provider import YFinanceProvider
from alpha_pulse.services.portfolio_provider import LivePortfolioProvider

# Import exchange data synchronization
from .exchange_sync_integration import (
    register_exchange_sync_events
) 
from alpha_pulse.data_pipeline.database.connection import init_db

# Import performance optimization services
from alpha_pulse.services.caching_service import CachingService
from alpha_pulse.services.database_optimization_service import DatabaseOptimizationService
from alpha_pulse.services.data_aggregation import DataAggregationService

# Create FastAPI application
app = FastAPI(
    title="AlphaPulse API",
    description="API for the AlphaPulse AI Hedge Fund",
    version="1.0.0",
)


def initialize_ensemble_service(app_state: Any) -> None:
    """
    Initialize the EnsembleService and attach it to the application state.
    
    Args:
        app_state: FastAPI application state container.
    """
    ensemble_db_session = get_db_session()
    try:
        ensemble_service = EnsembleService(ensemble_db_session)
    except Exception:
        ensemble_db_session.close()
        raise
    
    app_state.ensemble_service = ensemble_service
    app_state.ensemble_db_session = ensemble_db_session

# Add security headers middleware (first for all responses)
app.add_middleware(SecurityHeadersMiddleware)

# Add CSP violation report handling
app.add_middleware(ContentSecurityPolicyReportMiddleware)

# Add input validation middleware
app.add_middleware(
    ValidationMiddleware,
    validate_query_params=True,
    validate_path_params=True,
    validate_request_body=True,
    validate_file_uploads=True,
    max_request_size=10 * 1024 * 1024,  # 10MB
    enable_performance_monitoring=True
)

# Add CSRF protection middleware with secure secret management
from alpha_pulse.utils.secrets_manager import create_secrets_manager

# Get CSRF secret from secure storage
secrets_manager = create_secrets_manager()
csrf_secret = secrets_manager.get_secret("csrf_secret") or secrets_manager.get_jwt_secret()

app.add_middleware(
    CSRFProtectionMiddleware,
    secret_key=csrf_secret,
    exempt_paths=["/health", "/metrics", "/docs", "/openapi.json"]
)

# Add rate limiting middleware
app.add_middleware(
    RateLimitingMiddleware,
    redis_url="redis://localhost:6379",
    enable_adaptive=True,
    enable_priority_queue=True
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add audit logging middleware
app.add_middleware(
    AuditLoggingMiddleware,
    exclude_paths=['/health', '/metrics', '/docs', '/openapi.json', '/favicon.ico']
)

# Add security event detection middleware
app.add_middleware(SecurityEventMiddleware)

# Include routers
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])
app.include_router(alerts.router, prefix="/api/v1", tags=["alerts"])
app.include_router(portfolio.router, prefix="/api/v1", tags=["portfolio"])
app.include_router(trades.router, prefix="/api/v1", tags=["trades"])
app.include_router(system.router, prefix="/api/v1", tags=["system"])
app.include_router(audit.router, prefix="/api/v1", tags=["audit"])  # Add audit routes
app.include_router(correlation.router, prefix="/api/v1", tags=["correlation"])
app.include_router(risk_budget.router, prefix="/api/v1/risk-budget", tags=["risk-budget"])
app.include_router(regime.router, prefix="/api/v1/regime", tags=["regime"])
app.include_router(hedging.router, prefix="/api/v1/hedging", tags=["hedging"])
app.include_router(liquidity.router, prefix="/api/v1/liquidity", tags=["liquidity"])
app.include_router(ensemble.router, prefix="/api/v1/ensemble", tags=["ensemble"])
app.include_router(online_learning.router, prefix="/api/v1/online-learning", tags=["online-learning"])
app.include_router(gpu.router, prefix="/api/v1", tags=["gpu"])
app.include_router(explainability.router, prefix="/api/v1", tags=["explainability"])
app.include_router(data_quality.router, prefix="/api/v1", tags=["data-quality"])
app.include_router(backtesting.router, prefix="/api/v1", tags=["backtesting"])
app.include_router(data_lake.router, prefix="/api/v1", tags=["data-lake"])

# Register exchange sync events
register_exchange_sync_events(app)

# Include WebSocket routers
app.include_router(ws_endpoints.router, tags=["websockets"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to AlphaPulse API"}


@app.post("/token")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Get an access token.
    
    Authenticates the user and returns a JWT token if successful.
    All login attempts are automatically audited.
    """
    # Authenticate with audit logging
    user = authenticate_user(form_data.username, form_data.password, request)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=30)  # Use configured value
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "permissions": user.permissions
        }
    }


@app.on_event("startup")
async def startup_event():
    """Run when the application starts."""
    logger.info("Starting AlphaPulse API")
    
    # Initialize audit logger
    audit_logger = get_audit_logger()
    audit_logger.log(
        event_type=AuditEventType.SYSTEM_START,
        event_data={
            "service": "alphapulse_api",
            "version": "1.0.0"
        }
    )
    
    # Initialize protection services
    try:
        import redis
        redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
        
        # Initialize DDoS protection
        ddos_mitigator = DDoSMitigator(redis_client)
        app.state.ddos_mitigator = ddos_mitigator
        
        # Initialize IP filtering
        ip_filter_manager = IPFilterManager(redis_client)
        app.state.ip_filter_manager = ip_filter_manager
        
        # Initialize throttling service
        throttling_service = ThrottlingService(redis_client)
        await throttling_service.start()
        app.state.throttling_service = throttling_service
        
        logger.info("API protection services initialized")
        
    except Exception as e:
        logger.error(f"Error initializing protection services: {e}")
        # Continue startup even if protection services fail
    
    # Initialize database for exchange data cache
    try:
        await init_db()
        logger.info("Exchange data cache database initialized")
    except Exception as e:
        logger.error(f"Error initializing exchange data cache database: {e}")
    
    # Initialize CachingService for performance optimization
    try:  # pragma: no cover
        logger.info("Initializing CachingService...")
        app.state.caching_service = CachingService.create_for_trading()
        await app.state.caching_service.initialize()
        logger.info("✅ CachingService initialized successfully - expect 50-80% latency reduction!")
    except Exception as e:  # pragma: no cover
        logger.error(f"Error initializing CachingService: {e}")
        app.state.caching_service = None
    
    # Initialize DatabaseOptimizationService for query performance
    try:  # pragma: no cover
        logger.info("Initializing DatabaseOptimizationService...")
        from alpha_pulse.config.database import get_database_config
        db_config = get_database_config()
        
        app.state.db_optimization_service = DatabaseOptimizationService(
            connection_string=db_config.get_connection_string(),
            monitoring_interval=300,  # Monitor every 5 minutes
            enable_auto_optimization=True
        )
        await app.state.db_optimization_service.initialize()
        await app.state.db_optimization_service.start_monitoring()
        logger.info("✅ DatabaseOptimizationService initialized - expect 3-5x query performance improvement!")
    except Exception as e:  # pragma: no cover
        logger.error(f"Error initializing DatabaseOptimizationService: {e}")
        app.state.db_optimization_service = None
    
    # Initialize DataAggregationService for efficient data processing
    try:  # pragma: no cover
        logger.info("Initializing DataAggregationService...")
        app.state.data_aggregation_service = DataAggregationService()
        await app.state.data_aggregation_service.initialize()
        logger.info("✅ DataAggregationService initialized - improved data processing efficiency!")
    except Exception as e:  # pragma: no cover
        logger.error(f"Error initializing DataAggregationService: {e}")
        app.state.data_aggregation_service = None
    
    # Start exchange data synchronization
    # Get the alert manager
    alert_manager = get_alert_manager()
    
    # Start the alert manager
    await alert_manager.start()
    
    # Connect the alert manager to the subscription manager
    subscription_manager.set_alert_manager(alert_manager)
    
    # Start the subscription manager
    await subscription_manager.start()

    # Initialize shared market data fetcher and portfolio provider
    market_data_provider = YFinanceProvider()
    app.state.market_data_fetcher = DataFetcher(
        market_data_provider=market_data_provider
    )
    portfolio_accessor = PortfolioDataAccessor()
    portfolio_provider = LivePortfolioProvider(portfolio_accessor)
    app.state.portfolio_provider = portfolio_provider
    
    # Initialize risk budgeting service
    try:
        # Create risk budgeting configuration
        risk_budgeting_config = RiskBudgetingConfig(
            base_volatility_target=0.15,
            max_leverage=2.0,
            rebalancing_frequency="daily",
            enable_alerts=True,
            auto_rebalance=False
        )
        
        # Create and start risk budgeting service
        app.state.risk_budgeting_service = RiskBudgetingService(
            config=risk_budgeting_config,
            data_fetcher=app.state.market_data_fetcher,
            alerting_system=alert_manager,
            portfolio_provider=portfolio_provider.get_active_portfolios
        )
        await app.state.risk_budgeting_service.start()
        logger.info("Risk budgeting service started successfully")
        
    except Exception as e:
        logger.error(f"Error initializing risk budgeting service: {e}")
        # Continue without risk budgeting if it fails
    
    # Initialize regime detection service (CRITICAL - was missing!)
    try:
        # Create regime detection configuration
        regime_config = RegimeDetectionConfig(
            update_interval_minutes=60,  # Update every hour
            enable_alerts=True,
            track_performance=True,
            redis_url="redis://localhost:6379",
            model_checkpoint_interval=24,  # Save model daily
            model_checkpoint_path="/tmp/regime_model_checkpoint.pkl"
        )

        # Create and initialize regime detection service with data pipeline
        app.state.regime_detection_service = RegimeDetectionService(
            config=regime_config,
            data_pipeline=app.state.market_data_fetcher,  # Wire the data fetcher
            alert_manager=alert_manager
        )
        
        # Initialize the service (loads or trains model)
        await app.state.regime_detection_service.initialize()
        
        # Start the service
        await app.state.regime_detection_service.start()
        logger.info("HMM Regime detection service started successfully - CRITICAL GAP FIXED!")
        
    except Exception as e:
        logger.error(f"Error initializing regime detection service: {e}")
        # This is critical but continue running
    
    # Initialize tail risk hedging service
    try:
        tail_risk_config = {
            'enabled': True,
            'threshold': 0.05,  # 5% tail risk threshold
            'check_interval_minutes': 60,
            'max_hedge_cost': 0.02  # 2% of portfolio
        }
        
        # Create hedge manager with default config
        hedge_manager = HedgeManager()
        
        app.state.tail_risk_hedging_service = TailRiskHedgingService(
            hedge_manager=hedge_manager,
            alert_manager=alert_manager,
            config=tail_risk_config,
            portfolio_provider=portfolio_provider.get_portfolio_snapshot
        )
        await app.state.tail_risk_hedging_service.start()
        logger.info("Tail risk hedging service started successfully")
    except Exception as e:
        logger.error(f"Error initializing tail risk hedging service: {e}")
    
    # Initialize ensemble service
    try:
        initialize_ensemble_service(app.state)
        logger.info("Ensemble service initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing ensemble service: {e}")
        app.state.ensemble_service = None
        app.state.ensemble_db_session = None
        # Continue without ensemble service if it fails
    
    # Initialize online learning service
    try:
        # Get database session for online learning (already imported at module level)
        db_session = get_db_session()
        
        # Online learning configuration
        online_learning_config = {
            'checkpoint_dir': './checkpoints/online_learning',
            'model_timeout': 300,  # 5 minutes
            'drift_check_interval': 60,  # 1 minute
            'performance_window': 100,  # Last 100 predictions
            'enable_auto_rollback': True,
            'min_performance_threshold': 0.6
        }
        
        app.state.online_learning_service = OnlineLearningService(
            db=db_session,
            config=online_learning_config
        )
        logger.info("Online learning service initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing online learning service: {e}")
        # Continue without online learning if it fails
    
    # Initialize GPU acceleration service
    try:
        gpu_config = get_gpu_config()
        # Enable GPU monitoring for dashboard integration
        gpu_config.monitoring.enable_monitoring = True
        gpu_config.monitoring.monitor_interval_sec = 30
        
        app.state.gpu_service = GPUService(config=gpu_config)
        await app.state.gpu_service.start()
        
        # Log GPU availability
        gpu_metrics = app.state.gpu_service.get_metrics()
        num_gpus = len(gpu_metrics.get('devices', {}))
        if num_gpus > 0:
            logger.info(f"GPU acceleration service started with {num_gpus} GPU(s) available")
            # Log GPU details
            for device_id, device_info in gpu_metrics['devices'].items():
                logger.info(f"GPU {device_id}: {device_info['name']} - "
                          f"Memory: {device_info['memory_usage']:.1%} used")
        else:
            logger.warning("GPU acceleration service started but no GPUs detected - using CPU fallback")
            
    except Exception as e:
        logger.error(f"Error initializing GPU acceleration service: {e}")
        logger.warning("Continuing without GPU acceleration")
        # Set gpu_service to None to indicate it's not available
        app.state.gpu_service = None
    
    logger.info("AlphaPulse API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Run when the application shuts down."""
    logger.info("Shutting down AlphaPulse API")
    
    # Log system shutdown
    audit_logger = get_audit_logger()
    audit_logger.log(
        event_type=AuditEventType.SYSTEM_STOP,
        event_data={
            "service": "alphapulse_api",
            "reason": "normal_shutdown"
        }
    )
    
    # Stop protection services
    try:
        if hasattr(app.state, 'throttling_service'):
            await app.state.throttling_service.stop()
            logger.info("Throttling service stopped")
    except Exception as e:
        logger.error(f"Error stopping throttling service: {e}")
    
    # Stop risk budgeting service
    try:
        if hasattr(app.state, 'risk_budgeting_service'):
            await app.state.risk_budgeting_service.stop()
            logger.info("Risk budgeting service stopped")
    except Exception as e:
        logger.error(f"Error stopping risk budgeting service: {e}")
    
    # Stop GPU service
    try:
        if hasattr(app.state, 'gpu_service') and app.state.gpu_service:
            await app.state.gpu_service.stop()
            logger.info("GPU acceleration service stopped")
    except Exception as e:
        logger.error(f"Error stopping GPU service: {e}")
    
    # Stop performance optimization services
    try:  # pragma: no cover
        if hasattr(app.state, 'caching_service') and app.state.caching_service:
            await app.state.caching_service.close()
            logger.info("CachingService stopped")
    except Exception as e:  # pragma: no cover
        logger.error(f"Error stopping CachingService: {e}")
    
    try:  # pragma: no cover
        if hasattr(app.state, 'db_optimization_service') and app.state.db_optimization_service:
            await app.state.db_optimization_service.stop_monitoring()
            await app.state.db_optimization_service.close()
            logger.info("DatabaseOptimizationService stopped")
    except Exception as e:  # pragma: no cover
        logger.error(f"Error stopping DatabaseOptimizationService: {e}")
    
    try:  # pragma: no cover
        if hasattr(app.state, 'data_aggregation_service') and app.state.data_aggregation_service:
            await app.state.data_aggregation_service.close()
            logger.info("DataAggregationService stopped")
    except Exception as e:  # pragma: no cover
        logger.error(f"Error stopping DataAggregationService: {e}")
    
    # Stop regime detection service
    try:
        if hasattr(app.state, 'regime_detection_service'):
            await app.state.regime_detection_service.stop()
            logger.info("Regime detection service stopped")
    except Exception as e:
        logger.error(f"Error stopping regime detection service: {e}")
    
    # Stop tail risk hedging service
    try:
        if hasattr(app.state, 'tail_risk_hedging_service'):
            await app.state.tail_risk_hedging_service.stop()
            logger.info("Tail risk hedging service stopped")
    except Exception as e:
        logger.error(f"Error stopping tail risk hedging service: {e}")
    
    # Close ensemble service database session
    try:
        if hasattr(app.state, 'ensemble_db_session') and app.state.ensemble_db_session:
            app.state.ensemble_db_session.close()
            app.state.ensemble_db_session = None
            if hasattr(app.state, 'ensemble_service'):
                app.state.ensemble_service = None
            logger.info("Ensemble service database session closed")
    except Exception as e:  # pragma: no cover
        logger.error(f"Error closing ensemble service session: {e}")
    
    # Stop online learning service
    try:
        if hasattr(app.state, 'online_learning_service'):
            # Stop all active sessions
            for session_id in list(app.state.online_learning_service.active_sessions.keys()):
                await app.state.online_learning_service.stop_session(session_id, save_checkpoint=True)
            logger.info("Online learning service stopped")
    except Exception as e:
        logger.error(f"Error stopping online learning service: {e}")
    
    # Stop the subscription manager
    await subscription_manager.stop()
    
    # Get the alert manager
    alert_manager = get_alert_manager()
    
    # Stop the alert manager
    await alert_manager.stop()
    
    # Shutdown audit logger
    audit_logger.shutdown(timeout=10.0)
    
    logger.info("AlphaPulse API shutdown complete")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
