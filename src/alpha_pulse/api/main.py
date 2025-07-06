"""
Main API application.

This module defines the FastAPI application and routes.
"""
from loguru import logger
from datetime import timedelta
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm

# Import routers
from .routers import metrics, alerts, portfolio, system, trades, correlation, risk_budget, regime
from .routes import audit  # Add audit routes
from .websockets import endpoints as ws_endpoints
from .websockets.subscription import subscription_manager

# Import dependencies
from .dependencies import get_alert_manager
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
from alpha_pulse.risk.dynamic_budgeting import DynamicRiskBudgetManager
from alpha_pulse.services.regime_detection_service import (
    RegimeDetectionService,
    RegimeDetectionConfig
)

# Import exchange data synchronization
from .exchange_sync_integration import (
    register_exchange_sync_events
) 
from alpha_pulse.data_pipeline.database.connection import init_db

# Create FastAPI application
app = FastAPI(
    title="AlphaPulse API",
    description="API for the AlphaPulse AI Hedge Fund",
    version="1.0.0",
)

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

# Add CSRF protection middleware
app.add_middleware(
    CSRFProtectionMiddleware,
    secret_key="your-csrf-secret-key",  # In production, use proper secret management
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
    
    # Start exchange data synchronization
    # Get the alert manager
    alert_manager = get_alert_manager()
    
    # Start the alert manager
    await alert_manager.start()
    
    # Connect the alert manager to the subscription manager
    subscription_manager.set_alert_manager(alert_manager)
    
    # Start the subscription manager
    await subscription_manager.start()
    
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
        
        # Create dynamic risk budget manager
        budget_manager = DynamicRiskBudgetManager(
            base_volatility_target=risk_budgeting_config.base_volatility_target,
            max_leverage=risk_budgeting_config.max_leverage,
            rebalancing_frequency=risk_budgeting_config.rebalancing_frequency
        )
        
        # Create and start risk budgeting service
        app.state.risk_budgeting_service = RiskBudgetingService(
            budget_manager=budget_manager,
            monitoring_interval=60,  # 1 minute
            alert_manager=alert_manager,
            config=risk_budgeting_config
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
        
        # Create and initialize regime detection service
        app.state.regime_detection_service = RegimeDetectionService(
            config=regime_config,
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
    
    # Stop regime detection service
    try:
        if hasattr(app.state, 'regime_detection_service'):
            await app.state.regime_detection_service.stop()
            logger.info("Regime detection service stopped")
    except Exception as e:
        logger.error(f"Error stopping regime detection service: {e}")
    
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