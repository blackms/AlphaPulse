"""
Authentication utilities for the API with audit logging.
"""
from jose import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from pydantic import BaseModel

from alpha_pulse.config.secure_settings import get_secrets_manager
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Get audit logger
audit_logger = get_audit_logger()


class User(BaseModel):
    """User model."""
    username: str
    email: str
    full_name: str
    role: str
    permissions: List[str]
    disabled: bool = False
    tenant_id: Optional[str] = "00000000-0000-0000-0000-000000000001"  # Default tenant for demo users


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


# Demo users with proper password hashing
DEMO_USERS_DB = {
    "admin": UserInDB(
        username="admin",
        email="admin@alphapulse.io",
        full_name="Admin User",
        role="admin",
        permissions=[
            "view_metrics",
            "view_alerts",
            "acknowledge_alerts",
            "view_portfolio",
            "view_trades",
            "execute_trades",
            "view_system",
            "manage_users",
            "view_audit_logs"
        ],
        hashed_password=pwd_context.hash("admin123!@#"),  # Change in production
        disabled=False
    ),
    "trader": UserInDB(
        username="trader",
        email="trader@alphapulse.io",
        full_name="Trader User",
        role="trader",
        permissions=[
            "view_metrics",
            "view_alerts",
            "acknowledge_alerts",
            "view_portfolio",
            "view_trades",
            "execute_trades"
        ],
        hashed_password=pwd_context.hash("trader123!@#"),  # Change in production
        disabled=False
    ),
    "viewer": UserInDB(
        username="viewer",
        email="viewer@alphapulse.io",
        full_name="Viewer User",
        role="viewer",
        permissions=[
            "view_metrics",
            "view_portfolio",
            "view_trades"
        ],
        hashed_password=pwd_context.hash("viewer123!@#"),  # Change in production
        disabled=False
    )
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database."""
    if username in DEMO_USERS_DB:
        return DEMO_USERS_DB[username]
    return None


def authenticate_user(username: str, password: str, 
                     request: Optional[Request] = None) -> Optional[User]:
    """
    Authenticate a user with username and password.
    
    Args:
        username: Username
        password: Plain text password
        request: FastAPI request object for IP logging
        
    Returns:
        User object if authenticated, None otherwise
    """
    user = get_user(username)
    
    # Get client IP for audit logging
    client_ip = None
    if request and request.client:
        client_ip = request.client.host
        
    if not user:
        # Log failed login - unknown user
        with audit_logger.context(ip_address=client_ip):
            audit_logger.log_login(
                user_id=username,
                success=False,
                method="password",
                error="User not found"
            )
        return None
        
    if not verify_password(password, user.hashed_password):
        # Log failed login - wrong password
        with audit_logger.context(ip_address=client_ip, user_id=username):
            audit_logger.log_login(
                user_id=username,
                success=False,
                method="password",
                error="Invalid password"
            )
        return None
        
    if user.disabled:
        # Log failed login - disabled account
        with audit_logger.context(ip_address=client_ip, user_id=username):
            audit_logger.log_login(
                user_id=username,
                success=False,
                method="password",
                error="Account disabled"
            )
        return None
        
    # Log successful login
    with audit_logger.context(ip_address=client_ip, user_id=username):
        audit_logger.log_login(
            user_id=username,
            success=True,
            method="password"
        )
        
    return user


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Token payload data
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token
    """
    # Get JWT settings from secrets manager
    secrets_manager = get_secrets_manager()
    jwt_config = secrets_manager.get_secret("jwt_config", {
        "secret": "your-secret-key-change-in-production",
        "algorithm": "HS256",
        "expire_minutes": 30
    })
    
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=jwt_config["expire_minutes"])
        
    to_encode.update({"exp": expire})
    
    # Create token
    encoded_jwt = jwt.encode(
        to_encode, 
        jwt_config["secret"], 
        algorithm=jwt_config["algorithm"]
    )
    
    # Log token creation
    audit_logger.log(
        event_type=AuditEventType.AUTH_TOKEN_CREATED,
        event_data={
            'user_id': data.get('sub'),
            'expires_at': expire.isoformat()
        }
    )
    
    # Log secret access
    audit_logger.log_secret_access("jwt_config", "token_creation")
    
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme),
                          request: Request = None) -> User:
    """
    Get the current user from a JWT token.
    
    Args:
        token: JWT bearer token
        request: FastAPI request object
        
    Returns:
        Current user object
        
    Raises:
        HTTPException: If token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Get JWT settings
    secrets_manager = get_secrets_manager()
    jwt_config = secrets_manager.get_secret("jwt_config", {
        "secret": "your-secret-key-change-in-production",
        "algorithm": "HS256"
    })
    
    try:
        # Decode token
        payload = jwt.decode(
            token, 
            jwt_config["secret"], 
            algorithms=[jwt_config["algorithm"]]
        )
        username: str = payload.get("sub")
        
        if username is None:
            raise credentials_exception
            
    except jwt.JWTError as e:
        # Log failed authentication
        audit_logger.log(
            event_type=AuditEventType.AUTH_FAILED,
            event_data={
                'error': str(e),
                'token_preview': token[:20] + "..." if len(token) > 20 else token
            }
        )
        raise credentials_exception
        
    # Get user
    user = get_user(username)
    if user is None:
        # Log invalid user in token
        audit_logger.log(
            event_type=AuditEventType.AUTH_FAILED,
            event_data={
                'username': username,
                'error': 'User not found'
            }
        )
        raise credentials_exception
        
    # Store user in request state for middleware
    if request:
        request.state.user = user
        
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user.
    
    Args:
        current_user: Current user from token
        
    Returns:
        Active user object
        
    Raises:
        HTTPException: If user is disabled
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def has_permission(user: User, permission: str) -> bool:
    """
    Check if user has a specific permission.
    
    Args:
        user: User object
        permission: Permission to check
        
    Returns:
        True if user has permission
    """
    has_perm = permission in user.permissions
    
    # Log permission check for sensitive operations
    sensitive_permissions = [
        "execute_trades", 
        "manage_users", 
        "view_audit_logs",
        "modify_risk_limits"
    ]
    
    if permission in sensitive_permissions:
        audit_logger.log(
            event_type=AuditEventType.DATA_ACCESS,
            event_data={
                'user_id': user.username,
                'permission': permission,
                'granted': has_perm
            },
            data_classification="confidential"
        )
        
    return has_perm


class PermissionChecker:
    """Dependency for checking permissions."""
    
    def __init__(self, required_permissions: List[str]):
        """
        Initialize permission checker.
        
        Args:
            required_permissions: List of required permissions
        """
        self.required_permissions = required_permissions
        
    def __call__(self, user: User = Depends(get_current_active_user)) -> User:
        """
        Check if user has all required permissions.
        
        Args:
            user: Current active user
            
        Returns:
            User object if authorized
            
        Raises:
            HTTPException: If user lacks permissions
        """
        for permission in self.required_permissions:
            if not has_permission(user, permission):
                # Log unauthorized access attempt
                audit_logger.log(
                    event_type=AuditEventType.API_ERROR,
                    event_data={
                        'user_id': user.username,
                        'required_permission': permission,
                        'error': 'Insufficient permissions'
                    },
                    severity=AuditSeverity.WARNING,
                    success=False
                )
                
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Not enough permissions. Required: {permission}"
                )
                
        return user


# Convenience dependency factories
def require_admin(user: User = Depends(get_current_active_user)) -> User:
    """Require admin role."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user


def require_trader(user: User = Depends(get_current_active_user)) -> User:
    """Require trader role or higher."""
    if user.role not in ["admin", "trader"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Trader access required"
        )
    return user