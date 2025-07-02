"""
Secure authentication module for AlphaPulse API.

This module implements secure authentication with proper secret management,
password hashing, and user management.
"""
from jose import jwt, JWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import logging

from ..config.secure_settings import get_settings, get_secrets_manager
from ..utils.secrets_manager import SecretsManager

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class User(BaseModel):
    """User model for authentication."""
    username: str
    email: str
    full_name: str
    role: str
    permissions: List[str]
    disabled: bool = False


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token data extracted from JWT."""
    username: Optional[str] = None
    permissions: List[str] = []


class AuthManager:
    """Manages authentication and user operations."""
    
    def __init__(self, secrets_manager: SecretsManager = None):
        self.secrets_manager = secrets_manager or get_secrets_manager()
        self.settings = get_settings()
        self._users_cache = {}
        self._load_users()
    
    def _load_users(self):
        """Load users from secure storage."""
        # In production, this would load from a database
        # For now, we'll create a default admin user with secure password
        
        # Try to get admin password from secrets
        admin_password = self.secrets_manager.get_secret("admin_password")
        if not admin_password:
            # Generate a secure password for first run
            import secrets
            admin_password = secrets.token_urlsafe(32)
            logger.warning(f"Generated admin password: {admin_password}")
            logger.warning("Please store this password securely and update it!")
            
            # Store the hashed password in secrets
            hashed = self.get_password_hash(admin_password)
            self.secrets_manager.primary_provider.set_secret(
                "admin_password_hash", 
                hashed
            )
        
        # Create default admin user
        admin_hash = self.secrets_manager.get_secret("admin_password_hash")
        if not admin_hash:
            admin_hash = self.get_password_hash(admin_password)
        
        self._users_cache["admin"] = UserInDB(
            username="admin",
            email="admin@alphapulse.io",
            full_name="System Administrator",
            role="admin",
            permissions=[
                "view_metrics",
                "view_alerts",
                "acknowledge_alerts",
                "view_portfolio",
                "manage_portfolio",
                "view_trades",
                "execute_trades",
                "view_system",
                "manage_system",
                "view_users",
                "manage_users"
            ],
            hashed_password=admin_hash
        )
        
        # Load additional users from secrets if available
        users_data = self.secrets_manager.get_secret("users")
        if isinstance(users_data, dict):
            for username, user_info in users_data.items():
                if username not in self._users_cache:
                    self._users_cache[username] = UserInDB(**user_info)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get a user by username."""
        return self._users_cache.get(username)
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate a user with username and password."""
        user = self.get_user(username)
        if not user:
            logger.warning(f"Authentication failed: User {username} not found")
            return None
        
        if not self.verify_password(password, user.hashed_password):
            logger.warning(f"Authentication failed: Invalid password for user {username}")
            return None
        
        if user.disabled:
            logger.warning(f"Authentication failed: User {username} is disabled")
            return None
        
        logger.info(f"User {username} authenticated successfully")
        return user
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.settings.security.jwt_expiration_minutes
            )
        
        to_encode.update({"exp": expire})
        
        # Get JWT secret from secure storage
        jwt_secret = self.settings.security.jwt_secret
        algorithm = self.settings.security.jwt_algorithm
        
        encoded_jwt = jwt.encode(to_encode, jwt_secret, algorithm=algorithm)
        return encoded_jwt
    
    def decode_token(self, token: str) -> Optional[TokenData]:
        """Decode and validate a JWT token."""
        try:
            jwt_secret = self.settings.security.jwt_secret
            algorithm = self.settings.security.jwt_algorithm
            
            payload = jwt.decode(token, jwt_secret, algorithms=[algorithm])
            username: str = payload.get("sub")
            permissions: List[str] = payload.get("permissions", [])
            
            if username is None:
                return None
            
            return TokenData(username=username, permissions=permissions)
            
        except JWTError as e:
            logger.error(f"JWT decode error: {str(e)}")
            return None
    
    def create_user(self, username: str, password: str, email: str, 
                   full_name: str, role: str = "user", 
                   permissions: List[str] = None) -> UserInDB:
        """Create a new user."""
        if username in self._users_cache:
            raise ValueError(f"User {username} already exists")
        
        # Default permissions based on role
        if permissions is None:
            if role == "admin":
                permissions = [
                    "view_metrics", "view_alerts", "acknowledge_alerts",
                    "view_portfolio", "manage_portfolio", "view_trades",
                    "execute_trades", "view_system", "manage_system",
                    "view_users", "manage_users"
                ]
            elif role == "trader":
                permissions = [
                    "view_metrics", "view_alerts", "view_portfolio",
                    "view_trades", "execute_trades"
                ]
            else:  # viewer
                permissions = [
                    "view_metrics", "view_alerts", "view_portfolio",
                    "view_trades"
                ]
        
        # Create user with hashed password
        hashed_password = self.get_password_hash(password)
        user = UserInDB(
            username=username,
            email=email,
            full_name=full_name,
            role=role,
            permissions=permissions,
            hashed_password=hashed_password
        )
        
        # Store user
        self._users_cache[username] = user
        
        # Persist to secrets
        users_data = self.secrets_manager.get_secret("users") or {}
        users_data[username] = user.dict()
        self.secrets_manager.primary_provider.set_secret("users", users_data)
        
        logger.info(f"Created new user: {username} with role: {role}")
        return user
    
    def update_user_password(self, username: str, new_password: str) -> bool:
        """Update a user's password."""
        user = self.get_user(username)
        if not user:
            return False
        
        # Update password hash
        user.hashed_password = self.get_password_hash(new_password)
        
        # Persist changes
        users_data = self.secrets_manager.get_secret("users") or {}
        users_data[username] = user.dict()
        self.secrets_manager.primary_provider.set_secret("users", users_data)
        
        logger.info(f"Updated password for user: {username}")
        return True
    
    def check_permission(self, user: User, permission: str) -> bool:
        """Check if a user has a specific permission."""
        return permission in user.permissions or user.role == "admin"


# Global auth manager instance
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get or create the global auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get the current user from a JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    auth_manager = get_auth_manager()
    token_data = auth_manager.decode_token(token)
    
    if not token_data or not token_data.username:
        raise credentials_exception
    
    user = auth_manager.get_user(token_data.username)
    if user is None:
        raise credentials_exception
    
    # Convert UserInDB to User (without password)
    return User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        permissions=user.permissions,
        disabled=user.disabled
    )


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get the current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_permission(permission: str):
    """Dependency to require a specific permission."""
    async def permission_checker(current_user: User = Depends(get_current_active_user)):
        auth_manager = get_auth_manager()
        if not auth_manager.check_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    return permission_checker


# Export key functions and classes
__all__ = [
    "User",
    "UserInDB",
    "Token",
    "TokenData",
    "AuthManager",
    "get_auth_manager",
    "get_current_user",
    "get_current_active_user",
    "require_permission"
]