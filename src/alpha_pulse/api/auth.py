"""
Authentication utilities for the API.
"""
import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Demo credentials
DEMO_USERS = {
    "admin": {
        "username": "admin",
        "password": "password",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "role": "admin",
        "permissions": [
            "view_metrics",
            "view_alerts",
            "acknowledge_alerts",
            "view_portfolio",
            "view_trades",
            "view_system"
        ]
    }
}

# JWT settings
SECRET_KEY = "demo_secret_key_for_development_only"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password: str, username: str) -> bool:
    """
    Verify a password against the stored password for a user.
    
    In a real implementation, this would use password hashing.
    """
    if username not in DEMO_USERS:
        return False
    return plain_password == DEMO_USERS[username]["password"]


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """
    Authenticate a user with username and password.
    """
    if username not in DEMO_USERS:
        return None
    if not verify_password(password, username):
        return None
    return DEMO_USERS[username]


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict:
    """
    Get the current user from a JWT token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in DEMO_USERS:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    return DEMO_USERS[username]