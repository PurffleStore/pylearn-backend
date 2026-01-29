"""
Authentication module for MJ Learn Backend

This module provides:
- User authentication and authorization
- JWT token management
- Database models for user management
- Security utilities
"""

from .models import User, BlacklistedToken, RefreshToken
from .utils import token_required, anonymize_username
from .database import get_db_connection, init_db
from .routes import auth_bp

__all__ = [
    'User',
    'BlacklistedToken', 
    'RefreshToken',
    'token_required',
    'anonymize_username',
    'get_db_connection',
    'init_db',
    'auth_bp'
]