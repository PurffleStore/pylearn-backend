"""
Authentication utilities and security functions

Contains:
- JWT token validation decorator
- Security helpers
- Username anonymization for logging
- Cookie management utilities
"""

import os
import jwt
import hashlib
from functools import wraps
from flask import request, jsonify, current_app, make_response
from .database import get_db_connection
from .models import BlacklistedToken


def anonymize_username(username):
    """Create anonymous hash for logging while preserving uniqueness"""
    if not username:
        return "anonymous"
    return hashlib.sha256(f"user_{username}_salt".encode()).hexdigest()[:12]


def token_required(f):
    """
    JWT token validation decorator
    
    Validates access token from cookies and checks blacklist.
    Returns username to the decorated function.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('access_token')
        if not token:
            return jsonify({"message": "Token is missing"}), 401

        try:
            # Check blacklist
            conn = get_db_connection()
            if BlacklistedToken.is_blacklisted(conn, token):
                conn.close()
                return jsonify({"message": "Token has been revoked. Please log in again."}), 401
            conn.close()

            # Decode and validate token
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
            return f(data['username'], *args, **kwargs)

        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401
        except Exception as e:
            current_app.logger.exception("Auth error: %s", e)
            return jsonify({"message": "Server error"}), 500
    return decorated


def extract_username_from_request(req) -> str | None:
    """
    Extract username from various sources in request
    
    Checks in order:
    1. X-User header
    2. Request body JSON
    3. JWT cookie
    """
    # 1) Header
    hdr = req.headers.get("X-User")
    if hdr:
        return hdr

    # 2) Body
    data = req.get_json(silent=True) or {}
    if data.get("username"):
        return data.get("username")

    # 3) JWT cookie
    token = req.cookies.get("access_token")
    if token:
        try:
            payload = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])
            return payload.get("username")
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    return None


def add_cookie(resp, name: str, value: str, max_age: int):
    """
    Add secure cookie to response
    
    In prod: Secure + SameSite=None + Partitioned (works with third-party cookie protections).
    In dev: SameSite=Lax, not Secure.
    """
    IS_PROD = os.getenv("ENV", "dev").lower() == "prod"
    
    if IS_PROD:
        resp.headers.add(
            "Set-Cookie",
            f"{name}={value}; Path=/; Max-Age={max_age}; Secure; HttpOnly; SameSite=None; Partitioned"
        )
    else:
        resp.set_cookie(name, value, httponly=True, secure=False, samesite="Lax", max_age=max_age, path="/")


def validate_user_input(username: str, password: str) -> tuple[bool, str]:
    """
    Validate user input for signup/login
    
    Returns: (is_valid, error_message)
    """
    if not username or not password:
        return False, "Username and password are required"
    
    if len(username) < 3 or len(username) > 50:
        return False, "Username must be 3-50 characters"
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    # Additional validation can be added here
    # - Special character requirements
    # - Username format validation
    # - Password complexity checks
    
    return True, ""


def is_admin_user(conn, username: str) -> bool:
    """Check if user has admin role"""
    from .models import User
    user = User.find_by_username(conn, username)
    return user is not None and user.role == 'admin'


def log_security_event(event_type: str, username: str, ip_address: str, details: str = ""):
    """
    Log security events with anonymized usernames
    
    Args:
        event_type: Type of security event (login, logout, failed_login, etc.)
        username: Username (will be anonymized)
        ip_address: Request IP address
        details: Additional details about the event
    """
    user_hash = anonymize_username(username)
    current_app.logger.info(
        f"Security Event [{event_type}]: user_hash={user_hash}, ip={ip_address}, details={details}"
    )