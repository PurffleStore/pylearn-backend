"""
Authentication routes and endpoints

Contains all authentication-related Flask routes:
- User registration and login
- Token refresh and logout  
- Admin user management
- Database diagnostics
"""

import os
import datetime
import bcrypt
import jwt
from flask import Blueprint, request, jsonify, make_response, current_app

from .database import get_db_connection
from .models import User, BlacklistedToken, RefreshToken
from .utils import (
    token_required, 
    anonymize_username, 
    add_cookie, 
    validate_user_input,
    is_admin_user,
    log_security_event
)

# Create authentication blueprint
auth_bp = Blueprint('auth', __name__)

# Enable debug prints by setting AUTH_DEBUG=1 in environment (Railway Variables)
AUTH_DEBUG = os.getenv("AUTH_DEBUG", "0") == "1"

def _dbg(msg: str):
    if AUTH_DEBUG:
        print(msg)


@auth_bp.route("/dashboard")
@token_required
def dashboard(username):
    """Protected dashboard endpoint"""
    return jsonify({"message": f"Welcome {username} to your dashboard!"})


@auth_bp.route("/login", methods=["POST"])
def login():
    """User login endpoint"""
    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')

    # Input validation
    is_valid, error_msg = validate_user_input(username, password)
    if not is_valid:
        return jsonify({"message": error_msg}), 400
    
    # Normalize username to prevent case sensitivity issues
    username = username.lower()

    _dbg("=== LOGIN DEBUG START ===")
    _dbg(f"remote_addr={request.remote_addr} username_hash={anonymize_username(username)}")
    try:
        from .database import _read_db_env, _mask_secret  # safe helpers
        host, port, name, user, pwd, timeout = _read_db_env()
        _dbg(f"DB_HOST={host} DB_PORT={port} DB_NAME={name} DB_USER={_mask_secret(user)} DB_PASSWORD={_mask_secret(pwd)} TIMEOUT={timeout}")
        if pwd and str(pwd).startswith(("`", "'", '"')):
            _dbg("!! WARNING: DB_PASSWORD starts with a quote/backtick. Remove it in Railway Variables / .env.")
    except Exception as _e:
        _dbg(f"Could not read DB env for debug: {_e}")

    try:
        conn = get_db_connection()
        _dbg('DB connection opened for login')
        try:
            cur = conn.cursor()
            cur.execute('SELECT DATABASE(), USER()')
            row = cur.fetchone()
            _dbg(f"DB selected={row[0] if row else None} server_user={row[1] if row else None}")
        except Exception as _e:
            _dbg(f"DB info query failed (non-fatal): {_e}")
        user = User.find_by_username(conn, username)
        conn.close()
    except Exception as e:
        current_app.logger.exception("DB access error on login: %s", e)
        return jsonify({"message": "Database is unavailable"}), 503

    if not user:
        log_security_event("failed_login", username, request.remote_addr, "user_not_found")
        return jsonify({"message": "Invalid credentials"}), 401

    pw_ok = bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8'))
    _dbg(f"Password match={pw_ok}")
    if not pw_ok:
        log_security_event("failed_login", username, request.remote_addr, "wrong_password")
        return jsonify({"message": "Invalid credentials"}), 401

    # Successful login
    log_security_event("successful_login", username, request.remote_addr)

    # Generate tokens
    access_token = jwt.encode(
        {'username': username, 'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=15)},
        current_app.config['SECRET_KEY'],
        algorithm="HS256"
    )
    refresh_token = jwt.encode(
        {'username': username, 'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)},
        current_app.config['SECRET_KEY'],
        algorithm="HS256"
    )

    # Store refresh token
    try:
        conn = get_db_connection()
        RefreshToken.create_token(conn, username, refresh_token)
        conn.close()
    except Exception as e:
        current_app.logger.exception("DB write error on login: %s", e)
        return jsonify({"message": "Database is unavailable"}), 503

    _dbg('Tokens generated, storing refresh token and setting cookies')
    resp = make_response(jsonify({"message": "Login successful"}))
    add_cookie(resp, 'access_token', access_token, 900)                 # 15 min
    add_cookie(resp, 'refresh_token', refresh_token, 7*24*60*60)       # 7 days
    _dbg('=== LOGIN DEBUG END ===')
    return resp


@auth_bp.route("/refresh", methods=["POST"])
def refresh():
    """Token refresh endpoint"""
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        return jsonify({'message': 'Refresh token is missing'}), 400

    try:
        payload = jwt.decode(refresh_token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Refresh token expired'}), 401
    except Exception:
        return jsonify({'message': 'Invalid refresh token'}), 401

    username = payload.get("username", "")

    # Check if refresh token exists in DB
    try:
        conn = get_db_connection()
        if not RefreshToken.is_valid(conn, username, refresh_token):
            conn.close()
            return jsonify({"message": "Refresh token revoked"}), 401
        conn.close()
    except Exception as e:
        current_app.logger.exception("DB access error on refresh: %s", e)
        return jsonify({"message": "Database is unavailable"}), 503

    # Create new access token
    access_token = jwt.encode(
        {'username': username, 'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=15)},
        current_app.config['SECRET_KEY'],
        algorithm="HS256"
    )

    resp = make_response(jsonify({"message": "Token refreshed"}))
    add_cookie(resp, 'access_token', access_token, 900)
    return resp


@auth_bp.route("/logout", methods=["POST"])
@token_required
def logout(username):
    """User logout endpoint"""
    access_token = request.cookies.get("access_token")
    refresh_token = request.cookies.get("refresh_token")

    try:
        conn = get_db_connection()
        if access_token:
            BlacklistedToken.blacklist_token(conn, access_token)
        if refresh_token:
            RefreshToken.delete_token(conn, username, refresh_token)
        conn.close()
    except Exception as e:
        current_app.logger.exception("DB error on logout: %s", e)
        return jsonify({"message": "Database is unavailable"}), 503

    resp = make_response(jsonify({"message": "Logged out successfully"}))
    resp.delete_cookie("access_token", path="/")
    resp.delete_cookie("refresh_token", path="/")
    return resp


@auth_bp.route("/check-auth", methods=["GET"])
@token_required
def check_auth(username):
    """Check if user is authenticated"""
    return jsonify({"authenticated": True, "username": username}), 200


@auth_bp.route("/signup", methods=["POST"])
def signup():
    """User registration endpoint"""
    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    # Input validation
    is_valid, error_msg = validate_user_input(username, password)
    if not is_valid:
        return jsonify({"message": error_msg}), 400
    
    # Normalize username (prevent duplicates like "Admin" and "admin")
    username = username.lower()
    
    try:
        _dbg('=== SIGNUP DEBUG START ===')
        _dbg(f"remote_addr={request.remote_addr} username_hash={anonymize_username(username)}")
        conn = get_db_connection()
        _dbg('DB connection opened for signup')
        
        # Check if username already exists
        if User.find_by_username(conn, username):
            conn.close()
            return jsonify({"message": "Username already exists"}), 409
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Create new user
        if User.create_user(conn, username, password_hash.decode('utf-8')):
            conn.close()
            log_security_event("user_registered", username, request.remote_addr)
            _dbg('=== SIGNUP DEBUG END ===')
            return jsonify({"message": "User registered successfully"}), 201
        else:
            conn.close()
            return jsonify({"message": "Username already exists"}), 409
        
    except Exception as e:
        current_app.logger.exception("DB error on signup: %s", e)
        return jsonify({"message": "Database is unavailable"}), 503


@auth_bp.route("/admin/promote-user", methods=["POST"])
@token_required
def promote_user(username):
    """Promote a user to admin - ADMIN ONLY"""
    data = request.json or {}
    target_username = data.get('username', '').strip().lower()
    
    if not target_username:
        return jsonify({"message": "Username is required"}), 400
    
    try:
        conn = get_db_connection()
        
        # Security: Only admin users can promote others
        if not is_admin_user(conn, username):
            conn.close()
            log_security_event("unauthorized_access", username, request.remote_addr, "promote-user")
            return jsonify({"message": "Unauthorized - Admin access required"}), 403
        
        # Promote user
        if User.promote_to_admin(conn, target_username):
            conn.close()
            log_security_event("admin_action", username, request.remote_addr, f"promoted {target_username}")
            return jsonify({"message": f"User {target_username} promoted to admin"}), 200
        else:
            conn.close()
            return jsonify({"message": "User not found"}), 404
            
    except Exception as e:
        current_app.logger.exception("DB error on promote_user: %s", e)
        return jsonify({"message": "Database is unavailable"}), 503


@auth_bp.route("/admin/users", methods=["GET"])
@token_required
def get_users(username):
    """Get all users - ADMIN ONLY"""
    try:
        conn = get_db_connection()
        
        if not is_admin_user(conn, username):
            conn.close()
            log_security_event("unauthorized_access", username, request.remote_addr, "get-users")
            return jsonify({"message": "Unauthorized - Admin access required"}), 403
        
        users = User.get_all_users(conn)
        conn.close()
        return jsonify({"users": users}), 200
        
    except Exception as e:
        current_app.logger.exception("DB error on get_users: %s", e)
        return jsonify({"message": "Database is unavailable"}), 503


@auth_bp.route("/admin/create-first-admin", methods=["POST"])
def create_first_admin():
    """Create first admin user (only if no users exist)"""
    data = request.json or {}
    username = data.get('username', '').strip().lower()
    password = data.get('password', '')
    
    if not username or not password:
        return jsonify({"message": "Username and password required"}), 400
    
    try:
        conn = get_db_connection()
        
        # Only allow if no users exist yet
        if User.user_count(conn) > 0:
            conn.close()
            return jsonify({"message": "Admin already exists"}), 403
        
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        if User.create_user(conn, username, password_hash.decode('utf-8'), role="admin"):
            conn.close()
            return jsonify({"message": "First admin user created"}), 201
        else:
            conn.close()
            return jsonify({"message": "Failed to create admin user"}), 500
        
    except Exception as e:
        current_app.logger.exception("DB error creating first admin: %s", e)
        return jsonify({"message": "Database is unavailable"}), 503


@auth_bp.route("/db/diag", methods=["GET"])
@token_required
def db_diag(username):
    """Database diagnostics - ADMIN ONLY"""
    try:
        conn = get_db_connection()
        
        # Security: Only allow admin users to access diagnostic information
        if not is_admin_user(conn, username):
            conn.close()
            log_security_event("unauthorized_access", username, request.remote_addr, "db-diag")
            return jsonify({"message": "Unauthorized - Admin access required"}), 403
        
        conn.close()
    except Exception as e:
        current_app.logger.exception("DB access error in db_diag: %s", e)
        return jsonify({"message": "Database is unavailable"}), 503

    # Proceed with diagnostics for admin users only
    from .database import get_database_info
    info = get_database_info()
    
    log_security_event("admin_action", username, request.remote_addr, "accessed_db_diagnostics")
    return jsonify(info), 200
