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
import pyodbc
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

    try:
        conn = get_db_connection()
        user = User.find_by_username(conn, username)
        conn.close()
    except Exception as e:
        current_app.logger.exception("DB access error on login: %s", e)
        return jsonify({"message": "Database is unavailable"}), 503

    if not user:
        log_security_event("failed_login", username, request.remote_addr, "user_not_found")
        return jsonify({"message": "Invalid credentials"}), 401

    if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
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

    resp = make_response(jsonify({"message": "Login successful"}))
    add_cookie(resp, 'access_token', access_token, 900)                 # 15 min
    add_cookie(resp, 'refresh_token', refresh_token, 7*24*60*60)       # 7 days
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
        return jsonify({'message': 'Refresh token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid refresh token'}), 401

    try:
        conn = get_db_connection()
        username = RefreshToken.find_by_token(conn, refresh_token)
        conn.close()
    except Exception as e:
        current_app.logger.exception("DB access error on refresh: %s", e)
        return jsonify({"message": "Database is unavailable"}), 503

    if not username:
        return jsonify({'message': 'Invalid refresh token'}), 401

    # Generate new access token
    new_access = jwt.encode(
        {'username': username, 'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=15)},
        current_app.config['SECRET_KEY'],
        algorithm="HS256"
    )

    resp = make_response(jsonify({'access_token': new_access}))
    add_cookie(resp, 'access_token', new_access, 900)
    return resp


@auth_bp.route("/logout", methods=["POST"])
@token_required
def logout(username):
    """User logout endpoint"""
    token = request.cookies.get('access_token')
    if not token:
        return jsonify({"message": "Invalid token format"}), 401

    try:
        conn = get_db_connection()
        
        # Add to blacklist
        BlacklistedToken.add_to_blacklist(conn, token)
        
        # Delete refresh tokens
        RefreshToken.delete_user_tokens(conn, username)
        
        conn.close()
        
        log_security_event("logout", username, request.remote_addr)
        
    except Exception as e:
        current_app.logger.exception("DB write error on logout: %s", e)
        return jsonify({"message": "Database is unavailable"}), 503

    resp = make_response(jsonify({"message": "Logged out successfully!"}))
    resp.delete_cookie('access_token', path='/')
    resp.delete_cookie('refresh_token', path='/')
    return resp


@auth_bp.route("/check-auth", methods=["GET"])
@token_required
def check_auth(username):
    """Check authentication status"""
    return jsonify({"message": "Authenticated", "username": username}), 200


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
        conn = get_db_connection()
        
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
    """Promote a user to admin role - ADMIN ONLY"""
    try:
        conn = get_db_connection()
        
        # Check if current user is admin
        if not is_admin_user(conn, username):
            conn.close()
            log_security_event("unauthorized_access", username, request.remote_addr, "promote-user")
            return jsonify({"message": "Unauthorized - Admin access required"}), 403
        
        # Get target username from request
        data = request.json or {}
        target_user = data.get('username', '').strip().lower()
        
        if not target_user:
            conn.close()
            return jsonify({"message": "Username is required"}), 400
        
        # Check if target user exists
        target_user_obj = User.find_by_username(conn, target_user)
        if not target_user_obj:
            conn.close()
            return jsonify({"message": "User not found"}), 404
        
        if target_user_obj.role == 'admin':
            conn.close()
            return jsonify({"message": "User is already an admin"}), 400
        
        # Promote user to admin
        if User.promote_to_admin(conn, target_user):
            conn.close()
            log_security_event("user_promoted", username, request.remote_addr, f"promoted {target_user}")
            return jsonify({"message": f"User {target_user} promoted to admin successfully"}), 200
        else:
            conn.close()
            return jsonify({"message": "Failed to promote user"}), 500
        
    except Exception as e:
        current_app.logger.exception("DB error in promote-user: %s", e)
        return jsonify({"message": "Database is unavailable"}), 503


@auth_bp.route("/admin/users", methods=["GET"])
@token_required
def list_users(username):
    """List all users - ADMIN ONLY"""
    try:
        conn = get_db_connection()
        
        # Check if current user is admin
        if not is_admin_user(conn, username):
            conn.close()
            log_security_event("unauthorized_access", username, request.remote_addr, "list-users")
            return jsonify({"message": "Unauthorized - Admin access required"}), 403
        
        # Get all users
        users = User.get_all_users(conn)
        conn.close()
        
        log_security_event("admin_action", username, request.remote_addr, "viewed_user_list")
        return jsonify({"users": users, "total": len(users)}), 200
        
    except Exception as e:
        current_app.logger.exception("DB error in list-users: %s", e)
        return jsonify({"message": "Database is unavailable"}), 503


@auth_bp.route("/admin/create-first-admin", methods=["POST"])
def create_first_admin():
    """Create the first admin user - ONLY if no users exist"""
    try:
        conn = get_db_connection()
        
        # Check if any users exist
        if User.user_count(conn) > 0:
            conn.close()
            return jsonify({"message": "Users already exist. Cannot create first admin."}), 409
        
        # Create first admin user
        username = "admin"
        password = "admin123"  # Should be changed immediately
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Create admin user
        if User.create_user(conn, username, password_hash.decode('utf-8'), 'admin'):
            conn.close()
            log_security_event("first_admin_created", "system", request.remote_addr)
            return jsonify({
                "message": "First admin user created successfully",
                "username": "admin",
                "password": "admin123",
                "warning": "CHANGE THE PASSWORD IMMEDIATELY!"
            }), 201
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