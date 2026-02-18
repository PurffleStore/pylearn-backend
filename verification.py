"""
MJ Learn Backend - Main Flask Application

A clean, professional Flask application with modular authentication.

Main Features:
- JWT-based authentication system
- Role-based access control (admin/user)
- Secure token management with blacklisting
- CORS configuration for cross-origin requests
- Modular blueprint architecture
- Environment-based configuration
"""

import os
from dotenv import load_dotenv
import logging
from flask import Flask, request
from flask_cors import CORS

# Load environment variables first
BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, ".env"))

# --- Build local ChromaDB at startup (expects build_chroma_db.py in same folder) ---
_CHROMA_SCRIPT_PATH = os.path.join(BASEDIR, "build_chroma_db.py")

if os.path.exists(_CHROMA_SCRIPT_PATH):
    try:
        import importlib.util
        import traceback

        spec = importlib.util.spec_from_file_location("build_chroma_db_local", _CHROMA_SCRIPT_PATH)
        build_chroma_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_chroma_mod)

        if hasattr(build_chroma_mod, "build_chroma"):
            # Run the builder to create Chroma DB in the local assets folder
            build_chroma_mod.build_chroma()
            print("✅ build_chroma_db.build_chroma() executed successfully.")
        else:
            print("!! build_chroma_db.py found but no `build_chroma()` function present.")
    except Exception as e:
        print(f"!! Failed to run build_chroma_db.py: {e}")
        traceback.print_exc()
else:
    print("!! build_chroma_db.py not found in the application folder — skipping Chroma build.")
# --- End ChromaDB build block ---

# Import authentication module
from auth import auth_bp
from auth.database import ensure_database_initialized


def create_app():
    """Application factory pattern for Flask app creation"""
    app = Flask(__name__)
    
    # Security configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    if not app.config['SECRET_KEY']:
        raise RuntimeError("SECRET_KEY must be set in environment variables!")
    
    # Environment configuration
    IS_PROD = os.getenv("ENV", "dev").lower() == "prod"
    
    # CORS configuration
    _default_origins = "http://localhost:4200,http://127.0.0.1:4200"
    _origins = os.getenv("ALLOWED_ORIGINS", _default_origins)
    ALLOWED_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()]
    
    CORS(
        app,
        resources={r"/*": {"origins": ALLOWED_ORIGINS}},
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization", "X-Requested-With", "X-User"],
        expose_headers=["Set-Cookie"],
        methods=["GET", "POST", "OPTIONS"]
    )
    
    # API configuration for blueprints
    app.config["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY", "")
    
    # CORS handlers
    @app.after_request
    def add_cors_headers(resp):
        origin = request.headers.get("Origin")
        if origin and origin in ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Credentials"] = "true"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, X-User"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp
    
    @app.before_request
    def handle_options_early():
        if request.method == "OPTIONS":
            resp = app.make_default_options_response()
            origin = request.headers.get("Origin")
            if origin and origin in ALLOWED_ORIGINS:
                resp.headers["Access-Control-Allow-Origin"] = origin
                resp.headers["Access-Control-Allow-Credentials"] = "true"
            # Mirror browser's requested headers/methods
            req_headers = request.headers.get("Access-Control-Request-Headers", "Content-Type, Authorization, X-Requested-With, X-User")
            req_method = request.headers.get("Access-Control-Request-Method", "POST")
            resp.headers["Access-Control-Allow-Headers"] = req_headers
            resp.headers["Access-Control-Allow-Methods"] = req_method
            return resp
    
    # Initialize database before first request (Flask 3.x compatible)
    @app.before_request
    def maybe_initialize_database():
        if not hasattr(app, '_db_initialized'):
            try:
                ensure_database_initialized()
                app._db_initialized = True
            except Exception as e:
                app.logger.exception("Database initialization failed: %s", e)
    
    # Health check endpoint
    @app.route("/")
    def health():
        return {"status": "ok", "service": "MJ Learn Backend"}, 200
    
    # Register authentication blueprint
    app.register_blueprint(auth_bp, url_prefix="/auth")
    
    # Register other feature blueprints
    register_feature_blueprints(app)
    
    return app


def register_feature_blueprints(app):
    """Register feature blueprints with error handling"""
    blueprints = [       
        ("ragg.app", "rag_bp", "/rag"),
        ("pronunciation", "pronunciation_bp", "/pronunciation"), 
        ("staticchat", "staticchat_bp", "/staticchat"),   
        ("chat_llm", "chat_llm_bp", "/chat_llm"),        
        ("ragg.ingest_trigger", "ingest_trigger_bp", "/rag"),
    ]
    
    for module_name, blueprint_name, url_prefix in blueprints:
        try:
            module = __import__(module_name, fromlist=[blueprint_name])
            blueprint = getattr(module, blueprint_name)
            app.register_blueprint(blueprint, url_prefix=url_prefix)
            print(f"? Registered {blueprint_name}")
        except ImportError as e:
            print(f"?? Could not import {blueprint_name}: {e}")
        except AttributeError as e:
            print(f"?? Blueprint {blueprint_name} not found in {module_name}: {e}")


# Create Flask app instance
app = create_app()

# Configure logging
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    print("?? Starting MJ Learn Backend...")
    print(f"? SECRET_KEY loaded: {bool(app.config.get('SECRET_KEY'))}")
    print(f"? Environment: {os.getenv('ENV', 'development')}")
    print("=" * 60)
    
    port = int(os.getenv("PORT", "5000"))
    print(f"?? Server starting on http://localhost:{port}")
    print("?? Available endpoints:")
    print("   GET  /                    - Health check")
    print("   ?? Authentication:")
    print("   POST /auth/signup         - User registration")
    print("   POST /auth/login          - User login")
    print("   POST /auth/refresh        - Token refresh")
    print("   POST /auth/logout         - User logout")
    print("   GET  /auth/dashboard      - Protected endpoint")
    print("   GET  /auth/check-auth     - Auth status check")
    print("   GET  /auth/db/diag        - Database diagnostics (ADMIN)")
    print("   ?? Admin Management:")
    print("   GET  /auth/admin/users          - List all users (ADMIN)")
    print("   POST /auth/admin/promote-user   - Promote user to admin (ADMIN)")  
    print("   POST /auth/admin/create-first-admin - Create first admin")
    print("=" * 60)
    
    try:
        app.run(host="0.0.0.0", port=port, debug=True)
    except Exception as e:
        print(f"? Failed to start server: {e}")
        raise