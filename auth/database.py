"""
Database connection and initialization module

Handles:
- Database connection management
- Table creation and initialization
- Connection string configuration
- Database diagnostics
"""

import os
import pyodbc
from threading import Lock
from .models import get_table_definitions

# Database configuration
DB_SERVER = os.getenv("DB_SERVER", r"(localdb)\MSSQLLocalDB")
DB_DATABASE = os.getenv("DB_DATABASE", "AuthenticationDB1")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")

# Build connection string
is_local = (
    DB_SERVER.lower().startswith("localhost")
    or DB_SERVER.startswith(".")
    or DB_SERVER.lower().startswith("(localdb)")
    or "\\" in DB_SERVER
)

if is_local:
    # Windows local / LocalDB using modern ODBC driver
    CONN_STR = (
        f"DRIVER={{{DB_DRIVER}}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_DATABASE};"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
else:
    # Remote SQL auth
    CONN_STR = (
        f"DRIVER={{{DB_DRIVER}}};"
        f"SERVER={DB_SERVER};DATABASE={DB_DATABASE};"
        f"UID={os.getenv('DB_USER')};PWD={os.getenv('DB_PASSWORD')};"
        "Encrypt=yes;TrustServerCertificate=yes;"
    )

# Database initialization tracking
_db_init_done = False
_db_init_lock = Lock()


def get_db_connection():
    """
    Create a database connection with short timeout
    
    Raises:
        RuntimeError: If DB credentials are missing for remote connections
        pyodbc.Error: If connection fails
    """
    if "Trusted_Connection=yes" not in CONN_STR:
        if not os.getenv("DB_USER") or not os.getenv("DB_PASSWORD"):
            raise RuntimeError("DB_USER/DB_PASSWORD are not set in the environment.")
    return pyodbc.connect(CONN_STR, timeout=5)


def init_db():
    """
    Create database tables if they do not exist
    
    Creates:
    - Users table for authentication
    - BlacklistedTokens table for token management  
    - RefreshTokens table for refresh token storage
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get table definitions
    tables = get_table_definitions()
    
    # Create each table
    for table_name, sql in tables.items():
        cur.execute(sql)
    
    conn.commit()
    conn.close()


def ensure_database_initialized():
    """
    Ensure database is initialized (thread-safe)
    
    Call this from Flask app startup to initialize database once.
    Controlled by RUN_INIT_DB environment variable.
    """
    global _db_init_done
    should_init = os.getenv("RUN_INIT_DB", "0") == "1"
    
    if should_init and not _db_init_done:
        with _db_init_lock:
            if not _db_init_done:
                try:
                    init_db()
                    print("? Database initialized successfully")
                    return True
                except Exception as e:
                    print(f"? Database initialization failed: {e}")
                    raise
                finally:
                    _db_init_done = True
    
    return _db_init_done


def get_database_info():
    """
    Get database diagnostic information (admin only)
    
    Returns safe diagnostic information without exposing credentials.
    """
    info = {}
    
    # Get available drivers
    try:
        info["drivers_found"] = pyodbc.drivers()
    except Exception as e:
        info["drivers_found_error"] = str(e)

    # Safe database information
    info["database_name"] = DB_DATABASE
    info["server_type"] = "LocalDB" if is_local else "Remote"
    
    # Test connection
    try:
        conn = get_db_connection()
        conn.close()
        info["connection_status"] = "ok"
    except Exception as e:
        info["connection_status"] = "error"
        info["error_type"] = type(e).__name__
    
    return info


def test_database_connection():
    """
    Test database connection and return status
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        conn = get_db_connection()
        
        # Test basic query
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        
        conn.close()
        
        if result and result[0] == 1:
            return True, "Database connection successful"
        else:
            return False, "Database query failed"
            
    except Exception as e:
        return False, f"Database connection failed: {str(e)}"