#database.py (MySQL version)
import os
from threading import Lock
import mysql.connector
from .models import get_table_definitions

import time

# Enable debug prints by setting AUTH_DEBUG=1 in environment (Railway Variables)
AUTH_DEBUG = os.getenv("AUTH_DEBUG", "0") == "1"

def _mask_secret(value: str, show: int = 1) -> str:
    """Mask secrets but still show if an unwanted prefix like ` or ' exists."""
    if value is None:
        return "None"
    value = str(value)
    if len(value) <= show * 2:
        return f"{value[0:1]}*** (len={len(value)})"
    return f"{value[:show]}***{value[-show:]} (len={len(value)})"

def _debug_print_db_config(context: str, host: str, port: int, db: str, user: str, password: str, timeout: int):
    if not AUTH_DEBUG:
        return
    print("===== DB DEBUG =====")
    print("Context:", context)
    print("DB_HOST:", host)
    print("DB_PORT:", port)
    print("DB_NAME:", db)
    print("DB_USER:", _mask_secret(user))
    # do NOT print password; show only masked to detect accidental quotes/backticks
    print("DB_PASSWORD:", _mask_secret(password))
    print("DB_CONNECT_TIMEOUT:", timeout)
    # highlight common mistakes
    if password and str(password).startswith(("`", "'", '"')):
        print("!! WARNING: DB_PASSWORD starts with a quote/backtick. Remove it in Railway Variables / .env.")
    if host in (None, "", "localhost", "127.0.0.1"):
        print("!! WARNING: DB_HOST looks like localhost. Check Railway Variables.")
    if db in (None, "", "AuthenticationDB"):
        print("!! WARNING: DB_NAME looks like default. Check Railway Variables.")
    print("====================")

def _read_db_env():
    """Read DB config from env every time (avoids stale globals)."""
    host = os.getenv("DB_HOST") or os.getenv("DB_SERVER") or "localhost"
    port = int(os.getenv("DB_PORT", "3306"))
    name = os.getenv("DB_NAME") or os.getenv("DB_DATABASE") or "AuthenticationDB"
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    timeout = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))
    return host, port, name, user, password, timeout


# Accept both names to avoid breaking your existing .env
DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_CONNECT_TIMEOUT = _read_db_env()

_db_init_done = False
_db_init_lock = Lock()


def get_db_connection():
    # Re-read env on every call (important in Railway/containers)
    host, port, name, user, password, timeout = _read_db_env()
    _debug_print_db_config("get_db_connection()", host, port, name, user, password, timeout)

    if not user or not password:
        raise RuntimeError("DB_USER/DB_PASSWORD are not set in the environment.")

    try:
        return mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=name,
            connection_timeout=timeout,
            autocommit=False,
        )
    except mysql.connector.Error as e:
        # Print safe error details
        if AUTH_DEBUG:
            print("!! mysql.connector.Error:", getattr(e, "errno", None), str(e))
        raise


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    tables = get_table_definitions()
    for _, sql in tables.items():
        cur.execute(sql)

    conn.commit()
    conn.close()


def ensure_database_initialized():
    global _db_init_done
    should_init = os.getenv("RUN_INIT_DB", "0") == "1"

    if should_init and not _db_init_done:
        with _db_init_lock:
            if not _db_init_done:
                try:
                    init_db()
                    print("✅ Database initialized successfully")
                    return True
                finally:
                    _db_init_done = True

    return _db_init_done


def test_database_connection():
    host, port, name, user, password, timeout = _read_db_env()
    _debug_print_db_config("test_database_connection()", host, port, name, user, password, timeout)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        row = cur.fetchone()
        conn.close()
        return (row is not None and row[0] == 1), "Database connection successful"
    except Exception as e:
        return False, f"Database connection failed: {str(e)}"


def get_database_info():
    """Return safe DB info for diagnostics (no password)."""
    host, port, name, user, password, timeout = _read_db_env()
    info = {
        "db_host": host,
        "db_port": port,
        "db_name": name,
        "db_user_masked": _mask_secret(user),
        "connect_timeout": timeout,
    }
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT DATABASE(), USER(), VERSION()")
        row = cur.fetchone()
        conn.close()
        if row:
            info["db_selected"] = row[0]
            info["db_user_server"] = row[1]
            info["db_version"] = row[2]
        info["connection_ok"] = True
    except Exception as e:
        info["connection_ok"] = False
        info["error"] = str(e)
    return info
