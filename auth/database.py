# database.py (MySQL version)
import os
from threading import Lock
import mysql.connector
from .models import get_table_definitions

# Accept both names to avoid breaking your existing .env
DB_HOST = os.getenv("DB_HOST") or os.getenv("DB_SERVER") or "localhost"
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME") or os.getenv("DB_DATABASE") or "AuthenticationDB"
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_CONNECT_TIMEOUT = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))

_db_init_done = False
_db_init_lock = Lock()


def get_db_connection():
    if not DB_USER or not DB_PASSWORD:
        raise RuntimeError("DB_USER/DB_PASSWORD are not set in the environment.")

    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        connection_timeout=DB_CONNECT_TIMEOUT,
        autocommit=False,
    )


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
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        row = cur.fetchone()
        conn.close()
        return (row is not None and row[0] == 1), "Database connection successful"
    except Exception as e:
        return False, f"Database connection failed: {str(e)}"
