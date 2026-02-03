# models.py (MySQL version)
from typing import Optional, Dict, Any
from mysql.connector.errors import IntegrityError


class User:
    def __init__(self, username: str, password_hash: str, role: str = "user", user_id: int = None):
        self.id = user_id
        self.username = username
        self.password_hash = password_hash
        self.role = role

    @staticmethod
    def find_by_username(conn, username: str) -> Optional["User"]:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, password_hash, role FROM users WHERE username = %s",
            (username,),
        )
        row = cur.fetchone()
        if row:
            return User(user_id=row[0], username=row[1], password_hash=row[2], role=row[3])
        return None

    @staticmethod
    def create_user(conn, username: str, password_hash: str, role: str = "user") -> bool:
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                (username, password_hash, role),
            )
            conn.commit()
            return True
        except IntegrityError:
            return False

    @staticmethod
    def get_all_users(conn) -> list:
        cur = conn.cursor()
        cur.execute("SELECT id, username, role FROM users ORDER BY id")
        users = []
        for row in cur.fetchall():
            users.append({"id": row[0], "username": row[1], "role": row[2]})
        return users

    @staticmethod
    def promote_to_admin(conn, username: str) -> bool:
        cur = conn.cursor()
        cur.execute("UPDATE users SET role = %s WHERE username = %s", ("admin", username))
        conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def user_count(conn) -> int:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        return cur.fetchone()[0]

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "username": self.username, "role": self.role}


class BlacklistedToken:
    @staticmethod
    def is_blacklisted(conn, token: str) -> bool:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM blacklisted_tokens WHERE token = %s LIMIT 1", (token,))
        return cur.fetchone() is not None

    @staticmethod
    def add_to_blacklist(conn, token: str) -> bool:
        cur = conn.cursor()
        # Insert and ignore duplicates
        cur.execute("INSERT IGNORE INTO blacklisted_tokens (token) VALUES (%s)", (token,))
        conn.commit()
        return True


class RefreshToken:
    @staticmethod
    def find_by_token(conn, token: str) -> Optional[str]:
        cur = conn.cursor()
        cur.execute("SELECT username FROM refresh_tokens WHERE token = %s", (token,))
        row = cur.fetchone()
        return row[0] if row else None

    @staticmethod
    def create_token(conn, username: str, token: str) -> bool:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO refresh_tokens (username, token) VALUES (%s, %s)",
            (username, token),
        )
        conn.commit()
        return True

    @staticmethod
    def delete_user_tokens(conn, username: str) -> bool:
        cur = conn.cursor()
        cur.execute("DELETE FROM refresh_tokens WHERE username = %s", (username,))
        conn.commit()
        return True


def get_table_definitions():
    return {
        "users": """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(50) NOT NULL DEFAULT 'user'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        "blacklisted_tokens": """
            CREATE TABLE IF NOT EXISTS blacklisted_tokens (
                id INT AUTO_INCREMENT PRIMARY KEY,
                token VARCHAR(1000) NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        "refresh_tokens": """
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) NOT NULL,
                token VARCHAR(1000) NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
    }
