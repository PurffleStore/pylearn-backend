"""
Database models and schemas for authentication system

Contains:
- User model with role-based access
- Token blacklist model
- Refresh token model  
- Database table definitions
"""

import pyodbc
from typing import Optional, Dict, Any


class User:
    """User model for authentication and authorization"""
    
    def __init__(self, username: str, password_hash: str, role: str = 'user', user_id: int = None):
        self.id = user_id
        self.username = username
        self.password_hash = password_hash
        self.role = role
    
    @staticmethod
    def find_by_username(conn: pyodbc.Connection, username: str) -> Optional['User']:
        """Find user by username"""
        cur = conn.cursor()
        cur.execute("SELECT id, username, password_hash, role FROM Users WHERE username = ?", (username,))
        row = cur.fetchone()
        if row:
            return User(
                user_id=row[0],
                username=row[1], 
                password_hash=row[2],
                role=row[3]
            )
        return None
    
    @staticmethod
    def create_user(conn: pyodbc.Connection, username: str, password_hash: str, role: str = 'user') -> bool:
        """Create a new user"""
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO Users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, password_hash, role)
            )
            conn.commit()
            return True
        except pyodbc.IntegrityError:
            return False
    
    @staticmethod
    def get_all_users(conn: pyodbc.Connection) -> list:
        """Get all users (admin only)"""
        cur = conn.cursor()
        cur.execute("SELECT id, username, role FROM Users ORDER BY id")
        users = []
        for row in cur.fetchall():
            users.append({
                "id": row[0],
                "username": row[1], 
                "role": row[2]
            })
        return users
    
    @staticmethod
    def promote_to_admin(conn: pyodbc.Connection, username: str) -> bool:
        """Promote user to admin role"""
        cur = conn.cursor()
        cur.execute("UPDATE Users SET role = 'admin' WHERE username = ?", (username,))
        conn.commit()
        return cur.rowcount > 0
    
    @staticmethod
    def user_count(conn: pyodbc.Connection) -> int:
        """Get total user count"""
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Users")
        return cur.fetchone()[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (safe for JSON)"""
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role
            # Note: Never include password_hash in dict
        }


class BlacklistedToken:
    """Model for blacklisted JWT tokens"""
    
    @staticmethod
    def is_blacklisted(conn: pyodbc.Connection, token: str) -> bool:
        """Check if token is blacklisted"""
        cur = conn.cursor()
        cur.execute("SELECT token FROM BlacklistedTokens WHERE token = ?", (token,))
        return cur.fetchone() is not None
    
    @staticmethod 
    def add_to_blacklist(conn: pyodbc.Connection, token: str) -> bool:
        """Add token to blacklist"""
        cur = conn.cursor()
        # Check if already blacklisted
        cur.execute("SELECT token FROM BlacklistedTokens WHERE token = ?", (token,))
        if cur.fetchone():
            return True  # Already blacklisted
        
        cur.execute("INSERT INTO BlacklistedTokens (token) VALUES (?)", (token,))
        conn.commit()
        return True


class RefreshToken:
    """Model for refresh token management"""
    
    @staticmethod
    def find_by_token(conn: pyodbc.Connection, token: str) -> Optional[str]:
        """Find username by refresh token"""
        cur = conn.cursor()
        cur.execute("SELECT username FROM RefreshTokens WHERE token = ?", (token,))
        row = cur.fetchone()
        return row[0] if row else None
    
    @staticmethod
    def create_token(conn: pyodbc.Connection, username: str, token: str) -> bool:
        """Store refresh token"""
        cur = conn.cursor()
        cur.execute("INSERT INTO RefreshTokens (username, token) VALUES (?, ?)", (username, token))
        conn.commit()
        return True
    
    @staticmethod
    def delete_user_tokens(conn: pyodbc.Connection, username: str) -> bool:
        """Delete all refresh tokens for user"""
        cur = conn.cursor()
        cur.execute("DELETE FROM RefreshTokens WHERE username = ?", (username,))
        conn.commit()
        return True


# Database table creation SQL
def get_table_definitions():
    """Get SQL statements for creating authentication tables"""
    return {
        'users': """
            IF OBJECT_ID('Users', 'U') IS NULL
            CREATE TABLE Users (
                id INT IDENTITY(1,1) PRIMARY KEY,
                username NVARCHAR(100) UNIQUE NOT NULL,
                password_hash NVARCHAR(500) NOT NULL,
                role NVARCHAR(50) DEFAULT 'user'
            )
        """,
        
        'blacklisted_tokens': """
            IF OBJECT_ID('BlacklistedTokens', 'U') IS NULL
            CREATE TABLE BlacklistedTokens (
                id INT IDENTITY(1,1) PRIMARY KEY,
                token NVARCHAR(1000) UNIQUE NOT NULL,
                created_at DATETIME DEFAULT GETDATE()
            )
        """,
        
        'refresh_tokens': """
            IF OBJECT_ID('RefreshTokens', 'U') IS NULL
            CREATE TABLE RefreshTokens (
                id INT IDENTITY(1,1) PRIMARY KEY,
                username NVARCHAR(100) NOT NULL,
                token NVARCHAR(1000) UNIQUE NOT NULL,
                created_at DATETIME DEFAULT GETDATE(),
                FOREIGN KEY (username) REFERENCES Users(username) ON DELETE CASCADE
            )
        """
    }