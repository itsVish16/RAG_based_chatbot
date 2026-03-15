import logging
import jwt
import bcrypt  # ⬅️ Using directly to avoid passlib bugs
from datetime import datetime, timedelta
from typing import Optional
from app.config.settings import settings

logger = logging.getLogger(__name__)

class AuthService:
    """
    Handles secure Password Hashing and JWT token issuance using direct bcrypt.
    """

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password safely inside UTF-8 bytes."""
        # Bcrypt requires bytes
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')  # Store as string in DB

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against a stored hashed string."""
        try:
            return bcrypt.checkpw(
                plain_password.encode('utf-8'), 
                hashed_password.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Password verification crash: {e}")
            return False

    @staticmethod
    def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """
        Generates a secure JWT access token with user_id as payload 'sub'.
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode = {
            "sub": str(user_id),
            "exp": expire,
            "iat": datetime.utcnow()
        }
        
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt

# Singleton
auth_service = AuthService()
