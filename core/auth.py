from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any
from passlib.context import CryptContext
from jose import jwt, JWTError
from uuid import uuid4
from .domain import User
from .ftypes import Either

# Конфигурация (в реальном проекте вынести в .env)
SECRET_KEY = "your-super-secret-key-change-it"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Чистая функция проверки пароля."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Чистая функция хеширования."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Создание JWT токена."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def register_user(
    user_data: Dict[str, str], 
    existing_users: Tuple[User, ...]
) -> Either[str, User]:
    """
    Функциональная логика регистрации.
    Принимает данные и текущий список пользователей.
    Возвращает Either.Left(ошибка) или Either.Right(новый пользователь).
    """
    email = user_data.get("email")
    password = user_data.get("password")
    name = user_data.get("name")

    if not email or not password or not name:
        return Either.left("All fields are required")

    # Проверка на существование (FP style)
    if any(u.email == email for u in existing_users):
        return Either.left("Email already registered")

    new_user = User(
        id=f"user_{uuid4().hex[:8]}",
        email=email,
        password_hash=get_password_hash(password),
        name=name
    )
    return Either.right(new_user)

def authenticate_user(
    credentials: Dict[str, str],
    users: Tuple[User, ...]
) -> Either[str, User]:
    """Функциональная логика аутентификации."""
    email = credentials.get("email")
    password = credentials.get("password")

    # Поиск пользователя (FP style)
    user = next((u for u in users if u.email == email), None)

    if not user:
        return Either.left("User not found")
    
    if not password:
        return Either.left("Password is required")
    
    if not verify_password(password, user.password_hash):
        return Either.left("Incorrect password")
    
    return Either.right(user)

def get_user_from_token(token: str, users: Tuple[User, ...]) -> Optional[User]:
    """Извлечение пользователя из токена (безопасно)."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return next((u for u in users if u.email == email), None)
    except JWTError:
        return None