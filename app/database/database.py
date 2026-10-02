from sqlalchemy.orm import sessionmaker
from app.core.config import setting

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

DB_URL = setting.DB_URL

# Async DB URL format for MySQL:
engine = create_async_engine(DB_URL, echo=True, future=True)

# Create async session maker
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)
