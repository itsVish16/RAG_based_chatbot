from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from typing import AsyncGenerator

from app.config.settings import settings

class Base(DeclarativeBase):
    pass

engine = create_async_engine(
    settings.POSTGRES_URL,
    echo = settings.DEBUG if hasstr(settings,'DEBUG') else False,
    pool_pre_ping = True,
    pool_size = 10,
    max_overflow = 20,
    pool_recycle = 3600,
)

AsyncSessionLocal = async_sessionmaker(
    bind = engine,
    class_ = AsyncSession,
    expire_on_commit = False,
    autoflush = False,
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:

    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)