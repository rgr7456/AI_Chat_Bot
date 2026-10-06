"""PostgreSQL connection pool with pgvector support (synchronous psycopg3).

DB calls are run inside FastAPI's threadpool by the service layer, so a
synchronous pool keeps the code simple while staying off the event loop.
"""
from __future__ import annotations

from typing import Optional

from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from app.core.config import settings

_pool: Optional[ConnectionPool] = None


def _configure(conn) -> None:
    # pgvector adapters must be registered per connection.
    register_vector(conn)


def init_pool() -> ConnectionPool:
    """Create the pool (idempotent). Call once at startup."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.dsn,
            min_size=1,
            max_size=10,
            configure=_configure,
            open=True,
            kwargs={"application_name": "hrms-face-service"},
        )
    return _pool


def get_pool() -> ConnectionPool:
    if _pool is None:
        return init_pool()
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
