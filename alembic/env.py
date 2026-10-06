"""Alembic environment — wired to the app's settings (reads the DB URL from .env).

Migrations use raw SQL (op.execute) so pgvector's custom `vector` type needs no
SQLAlchemy mapping; therefore target_metadata stays None (no autogenerate).
"""
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from alembic import context

# Make the `app` package importable when alembic runs from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import settings  # noqa: E402

config = context.config

# Build the SQLAlchemy URL from the app settings (psycopg v3 sync driver).
_sqlalchemy_url = settings.dsn.replace("postgresql://", "postgresql+psycopg://", 1)
config.set_main_option("sqlalchemy.url", _sqlalchemy_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Raw-SQL migrations -> no ORM metadata / no autogenerate.
target_metadata = None


def run_migrations_offline() -> None:
    context.configure(
        url=_sqlalchemy_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
