from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.adapters.outbound.persistence.orm import metadata


def build_engine(url: str) -> Engine:
    """Postgres en Docker; SQLite para desarrollo y tests (en memoria si url == 'sqlite://')."""
    if url.startswith("sqlite"):
        kwargs: dict = {"connect_args": {"check_same_thread": False}}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
        return create_engine(url, **kwargs)
    return create_engine(url, pool_pre_ping=True)


def build_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def create_schema(engine: Engine) -> None:
    # v1 sin migraciones: las tablas se crean al arrancar (idempotente).
    metadata.create_all(engine)


def ping(engine) -> None:
    """Lanza la excepción del driver si la base no responde."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
