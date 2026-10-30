from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

def _normalize_db_url(url: str) -> str:
    """Force the psycopg2 driver for bare postgresql:// URLs.

    SQLAlchemy 2.1 changed the default driver for ``postgresql://`` from
    psycopg2 to psycopg (v3). This project pins ``psycopg2-binary`` in
    pyproject.toml and does not install psycopg v3, so a bare URL raises
    ``ModuleNotFoundError: No module named 'psycopg'`` on SQLAlchemy >= 2.1
    while working fine on 2.0.x. Naming the driver explicitly makes the code
    independent of that default.
    """
    if url.startswith("postgresql://"):
        return "postgresql+psycopg2://" + url[len("postgresql://"):]
    if url.startswith("postgres://"):
        return "postgresql+psycopg2://" + url[len("postgres://"):]
    return url


# Prefer explicit env override; default to service name for containerized deployment
# Default to docker-compose service name; override via env when needed
DATABASE_URL = _normalize_db_url(os.environ.get(
    'DATABASE_URL', "postgresql+psycopg2://alpha_user:alpha_pass@postgres:5432/alpha_arena"))

# Allow tuning via environment variables but provide sensible defaults for our workload
POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", "20"))
POOL_MAX_OVERFLOW = int(os.environ.get("DB_POOL_MAX_OVERFLOW", "20"))
POOL_RECYCLE = int(os.environ.get("DB_POOL_RECYCLE", "1800"))  # seconds
POOL_TIMEOUT = int(os.environ.get("DB_POOL_TIMEOUT", "30"))

engine = create_engine(
    DATABASE_URL,
    pool_size=POOL_SIZE,
    max_overflow=POOL_MAX_OVERFLOW,
    pool_recycle=POOL_RECYCLE,
    pool_timeout=POOL_TIMEOUT,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()