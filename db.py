from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

LOCAL_DATABASE_URL = "sqlite:///./app.db"

# SQLite needs thread-safe settings for FastAPI and a higher timeout for concurrent writes.
engine = create_engine(
    LOCAL_DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA busy_timeout=30000;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    try:
        # WAL can fail if another process is holding a write lock; keep startup resilient.
        cursor.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_remote_engine(database_url: str):
    return create_engine(database_url, pool_pre_ping=True)
