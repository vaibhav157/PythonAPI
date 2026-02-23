from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

LOCAL_DATABASE_URL = "sqlite:///./app.db"

# SQLite needs this flag when used with FastAPI threads.
engine = create_engine(LOCAL_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_remote_engine(database_url: str):
    return create_engine(database_url, pool_pre_ping=True)
