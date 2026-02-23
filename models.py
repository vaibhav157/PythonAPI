from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)


class PromptConfig(Base):
    __tablename__ = "prompt_configs"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, nullable=False, index=True)
    system_prompt = Column(Text, nullable=False)
    user_prompt = Column(Text, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class LocalRecord(Base):
    __tablename__ = "local_records"

    id = Column(Integer, primary_key=True, index=True)
    source_record_id = Column(String, index=True, nullable=True)
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    matches = relationship("MatchEvaluation", back_populates="local_record", cascade="all, delete-orphan")


class MatchEvaluation(Base):
    __tablename__ = "match_evaluations"

    id = Column(Integer, primary_key=True, index=True)
    local_record_id = Column(Integer, ForeignKey("local_records.id"), nullable=False, index=True)
    remote_record_id = Column(String, nullable=True, index=True)
    remote_payload = Column(JSON, nullable=False)
    match_percentage = Column(Float, nullable=False)
    reasoning = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    local_record = relationship("LocalRecord", back_populates="matches")
