"""
数据库连接工具
提供 PostgreSQL Session 和引擎管理
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator

from ..config import settings


_engine = None
_SessionLocal = None


def get_engine():
    """获取或创建数据库引擎（全局单例）"""
    global _engine
    if _engine is None:
        db_url = settings.postgres_checkpointer_url
        if not db_url:
            raise ValueError("未配置 postgres_checkpointer_url，无法创建数据库引擎")
        _engine = create_engine(
            db_url,
            connect_timeout=10,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory():
    """获取或创建 SessionFactory（全局单例）"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    return _SessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    上下文管理器，获得数据库会话。
    自动处理 commit/rollback 和关闭。

    用法：
        with get_session() as db:
            db.add(obj)
            db.commit()
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_workflow_db():
    """
    初始化工作流数据库表（在应用启动时调用一次）。
    如果表已存在则跳过。
    """
    from .models import Base
    engine = get_engine()
    Base.metadata.create_all(bind=engine, checkfirst=True)
