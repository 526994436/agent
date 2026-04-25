"""
工作流数据模型
使用 SQLAlchemy ORM 定义 PostgreSQL 表结构
"""

import enum
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Column, String, DateTime, Text, JSON, Integer,
    Enum, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class OnboardingState(str, enum.Enum):
    """
    入职工作流状态枚举。

    状态流转：
        OFFER_PENDING  →  offer_confirmed  →  OFFER_CONFIRMED
        OFFER_CONFIRMED  →  first_checkin  →  FIRST_CHECKIN
        FIRST_CHECKIN  →  (自动结束)  →  COMPLETED
        任意状态遇到不可恢复错误 → FAILED
    """
    OFFER_PENDING    = "offer_pending"     # 等待候选人确认 offer
    OFFER_CONFIRMED  = "offer_confirmed"   # offer 已确认，账号开通 + 设备申请
    FIRST_CHECKIN    = "first_checkin"     # 入职打卡，分配培训课程
    COMPLETED        = "completed"         # 全部完成
    FAILED           = "failed"            # 中途失败


class OnboardingInstance(Base):
    """
    入职工作流实例。

    记录一次完整入职流程的所有上下文，包括候选人信息、
    当前状态、各步骤完成时间戳和关联的 Java 后端 request_id。
    """
    __tablename__ = "onboarding_instances"

    # 主键
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # 候选人/员工基本信息
    candidate_name = Column(String, nullable=False)
    department     = Column(String)
    position       = Column(String)
    entry_date     = Column(String)   # YYYY-MM-DD
    salary         = Column(String)
    employee_id    = Column(String)
    email          = Column(String)
    device_type    = Column(String)
    course_list    = Column(JSON, default=list)

    # 当前状态
    state = Column(
        Enum(OnboardingState),
        default=OnboardingState.OFFER_PENDING,
        nullable=False,
        index=True,
    )
    current_step_detail = Column(Text, nullable=True)

    # 各步骤时间戳
    offer_sent_at          = Column(DateTime, nullable=True)
    offer_confirmed_at     = Column(DateTime, nullable=True)
    account_created_at     = Column(DateTime, nullable=True)
    equipment_applied_at   = Column(DateTime, nullable=True)
    checkin_at             = Column(DateTime, nullable=True)
    training_enrolled_at   = Column(DateTime, nullable=True)
    completed_at           = Column(DateTime, nullable=True)
    failed_at              = Column(DateTime, nullable=True)

    # Java 后端 request_id 追踪（每个原子步骤一个）
    offer_request_id     = Column(String, nullable=True)
    account_request_id   = Column(String, nullable=True)
    equipment_request_id = Column(String, nullable=True)
    training_request_id  = Column(String, nullable=True)

    # 元数据
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # 关联事件
    events = relationship("OnboardingEvent", back_populates="instance", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_onboarding_state_created", "state", "created_at"),
        Index("ix_onboarding_employee_id", "employee_id"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "candidate_name": self.candidate_name,
            "department": self.department,
            "position": self.position,
            "entry_date": self.entry_date,
            "salary": self.salary,
            "employee_id": self.employee_id,
            "email": self.email,
            "device_type": self.device_type,
            "course_list": self.course_list or [],
            "state": self.state.value if self.state else None,
            "current_step_detail": self.current_step_detail,
            "steps": [
                {"step": "offer_sent",          "at": self.offer_sent_at,          "request_id": self.offer_request_id},
                {"step": "offer_confirmed",      "at": self.offer_confirmed_at,     "request_id": None},
                {"step": "account_created",     "at": self.account_created_at,     "request_id": self.account_request_id},
                {"step": "equipment_applied",    "at": self.equipment_applied_at,    "request_id": self.equipment_request_id},
                {"step": "checkin",             "at": self.checkin_at,             "request_id": None},
                {"step": "training_enrolled",   "at": self.training_enrolled_at,   "request_id": self.training_request_id},
                {"step": "completed",           "at": self.completed_at,            "request_id": None},
            ],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class OnboardingEvent(Base):
    """
    入职工作流事件记录。

    每次状态转换都会写入一条事件记录，用于审计和问题追踪。
    """
    __tablename__ = "onboarding_events"

    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    instance_id = Column(String, ForeignKey("onboarding_instances.id"), nullable=False, index=True)
    event_type  = Column(String, nullable=False, index=True)  # offer_confirmed / first_checkin
    payload     = Column(JSON, default=dict)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    instance = relationship("OnboardingInstance", back_populates="events")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "instance_id": self.instance_id,
            "event_type": self.event_type,
            "payload": self.payload or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
