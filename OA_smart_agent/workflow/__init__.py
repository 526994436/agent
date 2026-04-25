"""
Workflow 模块
入职流程事件驱动状态机
"""

from .models import OnboardingInstance, OnboardingState, OnboardingEvent
from .state_machine import OnboardingStateMachine
from .database import get_session, get_engine, init_workflow_db

__all__ = [
    "OnboardingInstance",
    "OnboardingState",
    "OnboardingEvent",
    "OnboardingStateMachine",
    "get_session",
    "get_engine",
    "init_workflow_db",
]
