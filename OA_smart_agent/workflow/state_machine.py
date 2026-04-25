"""
入职工作流状态机引擎

定义状态转换规则，响应外部事件驱动状态流转，
并触发后续异步动作（Celery 任务）。
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from .models import OnboardingInstance, OnboardingEvent, OnboardingState
from .database import get_session

logger = logging.getLogger("oa_agent.workflow")


# 状态转换表：当前状态 → {事件类型: {下一状态, 触发的MCP工具, 步骤描述}}
_TRANSITIONS = {
    OnboardingState.OFFER_PENDING: {
        "offer_confirmed": {
            "next_state": OnboardingState.OFFER_CONFIRMED,
            "actions": ["create_oa_account", "apply_it_equipment"],
            "detail": "候选人已确认 offer，正在开通 OA 账号和申请 IT 设备",
        },
    },
    OnboardingState.OFFER_CONFIRMED: {
        "first_checkin": {
            "next_state": OnboardingState.FIRST_CHECKIN,
            "actions": ["enroll_training_courses"],
            "detail": "新员工已入职打卡，正在分配培训课程",
        },
    },
}


def _get_action_params(instance: OnboardingInstance, tool_name: str) -> dict:
    """根据工具名从实例上下文构建调用参数"""
    base = {
        "employee_id": instance.employee_id,
        "employee_name": instance.candidate_name,
        "department": instance.department,
    }
    if tool_name == "create_oa_account":
        return {
            **base,
            "position": instance.position,
            "email": instance.email,
            "mobile": "",
            "manager_id": "",
        }
    if tool_name == "apply_it_equipment":
        return {
            **base,
            "device_type": instance.device_type or "笔记本电脑",
            "device_spec": "",
            "quantity": 1,
            "reason": f"新员工入职: {instance.position}",
        }
    if tool_name == "enroll_training_courses":
        return {
            **base,
            "course_list": instance.course_list or [],
            "enrollment_type": "新员工入职培训",
            "deadline_days": 30,
        }
    return {}


class OnboardingStateMachine:
    """
    入职工作流状态机。

    接收外部事件，执行状态转换，更新数据库，并触发后续异步动作。
    """

    def __init__(self, instance: OnboardingInstance):
        self.instance = instance

    def can_handle(self, event_type: str) -> bool:
        """检查当前状态是否支持处理该事件类型"""
        transitions = _TRANSITIONS.get(self.instance.state, {})
        return event_type in transitions

    def get_next_state(self, event_type: str) -> Optional[OnboardingState]:
        """获取指定事件对应的下一状态"""
        transitions = _TRANSITIONS.get(self.instance.state, {})
        transition = transitions.get(event_type)
        return transition["next_state"] if transition else None

    def process(
        self,
        event_type: str,
        payload: Optional[dict] = None,
    ) -> OnboardingInstance:
        """
        处理事件：
        1. 写入事件记录（OnboardingEvent）
        2. 更新实例状态和时间戳
        3. 触发后续 Celery 任务

        如果事件无法处理（状态不支持），抛出 ValueError。
        """
        if not self.can_handle(event_type):
            raise ValueError(
                f"事件 '{event_type}' 无法在状态 '{self.instance.state.value}' 下处理"
            )

        transition = _TRANSITIONS[self.instance.state][event_type]
        next_state = transition["next_state"]
        actions = transition["actions"]
        detail = transition["detail"]

        # 1. 写入事件记录
        with get_session() as db:
            instance = db.merge(self.instance)
            event = OnboardingEvent(
                id=str(uuid.uuid4()),
                instance_id=instance.id,
                event_type=event_type,
                payload=payload or {},
            )
            db.add(event)

            # 2. 更新实例状态
            now = datetime.now(timezone.utc)
            instance.state = next_state
            instance.current_step_detail = detail
            instance.updated_at = now

            # 记录时间戳
            if event_type == "offer_confirmed":
                instance.offer_confirmed_at = now
            elif event_type == "first_checkin":
                instance.checkin_at = now

            db.commit()
            db.refresh(instance)

        logger.info(
            "workflow_state_transition",
            extra={
                "instance_id": instance.id,
                "from_state": self.instance.state.value,
                "to_state": next_state.value,
                "event_type": event_type,
                "actions": actions,
            },
        )

        # 3. 触发后续异步动作
        self._dispatch_actions(instance, actions)

        return instance

    def _dispatch_actions(self, instance: OnboardingInstance, actions: list):
        """触发 Celery 异步任务"""
        try:
            from .tasks import trigger_workflow_action
            for action in actions:
                params = _get_action_params(instance, action)
                trigger_workflow_action.delay(instance.id, action, params)
                logger.info(
                    "workflow_action_dispatched",
                    extra={
                        "instance_id": instance.id,
                        "action": action,
                        "params": params,
                    },
                )
        except Exception as e:
            logger.error(
                "workflow_action_dispatch_failed",
                extra={"instance_id": instance.id, "actions": actions, "error": str(e)},
            )

    def mark_completed(self):
        """标记工作流已完成（内部调用，无需事件驱动）"""
        with get_session() as db:
            instance = db.merge(self.instance)
            now = datetime.now(timezone.utc)
            instance.state = OnboardingState.COMPLETED
            instance.completed_at = now
            instance.current_step_detail = "入职流程全部完成"
            instance.updated_at = now
            db.commit()
            db.refresh(instance)
        logger.info("workflow_completed", extra={"instance_id": self.instance.id})

    def mark_failed(self, reason: str):
        """标记工作流失败"""
        with get_session() as db:
            instance = db.merge(self.instance)
            now = datetime.now(timezone.utc)
            instance.state = OnboardingState.FAILED
            instance.failed_at = now
            instance.current_step_detail = f"流程失败: {reason}"
            instance.updated_at = now
            db.commit()
            db.refresh(instance)
        logger.warning("workflow_failed", extra={"instance_id": self.instance.id, "reason": reason})
