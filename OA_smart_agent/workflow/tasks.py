"""
工作流异步任务（Celery）

使用 Celery 在后台异步执行 Java 后端 API 调用，
避免阻塞 API 请求。
"""

import time
import uuid
import logging

from celery import shared_task

from ..tasks import app as celery_app  # 复用主 Celery app
from ..config import settings
from .database import get_session
from .models import OnboardingInstance, OnboardingState
from .state_machine import OnboardingStateMachine

logger = logging.getLogger("oa_agent.workflow.tasks")


def _build_headers(request_id: str) -> dict:
    return {
        "Content-Type": "application/json",
        "X-Request-ID": request_id,
        "X-Source": "oa-smart-agent-workflow",
    }


# tool_name → Java 后端 URL 路径映射
_TOOL_URL_MAP = {
    "generate_offer_letter": "/api/onboarding/offer-letter",
    "create_oa_account":    "/api/onboarding/oa-account",
    "apply_it_equipment":   "/api/onboarding/it-equipment",
    "enroll_training_courses": "/api/onboarding/training",
}


# tool_name → instance 时间戳字段名
_TOOL_TIMESTAMP_FIELD = {
    "generate_offer_letter":  "offer_sent_at",
    "create_oa_account":     "account_created_at",
    "apply_it_equipment":    "equipment_applied_at",
    "enroll_training_courses": "training_enrolled_at",
}


def _sync_http_call(url: str, payload: dict, headers: dict, timeout: int = 30) -> dict:
    """在 Celery worker 线程池中同步执行 HTTP 调用"""
    import httpx
    with httpx.Client(timeout=httpx.Timeout(connect=5.0, read=float(timeout), write=5.0, pool=10.0)) as client:
        response = client.post(url, json=payload, headers=headers)
        if response.status_code >= 500:
            response.raise_for_status()
        return response.json()


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(Exception,),
)
def trigger_workflow_action(
    self,
    instance_id: str,
    tool_name: str,
    params: dict,
) -> dict:
    """
    异步调用 Java 后端 MCP 工具。

    步骤：
    1. 记录 Java 后端 request_id 到 instance
    2. 发起 HTTP 调用
    3. 更新 instance 时间戳字段
    4. 检查是否可以自动完成工作流（FIRST_CHECKIN 之后无更多动作）
    """
    logger.info(
        "workflow_action_start",
        extra={"instance_id": instance_id, "tool_name": tool_name, "params": params},
    )

    request_id = str(uuid.uuid4())
    url = f"{settings.java_backend_base_url}{_TOOL_URL_MAP.get(tool_name, '')}"

    payload = {
        "action_type": tool_name,
        "params": params,
        "request_id": request_id,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "workflow_celery",
        },
    }
    headers = _build_headers(request_id)

    try:
        result = _sync_http_call(url, payload, headers)
        logger.info(
            "workflow_action_success",
            extra={
                "instance_id": instance_id,
                "tool_name": tool_name,
                "request_id": request_id,
                "result": result,
            },
        )

        # 更新 instance：写入 request_id 和时间戳
        with get_session() as db:
            instance = db.query(OnboardingInstance).get(instance_id)
            if instance:
                # 写入 request_id
                request_id_field = f"{tool_name}_request_id"
                if hasattr(instance, request_id_field):
                    setattr(instance, request_id_field, request_id)

                # 写入时间戳
                ts_field = _TOOL_TIMESTAMP_FIELD.get(tool_name)
                if ts_field and hasattr(instance, ts_field):
                    from datetime import datetime, timezone
                    setattr(instance, ts_field, datetime.now(timezone.utc))

                db.commit()

                # 如果是 enroll_training_courses 完成后，自动标记 COMPLETED
                if tool_name == "enroll_training_courses":
                    sm = OnboardingStateMachine(instance)
                    sm.mark_completed()

        return {"success": True, "request_id": request_id, "data": result}

    except Exception as exc:
        logger.error(
            "workflow_action_failed",
            extra={
                "instance_id": instance_id,
                "tool_name": tool_name,
                "request_id": request_id,
                "error": str(exc),
            },
        )
        # 标记工作流失败
        with get_session() as db:
            instance = db.query(OnboardingInstance).get(instance_id)
            if instance:
                sm = OnboardingStateMachine(instance)
                sm.mark_failed(f"{tool_name} 执行失败: {exc}")

        # 让 Celery 重试
        raise self.retry(exc=exc)
