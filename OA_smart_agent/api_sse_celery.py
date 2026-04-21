"""
SSE-Celery 流式集成模块 (api_sse_celery.py)

提供异步任务与 SSE 流式推送的完整集成：

架构流程：
┌─────────┐     触发任务      ┌──────────────┐     提交任务     ┌──────────────┐
│  前端   │ ────────────────▶ │  Web API     │ ──────────────▶ │   Celery     │
│         │ ◀─────────────── │  (立即返回   │                 │   Worker     │
└─────────┘   task_id        │   task_id)   │                 └──────┬───────┘
                              └──────────────┘                        │
                                       │                              │
                              ┌─────────┴─────────┐                    │
                              │   SSE 长连接      │                    │
                              │  /chat/stream     │◀───────────────────┤
                              │  /chat/async/stream│                   │
                              └─────────┬─────────┘                    │
                                        │                              │
                              ┌─────────┴─────────┐                    │
                              │  Redis Pub/Sub    │◀───────────────────┘
                              │  task:{task_id}   │    推送进度事件
                              └───────────────────┘

SSE 事件类型（新增）：
- task_accepted: 任务已接受，等待执行
- task_progress: 任务执行进度更新
- task_chunk: 文本片段（用于打字机效果）
- task_action_pending: 动作待确认
- task_completed: 任务完成
- task_failed: 任务失败

使用方法：
1. POST /api/v1/chat/async/stream → 触发异步任务，返回 task_id
2. GET /api/v1/chat/async/stream/{task_id} → SSE 长连接，实时推送任务进度
"""

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import redis.asyncio as aioredis
from redis.asyncio.client import PubSub

from config import settings

logger = logging.getLogger("oa_agent.api_sse_celery")


# =============================================================================
# 事件类型枚举
# =============================================================================

class AsyncTaskEventType(str, Enum):
    """异步任务 SSE 事件类型"""
    TASK_ACCEPTED = "task_accepted"       # 任务已接受
    TASK_PROGRESS = "task_progress"        # 进度更新
    TASK_CHUNK = "task_chunk"              # 文本片段
    TASK_ACTION_PENDING = "task_action_pending"  # 动作待确认
    TASK_COMPLETED = "task_completed"      # 任务完成
    TASK_FAILED = "task_failed"            # 任务失败
    TASK_CANCELLED = "task_cancelled"      # 任务取消
    HEARTBEAT = "heartbeat"                # 心跳


# =============================================================================
# 任务进度事件
# =============================================================================

@dataclass
class TaskProgressEvent:
    """任务进度事件"""
    task_id: str
    event_type: str
    step: str = ""
    step_description: str = ""
    progress_percent: int = 0
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_sse_data(self) -> Dict[str, Any]:
        """转换为 SSE 数据格式"""
        result = {
            "task_id": self.task_id,
            "event_type": self.event_type,
            "step": self.step,
            "progress_percent": self.progress_percent,
            "message": self.message,
            "timestamp": self.timestamp,
        }
        if self.step:
            result["step_description"] = self.step_description
        if self.data:
            result.update(self.data)
        return result


# =============================================================================
# Redis Pub/Sub 事件通道
# =============================================================================

class TaskEventChannel:
    """
    任务事件通道 - 基于 Redis Pub/Sub 的任务进度推送

    设计说明：
    - 每个任务有独立的 Redis Channel：task:{task_id}
    - Celery Worker 执行过程中向 Channel 推送进度事件
    - SSE 接口订阅 Channel，实时接收并推送事件到前端
    - 支持多个 SSE 连接订阅同一任务（广播）
    """

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.redis_url
        self._redis: Optional[aioredis.Redis] = None
        self._pubsub_connections: Dict[str, PubSub] = {}

    async def _get_redis(self) -> aioredis.Redis:
        """获取或创建 Redis 连接"""
        if self._redis is None:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    def _get_channel_name(self, task_id: str) -> str:
        """获取任务对应的 Channel 名称"""
        return f"task:{task_id}"

    async def publish_progress(self, event: TaskProgressEvent):
        """
        Celery Worker 调用：发布任务进度事件

        Args:
            event: 任务进度事件
        """
        try:
            redis = await self._get_redis()
            channel = self._get_channel_name(event.task_id)
            message = json.dumps(event.to_sse_data(), ensure_ascii=False)
            await redis.publish(channel, message)

            logger.debug(
                "task_event_published",
                extra={
                    "task_id": event.task_id,
                    "event_type": event.event_type,
                    "step": event.step,
                    "progress_percent": event.progress_percent,
                }
            )
        except Exception as e:
            logger.error(
                "task_event_publish_failed",
                extra={"task_id": event.task_id, "error": str(e)}
            )

    async def subscribe(self, task_id: str) -> PubSub:
        """
        SSE 接口调用：订阅任务进度事件

        Args:
            task_id: 任务 ID

        Returns:
            PubSub 对象，需调用 listen() 获取事件
        """
        redis = await self._get_redis()
        pubsub = redis.pubsub()
        channel = self._get_channel_name(task_id)
        await pubsub.subscribe(channel)
        self._pubsub_connections[task_id] = pubsub
        return pubsub

    async def unsubscribe(self, task_id: str):
        """取消订阅"""
        if task_id in self._pubsub_connections:
            pubsub = self._pubsub_connections[task_id]
            await pubsub.unsubscribe()
            await pubsub.close()
            del self._pubsub_connections[task_id]

    async def close(self):
        """关闭所有连接"""
        for task_id in list(self._pubsub_connections.keys()):
            await self.unsubscribe(task_id)
        if self._redis:
            await self._redis.close()
            self._redis = None


# 全局事件通道实例
_event_channel: Optional[TaskEventChannel] = None


def get_event_channel() -> TaskEventChannel:
    """获取全局事件通道单例"""
    global _event_channel
    if _event_channel is None:
        _event_channel = TaskEventChannel()
    return _event_channel


# =============================================================================
# 任务状态存储（Redis）
# =============================================================================

class AsyncTaskStore:
    """
    异步任务状态存储 - 基于 Redis

    用于存储任务元数据和最终结果，SSE 接口可快速查询任务状态
    """

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.redis_url
        self._redis: Optional[aioredis.Redis] = None
        self._key_prefix = "async_task:"
        self._ttl = 86400  # 24小时过期

    async def _get_redis(self) -> aioredis.Redis:
        """获取或创建 Redis 连接"""
        if self._redis is None:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    def _get_key(self, task_id: str) -> str:
        """获取任务对应的 Redis Key"""
        return f"{self._key_prefix}{task_id}"

    async def create_task(
        self,
        task_id: str,
        session_id: str,
        user_id: str,
        status: str = "pending",
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        创建新任务记录

        Args:
            task_id: 任务 ID
            session_id: 会话 ID
            user_id: 用户 ID
            status: 初始状态
            metadata: 额外元数据

        Returns:
            是否创建成功
        """
        try:
            redis = await self._get_redis()
            key = self._get_key(task_id)

            data = {
                "task_id": task_id,
                "session_id": session_id,
                "user_id": user_id,
                "status": status,
                "created_at": time.time(),
                "updated_at": time.time(),
                "progress_percent": 0,
                "current_step": "",
                "current_message": "",
                "final_response": "",
                "requires_approval": False,
                "draft_action": None,
                "error": None,
                "metadata": json.dumps(metadata or {}, ensure_ascii=False),
            }

            # 使用 HSET 存储
            await redis.hset(key, mapping=data)
            await redis.expire(key, self._ttl)

            logger.info(
                "async_task_created",
                extra={"task_id": task_id, "session_id": session_id, "user_id": user_id}
            )
            return True

        except Exception as e:
            logger.error(
                "async_task_create_failed",
                extra={"task_id": task_id, "error": str(e)}
            )
            return False

    async def update_progress(
        self,
        task_id: str,
        status: str = None,
        progress_percent: int = None,
        current_step: str = None,
        current_message: str = None,
        event_type: str = None,
    ):
        """
        更新任务进度

        Args:
            task_id: 任务 ID
            status: 状态
            progress_percent: 进度百分比
            current_step: 当前步骤
            current_message: 当前消息
            event_type: 事件类型（用于触发 SSE 推送）
        """
        try:
            redis = await self._get_redis()
            key = self._get_key(task_id)

            updates = {"updated_at": time.time()}

            if status is not None:
                updates["status"] = status
            if progress_percent is not None:
                updates["progress_percent"] = progress_percent
            if current_step is not None:
                updates["current_step"] = current_step
            if current_message is not None:
                updates["current_message"] = current_message

            await redis.hset(key, mapping=updates)

            # 发布事件到 Pub/Sub
            if event_type:
                event = TaskProgressEvent(
                    task_id=task_id,
                    event_type=event_type,
                    step=current_step or "",
                    step_description=current_message or "",
                    progress_percent=progress_percent or 0,
                    message=current_message or "",
                )
                await get_event_channel().publish_progress(event)

        except Exception as e:
            logger.error(
                "async_task_update_failed",
                extra={"task_id": task_id, "error": str(e)}
            )

    async def complete_task(
        self,
        task_id: str,
        final_response: str = None,
        requires_approval: bool = False,
        draft_action: Dict[str, Any] = None,
    ):
        """
        完成任务

        Args:
            task_id: 任务 ID
            final_response: 最终回复
            requires_approval: 是否需要审批
            draft_action: 审批草稿
        """
        try:
            redis = await self._get_redis()
            key = self._get_key(task_id)

            updates = {
                "status": "completed",
                "progress_percent": 100,
                "current_step": "完成",
                "current_message": "任务执行完成",
                "updated_at": time.time(),
                "final_response": final_response or "",
                "requires_approval": requires_approval,
                "draft_action": json.dumps(draft_action, ensure_ascii=False) if draft_action else "",
                "completed_at": time.time(),
            }

            await redis.hset(key, mapping=updates)

            # 发布完成事件
            event = TaskProgressEvent(
                task_id=task_id,
                event_type=AsyncTaskEventType.TASK_COMPLETED.value,
                step="完成",
                step_description="任务执行完成",
                progress_percent=100,
                message=final_response or "任务执行完成",
                data={
                    "requires_approval": requires_approval,
                    "draft_action": draft_action,
                    "final_response": final_response,
                }
            )
            await get_event_channel().publish_progress(event)

        except Exception as e:
            logger.error(
                "async_task_complete_failed",
                extra={"task_id": task_id, "error": str(e)}
            )

    async def fail_task(self, task_id: str, error: str):
        """
        标记任务失败

        Args:
            task_id: 任务 ID
            error: 错误信息
        """
        try:
            redis = await self._get_redis()
            key = self._get_key(task_id)

            updates = {
                "status": "failed",
                "updated_at": time.time(),
                "error": error,
            }

            await redis.hset(key, mapping=updates)

            # 发布失败事件
            event = TaskProgressEvent(
                task_id=task_id,
                event_type=AsyncTaskEventType.TASK_FAILED.value,
                step="失败",
                step_description="任务执行失败",
                progress_percent=0,
                message=error,
                data={"error": error}
            )
            await get_event_channel().publish_progress(event)

        except Exception as e:
            logger.error(
                "async_task_fail_failed",
                extra={"task_id": task_id, "error": str(e)}
            )

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态

        Args:
            task_id: 任务 ID

        Returns:
            任务状态字典，不存在返回 None
        """
        try:
            redis = await self._get_redis()
            key = self._get_key(task_id)

            data = await redis.hgetall(key)

            if not data:
                return None

            # 解析 JSON 字段
            if data.get("draft_action"):
                try:
                    data["draft_action"] = json.loads(data["draft_action"])
                except:
                    data["draft_action"] = None

            if data.get("metadata"):
                try:
                    data["metadata"] = json.loads(data["metadata"])
                except:
                    data["metadata"] = {}

            return data

        except Exception as e:
            logger.error(
                "async_task_get_failed",
                extra={"task_id": task_id, "error": str(e)}
            )
            return None

    async def close(self):
        """关闭连接"""
        if self._redis:
            await self._redis.close()
            self._redis = None


# 全局任务状态存储实例
_task_store: Optional[AsyncTaskStore] = None


def get_task_store() -> AsyncTaskStore:
    """获取全局任务状态存储单例"""
    global _task_store
    if _task_store is None:
        _task_store = AsyncTaskStore()
    return _task_store


# =============================================================================
# SSE 事件格式化
# =============================================================================

def format_task_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """格式化 SSE 事件数据"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def format_task_sse_comment(message: str) -> str:
    """格式化 SSE 注释"""
    return f": {message}\n\n"


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "AsyncTaskEventType",      # 任务事件类型枚举
    "TaskProgressEvent",       # 任务进度事件
    "TaskEventChannel",        # Redis 事件通道
    "AsyncTaskStore",          # 异步任务存储
    "get_event_channel",       # 获取事件通道
    "get_task_store",          # 获取任务存储
    "format_task_sse_event",   # SSE 事件格式化
    "format_task_sse_comment", # SSE 注释格式化
]
