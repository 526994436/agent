"""
Celery 异步任务队列 (tasks.py)

提供异步执行 LangGraph 工作流的能力，支持：
1. 异步聊天任务（带 SSE 流式推送）
2. 异步多模态分析任务
3. 任务状态查询
4. 任务取消
5. Schema 强约束集成
6. 草稿箱功能集成
7. 任务挂起和恢复

SSE 流式推送：
- Celery Worker 执行过程中，通过 Redis Pub/Sub 推送进度事件
- SSE 接口订阅事件，实时推送给前端
- 支持打字机效果的文本片段推送

Schema 强约束：
- 每个业务操作都有固定的字段定义
- 代码计算缺失参数，大模型只负责自然语言提问
- 杜绝 LLM 幻觉乱问、漏传参数

草稿箱集成：
- 未提交草稿全程隔离在 Python 层
- 用户中途插问、切换业务，自动触发任务挂起
- 切换回来无缝续填

使用方法：
1. 启动 Worker：celery -A tasks worker --loglevel=info --concurrency=4
2. 启动 API：uvicorn main:app --workers 1
3. 触发任务：POST /api/v1/chat/async/stream
4. 订阅进度：GET /api/v1/chat/async/stream/{task_id}

依赖：
- Redis 作为消息代理（Broker）和 Pub/Sub 通道
- 配置项：settings.celery_broker_url, settings.redis_url
"""

from __future__ import annotations

import asyncio
import time
import logging
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from celery import Celery
from celery.exceptions import IgnoreError

logger = logging.getLogger("oa_agent.tasks")


# 尝试导入配置，如果失败则使用默认值
try:
    from config import settings
except ImportError:
    settings = type('Settings', (), {
        'celery_broker_url': 'redis://localhost:6379/0',
        'celery_result_backend': 'redis://localhost:6379/0',
        'celery_result_expires': 86400,
        'celery_task_acks_late': True,
        'celery_worker_prefetch_multiplier': 1,
        'redis_url': 'redis://localhost:6379',
    })()


# =============================================================================
# Celery 应用初始化
# =============================================================================

app = Celery(
    "oa_smart_agent",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Celery 配置
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    result_expires=settings.celery_result_expires,
    task_acks_late=settings.celery_task_acks_late,
    worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,
    task_routes={
        "tasks.submit_chat_task": {"queue": "chat"},
        "tasks.process_multimodal_async": {"queue": "multimodal"},
    },
)


# =============================================================================
# 任务状态类型定义
# =============================================================================

class TaskStatusEnum(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    STARTED = "started"
    PROGRESS = "progress"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass
class TaskProgress:
    """任务进度信息"""
    progress_percent: int = 0
    current_step: str = ""
    message: str = ""


@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    status: TaskStatusEnum
    progress: Optional[TaskProgress] = None
    session_id: Optional[str] = None
    requires_approval: bool = False
    draft_action: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    # 新增：草稿相关字段
    has_active_draft: bool = False
    suspended_task_count: int = 0
    missing_fields: List[str] = field(default_factory=list)


# =============================================================================
# Draft 管理集成
# =============================================================================

def _publish_draft_event(task_id: str, event_type: str, draft_data: dict,
                        message: str = "", progress_percent: int = 0):
    """
    发布草稿相关事件

    Args:
        task_id: 任务 ID
        event_type: 事件类型
        draft_data: 草稿数据
        message: 消息
        progress_percent: 进度百分比
    """
    event = {
        "task_id": task_id,
        "event_type": event_type,
        "progress_percent": progress_percent,
        "message": message,
        "timestamp": time.time(),
        "data": draft_data,
    }

    import redis
    from urllib.parse import urlparse

    try:
        parsed = urlparse(settings.redis_url)
        r = redis.Redis(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            password=parsed.password,
            decode_responses=True,
        )

        channel = f"task:{task_id}"
        r.publish(channel, json.dumps(event, ensure_ascii=False))
        r.close()

    except Exception as e:
        logger.warning(f"_publish_draft_event failed: {e}")


# =============================================================================
# Celery Worker 进度推送辅助
# =============================================================================

def _publish_progress(
    task_id: str,
    event_type: str,
    step: str = "",
    step_description: str = "",
    progress_percent: int = 0,
    message: str = "",
    data: Dict[str, Any] = None,
):
    """
    Celery Worker 调用：发布任务进度到 Redis Pub/Sub

    Args:
        task_id: 任务 ID
        event_type: 事件类型
        step: 当前步骤
        step_description: 步骤描述
        progress_percent: 进度百分比
        message: 消息
        data: 额外数据
    """
    import json
    import redis
    from urllib.parse import urlparse

    parsed = urlparse(settings.redis_url)
    r = redis.Redis(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password,
        decode_responses=True,
    )

    try:
        event = {
            "task_id": task_id,
            "event_type": event_type,
            "step": step,
            "step_description": step_description,
            "progress_percent": progress_percent,
            "message": message,
            "timestamp": time.time(),
            "data": data or {},
        }

        channel = f"task:{task_id}"
        r.publish(channel, json.dumps(event, ensure_ascii=False))

        logger.debug(
            f"task_progress_published: {event_type}",
            extra={
                "task_id": task_id,
                "step": step,
                "progress_percent": progress_percent,
            }
        )
    finally:
        r.close()


def _update_task_status(
    task_id: str,
    status: str = None,
    progress_percent: int = None,
    current_step: str = None,
    current_message: str = None,
    final_response: str = None,
    requires_approval: bool = None,
    draft_action: Dict[str, Any] = None,
    error: str = None,
):
    """
    Celery Worker 调用：更新任务状态到 Redis Hash

    Args:
        task_id: 任务 ID
        status: 状态
        progress_percent: 进度百分比
        current_step: 当前步骤
        current_message: 当前消息
        final_response: 最终回复
        requires_approval: 是否需要审批
        draft_action: 审批草稿
        error: 错误信息
    """
    import json
    import redis
    from urllib.parse import urlparse

    parsed = urlparse(settings.redis_url)
    r = redis.Redis(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password,
        decode_responses=True,
    )

    try:
        key = f"async_task:{task_id}"
        updates = {"updated_at": time.time()}

        if status is not None:
            updates["status"] = status
        if progress_percent is not None:
            updates["progress_percent"] = progress_percent
        if current_step is not None:
            updates["current_step"] = current_step
        if current_message is not None:
            updates["current_message"] = current_message
        if final_response is not None:
            updates["final_response"] = final_response
        if requires_approval is not None:
            updates["requires_approval"] = requires_approval
        if draft_action is not None:
            updates["draft_action"] = json.dumps(draft_action, ensure_ascii=False)
        if error is not None:
            updates["error"] = error

        r.hset(key, mapping=updates)
        r.expire(key, 86400)  # 24小时过期

    finally:
        r.close()


def _split_text_for_streaming(text: str, chunk_size: int = 20) -> list:
    """
    将文本分割成适合流式输出的块

    Args:
        text: 原始文本
        chunk_size: 每块大小

    Returns:
        文本块列表
    """
    if not text:
        return []

    chunks = []
    sentence_delimiters = ['。', '！', '？', '；', '\n']

    parts = [text]
    for delimiter in sentence_delimiters:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(delimiter))
        parts = [p + delimiter for p in new_parts[:-1]] + [new_parts[-1]] if new_parts else []

    parts = [p for p in parts if p.strip()]

    for part in parts:
        if len(part) <= chunk_size:
            chunks.append(part)
        else:
            sub_delimiters = ['，', '、', ',', ' ']
            sub_parts = [part]
            for delimiter in sub_delimiters:
                new_sub_parts = []
                for sp in sub_parts:
                    new_sub_parts.extend(sp.split(delimiter))
                sub_parts = [s + delimiter for s in new_sub_parts[:-1]] + [new_sub_parts[-1]] if new_sub_parts else []

            for sp in sub_parts:
                sp = sp.strip()
                if sp:
                    if len(sp) <= chunk_size:
                        chunks.append(sp)
                    else:
                        for i in range(0, len(sp), chunk_size):
                            chunk = sp[i:i+chunk_size]
                            if chunk:
                                chunks.append(chunk)

    return chunks if chunks else [text]


# =============================================================================
# Celery 任务定义
# =============================================================================

@app.task(bind=True, name="tasks.submit_chat_task", max_retries=3, default_retry_delay=60)
def submit_chat_task(
    self,
    query: str,
    session_id: str,
    user_id: str,
    user_token: str,
    trace_id: str,
    enable_streaming: bool = True,
    image_data: Optional[str] = None,
    file_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    异步聊天任务（支持 SSE 流式推送 + 多模态 + 文件解析）

    Args:
        query: 用户问题
        session_id: 会话ID
        user_id: 用户ID
        user_token: 用户Token
        trace_id: 追踪ID
        enable_streaming: 是否启用流式推送
        image_data: Base64 编码的图片数据（可选，用于多模态处理）
        file_data: 上传的文件数据（可选，用于文档解析）

    Returns:
        任务结果字典
    """
    task_id = self.request.id
    start_time = time.time()

    logger.info(
        "async_chat_task_started",
        extra={
            "task_id": task_id,
            "session_id": session_id,
            "user_id": user_id,
            "trace_id": trace_id,
            "has_file": file_data is not None,
            "file_name": file_data.get("file_name") if file_data else None,
        }
    )

    try:
        # 更新状态：开始执行
        _update_task_status(
            task_id,
            status="started",
            progress_percent=5,
            current_step="初始化",
            current_message="正在准备执行环境...",
        )
        _publish_progress(
            task_id,
            event_type="task_progress",
            step="初始化",
            step_description="正在准备执行环境...",
            progress_percent=5,
            message="任务已接受，正在准备执行...",
        )

        # 导入核心模块
        try:
            from graph import get_agent_graph
            from langchain_core.messages import HumanMessage
            from schemas import AgentState
            from history import save_conversation_history, update_session_status, SessionStatus
        except ImportError as e:
            raise RuntimeError(f"无法导入核心模块: {e}")

        # 更新状态：加载配置
        _update_task_status(
            task_id,
            status="progress",
            progress_percent=10,
            current_step="加载配置",
            current_message="正在加载系统配置...",
        )
        _publish_progress(
            task_id,
            event_type="task_progress",
            step="加载配置",
            step_description="正在加载系统配置...",
            progress_percent=10,
            message="正在加载系统配置...",
        )

        # 获取 Agent Graph
        graph = get_agent_graph()
        config = {
            "configurable": {
                "thread_id": session_id,
            }
        }

        # 更新状态：意图识别
        _update_task_status(
            task_id,
            status="progress",
            progress_percent=20,
            current_step="意图识别",
            current_message="正在进行意图识别...",
        )
        _publish_progress(
            task_id,
            event_type="task_progress",
            step="意图识别",
            step_description="正在进行意图识别...",
            progress_percent=20,
            message="正在分析您的意图...",
        )

        # ─────────────────────────────────────────
        # 会话恢复：检查是否是继续之前的对话
        # ─────────────────────────────────────────
        is_resuming_session = False
        restored_state = {}
        
        try:
            saved_state = graph.get_state(config)
            if saved_state and saved_state.values:
                # 发现历史状态，说明是继续对话
                is_resuming_session = True
                restored_state = saved_state.values
                
                logger.info(
                    "session_resumed_in_task",
                    extra={
                        "task_id": task_id,
                        "session_id": session_id,
                        "has_pending_slots": bool(restored_state.get("pending_slots")),
                        "has_confirmed_slots": bool(restored_state.get("confirmed_slots")),
                        "messages_count": len(restored_state.get("messages", [])),
                        "component": "tasks",
                    }
                )
                
                # 推送恢复提示
                _publish_progress(
                    task_id,
                    event_type="task_progress",
                    step="会话恢复",
                    step_description="正在恢复历史会话...",
                    progress_percent=8,
                    message="正在恢复您的会话上下文...",
                )
        except Exception as e:
            logger.warning(
                "session_restore_warning_in_task",
                extra={
                    "task_id": task_id,
                    "session_id": session_id,
                    "error": str(e),
                    "component": "tasks",
                }
            )

        # ─────────────────────────────────────────
        # 构建输入状态
        # ─────────────────────────────────────────
        
        # 如果是继续对话，恢复历史消息
        if is_resuming_session and restored_state.get("messages"):
            # 追加新消息而不是替换
            existing_messages = list(restored_state.get("messages", []))
            existing_messages.append(HumanMessage(content=query))
            messages_input = existing_messages
        else:
            # 新对话
            messages_input = [HumanMessage(content=query)]

        # 构建初始状态
        input_state: AgentState = {
            "messages": messages_input,
            "user_token": user_token,
            "intent": None,
            "extracted_params": None,
            "requires_approval": False,
            "action_payload": None,
            "interrupted": False,
            "retrieved_docs": None,
            "final_response": None,
            "session_id": session_id,
            "reranked_docs": None,
            "reflection_results": None,
            "needs_escalation": False,
            "kg_context": None,
            "user_id": user_id,
            "compressed_context": None,
            "pending_slots": restored_state.get("pending_slots", []),
            "confirmed_slots": restored_state.get("confirmed_slots", {}),
            "multimodal_context": None,
        }

        # ─────────────────────────────────────────
        # 多模态图片处理（如果提供了图片）
        # ─────────────────────────────────────────
        multimodal_context = None
        if image_data:
            try:
                # 导入配置检查多模态是否启用
                from config import settings as config_settings
                if config_settings.multimodal_enabled:
                    _publish_progress(
                        task_id,
                        event_type="task_progress",
                        step="多模态处理",
                        step_description="正在分析图片内容...",
                        progress_percent=15,
                        message="正在分析图片内容...",
                    )

                    from multimodal import get_multimodal_processor
                    processor = get_multimodal_processor()
                    analysis_result = processor.analyze_image(image_data, query)

                    multimodal_context = {
                        "raw_text": analysis_result.get("raw_text", ""),
                        "description": analysis_result.get("description", ""),
                        "extracted_data": analysis_result.get("extracted_data", {}),
                        "confidence": analysis_result.get("confidence", 0.0),
                        "scene_type": analysis_result.get("scene_type", "general"),
                    }

                    # 将图片描述融入 query，增强上下文
                    if analysis_result.get("description"):
                        description = analysis_result.get("description")
                        input_state["messages"] = [
                            HumanMessage(content=f"用户上传了一张图片，描述：{description}\n\n用户原始问题：{query}")
                        ]

                    # 存储多模态上下文到状态
                    input_state["multimodal_context"] = multimodal_context

                    logger.info(
                        "async_multimodal_processed",
                        extra={
                            "task_id": task_id,
                            "scene_type": multimodal_context.get("scene_type"),
                            "confidence": multimodal_context.get("confidence"),
                            "component": "tasks",
                        }
                    )
            except Exception as e:
                logger.warning(
                    "async_multimodal_failed",
                    extra={"task_id": task_id, "error": str(e), "component": "tasks"}
                )
                # 多模态处理失败不影响主流程，继续执行

        # ─────────────────────────────────────────
        # 文件解析处理（如果提供了文件）
        # ─────────────────────────────────────────
        parsed_file_content = None
        if file_data:
            try:
                _publish_progress(
                    task_id,
                    event_type="task_progress",
                    step="文件解析",
                    step_description="正在解析文件内容...",
                    progress_percent=20,
                    message="正在解析上传的文件...",
                )

                file_name = file_data.get("file_name", "")
                file_content_b64 = file_data.get("file_content", "")

                if file_content_b64:
                    import base64
                    import io
                    import os

                    # 解码 Base64 文件内容
                    file_bytes = base64.b64decode(file_content_b64)

                    # 获取文件扩展名
                    file_ext = os.path.splitext(file_name)[1].lower().lstrip(".")

                    # 创建临时文件
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                        tmp_file.write(file_bytes)
                        tmp_path = tmp_file.name

                    try:
                        # 根据文件类型选择解析器
                        if file_ext == "pdf":
                            from data_ingestion.parsers import PDFParser
                            parser = PDFParser()
                        elif file_ext in ("docx", "doc"):
                            from data_ingestion.parsers import WordParser
                            parser = WordParser()
                        elif file_ext in ("xlsx", "xls", "csv"):
                            from data_ingestion.parsers import ExcelParser
                            parser = ExcelParser()
                        elif file_ext == "pptx":
                            from data_ingestion.parsers import PPTXParser
                            parser = PPTXParser()
                        elif file_ext in ("txt", "md"):
                            from data_ingestion.parsers import TextParser
                            parser = TextParser()
                        else:
                            from data_ingestion.parsers import TextParser
                            parser = TextParser()

                        # 解析文件
                        parsed_text, _ = parser.parse(tmp_path)
                        parsed_file_content = parsed_text

                        logger.info(
                            "async_file_parsed",
                            extra={
                                "task_id": task_id,
                                "file_name": file_name,
                                "file_ext": file_ext,
                                "parsed_length": len(parsed_text) if parsed_text else 0,
                                "component": "tasks",
                            }
                        )

                        # 将文件内容融入 query，增强上下文
                        if parsed_text:
                            file_type_name = {
                                "pdf": "PDF 文档",
                                "docx": "Word 文档",
                                "doc": "Word 文档",
                                "xlsx": "Excel 表格",
                                "xls": "Excel 表格",
                                "csv": "CSV 文件",
                                "pptx": "PowerPoint 演示文稿",
                                "txt": "文本文件",
                                "md": "Markdown 文档",
                            }.get(file_ext, "文档")

                            # 如果已有图片描述，合并；否则创建新的消息
                            if multimodal_context:
                                input_state["messages"] = [
                                    HumanMessage(
                                        content=f"用户上传了一张图片，描述：{multimodal_context.get('description', '')}\n\n"
                                                f"用户还上传了一个 {file_type_name}《{file_name}》，内容如下：\n\n{parsed_text}\n\n"
                                                f"用户问题：{query}"
                                    )
                                ]
                            else:
                                input_state["messages"] = [
                                    HumanMessage(
                                        content=f"用户上传了一个 {file_type_name}《{file_name}》，内容如下：\n\n{parsed_text}\n\n"
                                                f"请根据以上内容回答用户问题：{query}"
                                    )
                                ]
                    finally:
                        # 删除临时文件
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

            except Exception as e:
                logger.warning(
                    "async_file_parse_failed",
                    extra={"task_id": task_id, "error": str(e), "component": "tasks"}
                )
                # 文件解析失败不影响主流程，继续执行

        # 更新状态：执行推理
        _update_task_status(
            task_id,
            status="progress",
            progress_percent=30,
            current_step="执行推理",
            current_message="正在执行推理...",
        )
        _publish_progress(
            task_id,
            event_type="task_progress",
            step="执行推理",
            step_description="正在执行推理...",
            progress_percent=30,
            message="正在处理您的请求...",
        )

        # 执行 LangGraph 工作流
        state_after_invoke = graph.invoke(input_state, config=config)
        latest_state = graph.get_state(config)
        requires_approval = latest_state.values.get("requires_approval", False)
        execution_success = latest_state.values.get("execution_success", True)
        execution_error = latest_state.values.get("execution_error")

        # 获取回复内容
        messages = state_after_invoke.get("messages", [])
        final_msg = messages[-1].content if messages else state_after_invoke.get("final_response", "")

        # 更新状态：处理结果
        _update_task_status(
            task_id,
            status="progress",
            progress_percent=60,
            current_step="处理结果",
            current_message="正在处理结果...",
        )
        _publish_progress(
            task_id,
            event_type="task_progress",
            step="处理结果",
            step_description="正在处理结果...",
            progress_percent=60,
            message="正在生成回复...",
        )

        # 如果启用流式推送，逐块发送文本
        if enable_streaming and final_msg:
            chunks = _split_text_for_streaming(final_msg, chunk_size=15)
            total_chunks = len(chunks)
            base_progress = 60
            progress_per_chunk = (30 / total_chunks) if total_chunks > 0 else 0

            for i, chunk in enumerate(chunks):
                chunk_progress = int(base_progress + (i * progress_per_chunk))

                _publish_progress(
                    task_id,
                    event_type="task_chunk",
                    step="生成回复",
                    step_description=f"正在生成回复 ({i+1}/{total_chunks})...",
                    progress_percent=chunk_progress,
                    message=chunk,
                    data={
                        "text": chunk,
                        "chunk_index": i,
                        "total_chunks": total_chunks,
                        "is_incremental": True,
                    },
                )

                # 小延迟，避免推送过快
                time.sleep(0.02)

        # 处理审批状态
        draft_action_data = None
        if requires_approval:
            action_payload = latest_state.values.get("action_payload", {})
            draft_action_data = {
                "action_type": action_payload.get("action_type", ""),
                "extracted_params": action_payload.get("params", {}),
                "payload": action_payload,
                "confirmation_message": final_msg,
            }

            _publish_progress(
                task_id,
                event_type="task_action_pending",
                step="等待审批",
                step_description="请确认操作详情",
                progress_percent=95,
                message="操作已准备好，请确认执行",
                data={
                    "draft_action": draft_action_data,
                    "confirmation_message": final_msg,
                },
            )
        else:
            _publish_progress(
                task_id,
                event_type="task_progress",
                step="生成回复",
                step_description="回复生成完成",
                progress_percent=90,
                message="回复生成完成",
            )

        # 更新状态：保存历史
        _update_task_status(
            task_id,
            status="progress",
            progress_percent=95,
            current_step="保存历史",
            current_message="正在保存对话历史...",
        )
        _publish_progress(
            task_id,
            event_type="task_progress",
            step="保存历史",
            step_description="正在保存对话历史...",
            progress_percent=95,
            message="正在保存历史记录...",
        )

        # ─────────────────────────────────────────
        # 槽位清理逻辑：根据执行结果决定是否清空槽位
        # ─────────────────────────────────────────
        # 如果执行成功或不需要审批（草稿模式），清空槽位
        # 如果执行失败，保留槽位让用户可以续填
        slots_to_save = {}
        if execution_success or not latest_state.values.get("pending_slots"):
            slots_to_save = {"pending_slots": [], "confirmed_slots": {}}
        else:
            # 保留槽位状态
            slots_to_save = {
                "pending_slots": latest_state.values.get("pending_slots", []),
                "confirmed_slots": latest_state.values.get("confirmed_slots", {}),
            }
            logger.info(
                "execution_failed_keeping_slots",
                extra={
                    "task_id": task_id,
                    "session_id": session_id,
                    "pending_slots": slots_to_save["pending_slots"],
                    "execution_error": execution_error,
                }
            )

        # 合并槽位到 state 用于保存
        state_for_save = {**latest_state.values, **slots_to_save}

        # 异步保存历史
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                status_value = SessionStatus.APPROVAL_PENDING.value if requires_approval else SessionStatus.COMPLETED.value
                loop.run_until_complete(
                    save_conversation_history(
                        session_id=session_id,
                        user_id=user_id,
                        state=state_for_save,
                        status=status_value,
                    )
                )
            finally:
                loop.close()
        except Exception as e:
            logger.warning(
                "history_save_in_task_failed",
                extra={"task_id": task_id, "error": str(e)}
            )

        # 计算耗时
        duration_ms = int((time.time() - start_time) * 1000)

        # 完成任务
        _update_task_status(
            task_id,
            status="success",
            progress_percent=100,
            current_step="完成",
            current_message="任务执行完成",
            final_response=final_msg,
            requires_approval=requires_approval,
            draft_action=draft_action_data,
        )
        _publish_progress(
            task_id,
            event_type="task_completed",
            step="完成",
            step_description="任务执行完成",
            progress_percent=100,
            message="任务执行完成",
            data={
                "final_response": final_msg,
                "requires_approval": requires_approval,
                "draft_action": draft_action_data,
                "duration_ms": duration_ms,
            },
        )

        logger.info(
            "async_chat_task_completed",
            extra={
                "task_id": task_id,
                "session_id": session_id,
                "duration_ms": duration_ms,
                "requires_approval": requires_approval,
            }
        )

        return {
            "task_id": task_id,
            "session_id": session_id,
            "status": "success",
            "requires_approval": requires_approval,
            "draft_action": draft_action_data,
            "final_response": final_msg,
            "duration_ms": duration_ms,
        }

    except Exception as e:
        logger.error(
            "async_chat_task_failed",
            extra={
                "task_id": task_id,
                "session_id": session_id,
                "error": str(e),
            }
        )

        # 发布失败事件
        _update_task_status(
            task_id,
            status="failure",
            error=str(e),
        )
        _publish_progress(
            task_id,
            event_type="task_failed",
            step="失败",
            step_description="任务执行失败",
            progress_percent=0,
            message=f"任务执行失败: {str(e)}",
            data={"error": str(e)},
        )

        raise


@app.task(bind=True, name="tasks.process_multimodal_async", max_retries=3)
def process_multimodal_async(
    self,
    image_data: str,
    query: str,
    session_id: str,
    user_id: str,
    user_token: str,
    mode: str,
    trace_id: str,
    enable_streaming: bool = True,
) -> Dict[str, Any]:
    """
    异步多模态分析任务（支持 SSE 流式推送）

    Args:
        image_data: 图片数据（base64 编码）
        query: 用户问题
        session_id: 会话ID
        user_id: 用户ID
        user_token: 用户Token
        mode: 分析模式
        trace_id: 追踪ID
        enable_streaming: 是否启用流式推送

    Returns:
        任务结果字典
    """
    task_id = self.request.id
    start_time = time.time()

    logger.info(
        "async_multimodal_task_started",
        extra={
            "task_id": task_id,
            "session_id": session_id,
            "mode": mode,
        }
    )

    try:
        # 初始化进度
        _update_task_status(
            task_id,
            status="started",
            progress_percent=5,
            current_step="初始化",
            current_message="正在准备多模态处理...",
        )
        _publish_progress(
            task_id,
            event_type="task_progress",
            step="初始化",
            step_description="正在准备多模态处理...",
            progress_percent=5,
            message="任务已接受，正在准备处理...",
        )

        # 导入多模态模块
        try:
            from multimodal import MultimodalProcessor
        except ImportError as e:
            raise RuntimeError(f"无法导入多模态模块: {e}")

        # 创建处理器
        _update_task_status(
            task_id,
            status="progress",
            progress_percent=10,
            current_step="加载模型",
            current_message="正在加载多模态模型...",
        )
        _publish_progress(
            task_id,
            event_type="task_progress",
            step="加载模型",
            step_description="正在加载多模态模型...",
            progress_percent=10,
            message="正在加载图像分析模型...",
        )

        # 使用 get_multimodal_processor 从配置读取模型信息
        from multimodal import get_multimodal_processor
        processor = get_multimodal_processor()

        # 图像分析
        _update_task_status(
            task_id,
            status="progress",
            progress_percent=30,
            current_step="图像分析",
            current_message="正在分析图像内容...",
        )
        _publish_progress(
            task_id,
            event_type="task_progress",
            step="图像分析",
            step_description="正在分析图像内容...",
            progress_percent=30,
            message="正在分析上传的图片...",
        )

        # 执行图像分析
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                processor.process(
                    image_data=image_data,
                    query=query,
                    user_id=user_id,
                    user_token=user_token,
                    mode=mode,
                )
            )
        finally:
            loop.close()

        # 处理结果
        _update_task_status(
            task_id,
            status="progress",
            progress_percent=70,
            current_step="处理结果",
            current_message="正在处理分析结果...",
        )

        # 提取分析结果
        analysis_result = result.get("analysis_result", {})
        final_msg = result.get("message", "图片分析完成")
        requires_approval = result.get("requires_approval", False)
        draft_action = result.get("draft_action")

        # 流式推送结果
        if enable_streaming and final_msg:
            chunks = _split_text_for_streaming(final_msg, chunk_size=15)
            base_progress = 70
            progress_per_chunk = 25 / len(chunks) if chunks else 0

            for i, chunk in enumerate(chunks):
                chunk_progress = int(base_progress + (i * progress_per_chunk))
                _publish_progress(
                    task_id,
                    event_type="task_chunk",
                    step="生成回复",
                    step_description=f"正在生成回复 ({i+1}/{len(chunks)})...",
                    progress_percent=chunk_progress,
                    message=chunk,
                    data={
                        "text": chunk,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "is_incremental": True,
                    },
                )
                time.sleep(0.02)

        # 完成
        _update_task_status(
            task_id,
            status="success",
            progress_percent=100,
            current_step="完成",
            current_message="任务执行完成",
            final_response=final_msg,
            requires_approval=requires_approval,
            draft_action=draft_action,
        )
        _publish_progress(
            task_id,
            event_type="task_completed",
            step="完成",
            step_description="任务执行完成",
            progress_percent=100,
            message="任务执行完成",
            data={
                "final_response": final_msg,
                "requires_approval": requires_approval,
                "draft_action": draft_action,
                "analysis_result": analysis_result,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
        )

        logger.info(
            "async_multimodal_task_completed",
            extra={"task_id": task_id, "session_id": session_id}
        )

        return {
            "task_id": task_id,
            "session_id": session_id,
            "status": "success",
            "requires_approval": requires_approval,
            "draft_action": draft_action,
            "final_response": final_msg,
            "analysis_result": analysis_result,
            "duration_ms": int((time.time() - start_time) * 1000),
        }

    except Exception as e:
        logger.error(
            "async_multimodal_task_failed",
            extra={"task_id": task_id, "error": str(e)}
        )

        _update_task_status(
            task_id,
            status="failure",
            error=str(e),
        )
        _publish_progress(
            task_id,
            event_type="task_failed",
            step="失败",
            step_description="任务执行失败",
            progress_percent=0,
            message=f"任务执行失败: {str(e)}",
            data={"error": str(e)},
        )

        raise


# =============================================================================
# 任务状态查询（同步函数，供 API 调用）
# =============================================================================

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """
    获取任务状态

    Args:
        task_id: 任务ID

    Returns:
        任务状态字典，不存在返回 None
    """
    import json
    import redis
    from urllib.parse import urlparse

    parsed = urlparse(settings.redis_url)
    r = redis.Redis(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password,
        decode_responses=True,
    )

    try:
        # 先从 Redis Hash 获取
        key = f"async_task:{task_id}"
        data = r.hgetall(key)

        if data:
            # 解析 JSON 字段
            if data.get("draft_action"):
                try:
                    data["draft_action"] = json.loads(data["draft_action"])
                except:
                    data["draft_action"] = None

            return data

        # 降级：从 Celery 结果后端获取
        from celery.result import AsyncResult
        async_result = AsyncResult(task_id, app=app)

        if async_result.ready():
            if async_result.successful():
                return {
                    "task_id": task_id,
                    "status": "success",
                    "result": async_result.result,
                }
            elif async_result.failed():
                return {
                    "task_id": task_id,
                    "status": "failure",
                    "error": str(async_result.result),
                }
            else:
                return {
                    "task_id": task_id,
                    "status": "cancelled",
                }
        else:
            return {
                "task_id": task_id,
                "status": "pending",
            }

    finally:
        r.close()


def cancel_task(task_id: str) -> bool:
    """
    取消任务

    Args:
        task_id: 任务ID

    Returns:
        是否成功取消
    """
    try:
        from celery.result import AsyncResult
        async_result = AsyncResult(task_id, app=app)

        if async_result.state in ("PENDING", "STARTED"):
            async_result.revoke(terminate=True)

            # 更新状态
            import redis
            from urllib.parse import urlparse
            parsed = urlparse(settings.redis_url)
            r = redis.Redis(
                host=parsed.hostname or "localhost",
                port=parsed.port or 6379,
                password=parsed.password,
            )
            try:
                key = f"async_task:{task_id}"
                r.hset(key, "status", "cancelled")
                r.hset(key, "updated_at", time.time())

                # 发布取消事件
                channel = f"task:{task_id}"
                import json
                event = {
                    "task_id": task_id,
                    "event_type": "task_cancelled",
                    "step": "取消",
                    "step_description": "任务已被取消",
                    "progress_percent": 0,
                    "message": "任务已被用户取消",
                    "timestamp": time.time(),
                }
                r.publish(channel, json.dumps(event))
            finally:
                r.close()

            return True
    except Exception:
        pass

    return False


# =============================================================================
# 附加功能：Schema 强约束集成类
# =============================================================================

class DraftSchemaIntegration:
    """
    草稿 Schema 强约束集成器

    【功能】
    1. 根据 action_type 获取 Schema 定义
    2. 计算缺失字段，生成提问
    3. 从对话中提取参数，填充草稿
    4. 提供校验失败的修复建议
    """

    @staticmethod
    def get_schema_fields(action_type: str) -> dict:
        """获取 Schema 字段定义"""
        try:
            from draft_schemas import SchemaRegistry
            schema_class = SchemaRegistry.get_schema_class(action_type)
            if schema_class:
                return {
                    "required": schema_class.get_required_fields(),
                    "optional": schema_class.get_optional_fields(),
                    "all": schema_class.get_all_fields() if hasattr(schema_class, 'get_all_fields') 
                           else schema_class.get_required_fields() + schema_class.get_optional_fields(),
                }
        except ImportError:
            pass
        return {"required": [], "optional": [], "all": []}

    @staticmethod
    def compute_missing(action_type: str, params: dict) -> List[str]:
        """计算缺失字段"""
        try:
            from draft_schemas import compute_missing_fields
            return compute_missing_fields(action_type, params)
        except ImportError:
            return []

    @staticmethod
    def build_questions(action_type: str, missing: List[str]) -> List[dict]:
        """生成提问列表"""
        try:
            from draft_schemas import build_field_questions
            return build_field_questions(action_type, missing)
        except ImportError:
            return [{"field": f, "question": f"请提供 {f}"} for f in missing]

    @staticmethod
    def get_field_description(action_type: str, field: str) -> str:
        """获取字段描述"""
        try:
            from draft_schemas import SchemaRegistry
            return SchemaRegistry.get_field_description(action_type, field)
        except ImportError:
            return field

    @staticmethod
    def validate_params(action_type: str, params: dict) -> Tuple[bool, List[str]]:
        """校验参数"""
        try:
            from draft_schemas import SchemaRegistry
            schema_class = SchemaRegistry.get_schema_class(action_type)
            if not schema_class:
                return False, ["未知的业务类型"]

            try:
                # 过滤空值
                filtered = {k: v for k, v in params.items() if v is not None and v != ""}
                instance = schema_class(**filtered)
                missing = instance.get_missing_fields()
                return len(missing) == 0, missing
            except Exception as e:
                return False, [str(e)]

        except ImportError:
            return True, []
