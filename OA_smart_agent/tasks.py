"""
Celery 异步任务队列 (tasks.py)

提供异步执行 LangGraph 工作流的能力，支持：
1. 异步聊天任务（带 SSE 流式推送）
2. 异步多模态分析任务
3. 任务状态查询和取消
4. LangGraph ReAct 模式 + Human-in-the-loop
5. 会话恢复（自动恢复历史上下文）
6. 文件解析（PDF/Word/Excel/PPTX/TXT）
7. 图片多模态分析

核心任务：
- submit_chat_task       — 异步聊天（支持图片、文件、ReAct 模式）
- process_multimodal_async — 独立多模态分析任务

进度推送：
- Celery Worker 执行过程中，通过 Redis Pub/Sub 推送进度事件
- SSE 接口订阅事件，实时推送给前端
- 支持打字机效果的文本片段推送
- 任务状态存储在 Redis Hash（24小时过期）

LangGraph ReAct 模式：
- graph.invoke() 在工具调用前自动中断（interrupt_before=["tools"]）
- 中断时返回 pending_tool_calls，供前端确认审批
- 用户确认后通过 api.py 的 approve/reject 接口继续执行

会话恢复：
- 自动检测历史状态（graph.get_state）
- 恢复消息上下文，继续对话

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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

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
            from history import save_conversation_history, SessionStatus
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

        # 构建初始状态（ReAct 模式简化版）
        input_state: AgentState = {
            "messages": messages_input,
            "user_token": user_token,
            "interrupted": False,
            "retrieved_docs": None,
            "session_id": session_id,
            "user_id": user_id,
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

        # 执行 LangGraph 工作流（流式模式）
        # 注意：在 ReAct 模式下，graph.stream() 会实时推送消息块，支持真正的流式输出
        token_buffer = ""
        last_publish_time = time.time()
        final_msg = ""
        
        # 使用 stream 替代 invoke，并监听消息流
        for event in graph.stream(input_state, config=config, stream_mode="messages"):
            # LangGraph 会实时推送消息块 (Chunk)
            msg_chunk = event[0]
            
            # 只有 AI 消息才处理
            if msg_chunk.type == "ai" and msg_chunk.content:
                token_buffer += msg_chunk.content
                final_msg += msg_chunk.content
                
                # 微批处理触发条件：缓冲区超过 15 个字，或者距离上次发送超过 0.2 秒
                current_time = time.time()
                if len(token_buffer) >= 15 or (current_time - last_publish_time) > 0.2:
                    _publish_progress(
                        task_id,
                        event_type="task_chunk",
                        step="生成回复",
                        step_description="AI 正在作答...",
                        progress_percent=80,
                        message="接收流式数据中",
                        data={
                            "text": token_buffer,
                            "is_incremental": True,
                        },
                    )
                    # 清空缓冲池，重置时间
                    token_buffer = ""
                    last_publish_time = current_time

        # 把最后一点没达到 15 个字的零星尾巴推出去
        if token_buffer:
            _publish_progress(
                task_id,
                event_type="task_chunk",
                step="生成回复",
                step_description="AI 正在作答...",
                progress_percent=85,
                message="接收流式数据中",
                data={
                    "text": token_buffer,
                    "is_incremental": True,
                },
            )
        
        # 流结束后，获取最终状态以检查中断和工具调用
        latest_state = graph.get_state(config)
        is_interrupted = latest_state.metadata.get("interrupted", False) if latest_state.metadata else False
        
        # 获取消息列表用于后续工具调用检查
        messages = latest_state.values.get("messages", [])

        # 只检查最后一条消息是否有待执行的业务工具调用（RAG 等自动工具不会有 tool_calls）
        APPROVAL_REQUIRED_TOOLS = {
            "leave_request",
            "expense_reimburse",
            "password_reset",
            "permission_open",
            "execute_action",
        }
        last_message = messages[-1] if messages else None
        pending_tool_calls = (
            [last_message]
            if (
                last_message
                and hasattr(last_message, 'tool_calls')
                and last_message.tool_calls
                and any(tc.name in APPROVAL_REQUIRED_TOOLS for tc in last_message.tool_calls)
            )
            else []
        )
        has_pending_tools = bool(pending_tool_calls)

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

        # 处理审批状态（ReAct 模式）
        # 在新模式下，"审批"由 LangGraph 的 interrupt_before=["tools"] 自动处理
        # 当图被中断时，latest_state.metadata 会标记为 interrupted
        requires_approval = is_interrupted or has_pending_tools
        draft_action_data = None
        
        if requires_approval:
            # 构建待审批信息（从 tool_calls 中提取）
            tool_info = pending_tool_calls[0] if pending_tool_calls else {}
            tool_name = tool_info.tool_calls[0].name if hasattr(tool_info, 'tool_calls') and tool_info.tool_calls else "unknown"
            tool_args = tool_info.tool_calls[0].args if hasattr(tool_info, 'tool_calls') and tool_info.tool_calls else {}
            
            draft_action_data = {
                "action_type": tool_name,
                "extracted_params": tool_args,
                "payload": {
                    "action_type": tool_name,
                    "params": tool_args,
                },
                "confirmation_message": f"系统准备执行【{tool_name}】，请确认是否继续？",
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
                    "confirmation_message": draft_action_data["confirmation_message"],
                    "is_interrupted": is_interrupted,
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
        # 状态保存（ReAct 模式简化版）
        # ─────────────────────────────────────────
        # 新模式下不再需要 pending_slots/confirmed_slots 字段
        # LangGraph 会自动管理对话状态
        state_for_save = latest_state.values

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


