"""
API 路由模块

提供聊天接口、审批接口等核心业务 API
"""

# ========== 导入依赖包 ==========
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List
from pydantic import BaseModel, Field
from langgraph.types import Command
import logging
import uuid
import asyncio
import json
import re
import time
from datetime import datetime, timezone


from schemas import (  
    ChatRequest,  
    ChatResponse, 
    DraftAction,  
    ApproveRequest,  
    ApproveResponse,  
    AgentState,  
)
from graph import get_agent_graph
from config import settings  
from auth import (  
    verify_token,  
    check_action_permission,  
    TokenPayload,  
)
from middleware import StructuredLoggingMiddleware 
from metrics import (  
    set_active_sessions,  
    set_approval_pending,  
    increment_sessions_created,  
)

from api_sse_celery import format_task_sse_event as format_sse_event
from api_sse_celery import format_task_sse_comment as format_sse_comment

# ========== 初始化日志 ==========
logger = logging.getLogger("oa_agent.api")  

# =============================================================================
# 辅助函数
# =============================================================================

router = APIRouter(prefix="/api/v1", tags=["智能服务接口"])  # 所有接口都以 /api/v1 开头

# =============================================================================
# 审核操作端点
# =============================================================================


# =============================================================================
# 聊天接口 - 异步 SSE 模式
# =============================================================================

class FileUploadData(BaseModel):
    """
    文件上传数据 - 用户上传的文件内容
    
    支持的文件格式：PDF、Word (DOCX/DOC)、Excel (XLSX/XLS/CSV)、PowerPoint (PPTX)、纯文本 (TXT)
    """
    file_name: str = Field(..., description="文件名，包含扩展名，如 document.pdf")
    file_content: str = Field(..., description="Base64 编码的文件内容")
    file_type: Optional[str] = Field(None, description="文件 MIME 类型，如 application/pdf")


class ChatAsyncRequest(BaseModel):
    """异步聊天请求"""
    session_id: str = Field(..., description="会话唯一标识，UUID 格式")
    user_token: str = Field(..., description="用户鉴权 Token")
    query: str = Field(..., description="用户输入的查询内容", min_length=1, max_length=2000)
    image_data: Optional[str] = Field(None, description="Base64 编码的图片")
    file_data: Optional[FileUploadData] = Field(None, description="上传的文件内容，支持 PDF、Word、Excel、PPTX、TXT")


class ChatAsyncResponse(BaseModel):
    """异步聊天响应 - 返回 task_id"""
    task_id: str
    session_id: str
    message: str


@router.post("/chat/async/stream", response_model=ChatAsyncResponse)
async def chat_async_stream(
    request: Request,
    chat_request: ChatAsyncRequest,
    token_info: TokenPayload = Depends(verify_token),
) -> ChatAsyncResponse:
    """
    异步聊天接口 - 提交任务并通过 SSE 接收进度

    流程：
    1. POST /api/v1/chat/async/stream → 提交任务，返回 task_id
    2. GET /api/v1/chat/async/stream/{task_id} → SSE 流接收进度

    支持新对话和继续对话：
    - 新对话：前端生成新的 session_id
    - 继续对话：前端传递之前的 session_id，系统自动恢复上下文
    """
    session_id = chat_request.session_id
    user_token = chat_request.user_token
    query = chat_request.query
    image_data = chat_request.image_data
    file_data = chat_request.file_data

    trace_id = getattr(request.state, "trace_id", "")
    user_id = token_info.user_id

    logger.info(
        "chat_async_request",
        extra={
            "trace_id": trace_id,
            "session_id": session_id,
            "user_id": user_id,
            "query_length": len(query),
            "has_image": bool(image_data),
            "has_file": bool(file_data),
            "file_name": file_data.file_name if file_data else None,
            "component": "api",
        }
    )

    # ─────────────────────────────────────────
    # 检查是否是继续对话，验证 session_id
    # ─────────────────────────────────────────
    try:
        from graph import get_agent_graph
        graph = get_agent_graph()
        config = {
            "configurable": {
                "thread_id": session_id,
            }
        }
        saved_state = graph.get_state(config)
        if saved_state and saved_state.values:
            logger.info(
                "session_resume_detected",
                extra={
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "has_pending_slots": bool(saved_state.values.get("pending_slots")),
                    "has_confirmed_slots": bool(saved_state.values.get("confirmed_slots")),
                    "component": "api",
                }
            )
    except Exception as e:
        logger.warning(
            "session_check_warning",
            extra={"session_id": session_id, "error": str(e), "component": "api"}
        )

    # ─────────────────────────────────────────
    # 提交 Celery 任务
    # ─────────────────────────────────────────
    try:
        from tasks import submit_chat_task
        
        # 提交异步任务
        result = submit_chat_task.apply_async(
            kwargs={
                "query": query,
                "session_id": session_id,
                "user_id": user_id,
                "user_token": user_token,
                "trace_id": trace_id,
                "enable_streaming": True,
                "image_data": image_data,
                "file_data": file_data.model_dump() if file_data else None,
            }
        )
        
        task_id = result.id

        logger.info(
            "chat_async_task_submitted",
            extra={
                "trace_id": trace_id,
                "session_id": session_id,
                "task_id": task_id,
                "component": "api",
            }
        )

        return ChatAsyncResponse(
            task_id=task_id,
            session_id=session_id,
            message="任务已提交，正在处理中..."
        )

    except Exception as e:
        logger.error(
            "chat_async_submit_failed",
            extra={
                "trace_id": trace_id,
                "session_id": session_id,
                "error": str(e),
                "component": "api",
            }
        )
        raise HTTPException(status_code=500, detail=f"任务提交失败：{str(e)}")


@router.get("/chat/async/stream/{task_id}")
async def chat_async_stream_sse(
    request: Request,
    task_id: str,
    token_info: TokenPayload = Depends(verify_token),
):
    """
    SSE 流式接口 - 接收任务进度

    返回 SSE 流，包含：
    - task_accepted: 任务已接受
    - task_progress: 进度更新
    - task_chunk: 文本片段（打字机效果）
    - task_action_pending: 动作待确认
    - task_completed: 任务完成
    - task_failed: 任务失败
    - heartbeat: 心跳保活
    """
    from api_sse_celery import get_event_channel, format_task_sse_event, format_task_sse_comment

    trace_id = getattr(request.state, "trace_id", "")

    logger.info(
        "sse_connection_opened",
        extra={"trace_id": trace_id, "task_id": task_id, "component": "api"}
    )

    async def event_generator():
        event_channel = get_event_channel()
        channel_name = f"task:{task_id}"

        # 发送初始连接成功消息
        yield format_task_sse_comment("SSE 连接已建立")

        try:
            # 订阅 Redis Pub/Sub
            pubsub = await event_channel._get_redis()
            subscriber = pubsub.pubsub()
            await subscriber.subscribe(channel_name)

            # 发送任务已接受确认
            yield format_task_sse_event("task_accepted", {"task_id": task_id})

            # 持续接收事件
            last_event_time = time.time()
            heartbeat_interval = 15  # 15秒心跳

            while True:
                # 非阻塞方式读取消息
                message = await subscriber.get_message(ignore_subscribe_messages=True, timeout=1.0)

                if message and message.get("type") == "message":
                    last_event_time = time.time()
                    data = message.get("data", "{}")
                    
                    # 解析事件数据
                    try:
                        import json
                        event_data = json.loads(data) if isinstance(data, str) else data
                        event_type = event_data.get("event_type", "unknown")
                        
                        # 转发事件
                        yield format_task_sse_event(event_type, event_data)

                        # 检查是否完成
                        if event_type in ["task_completed", "task_failed", "task_cancelled"]:
                            break
                    except json.JSONDecodeError:
                        pass

                # 心跳保活
                current_time = time.time()
                if current_time - last_event_time > heartbeat_interval:
                    yield format_task_sse_event("heartbeat", {
                        "task_id": task_id,
                        "timestamp": current_time
                    })
                    last_event_time = current_time

                # 检查任务是否超时（30分钟）
                if current_time - last_event_time > 1800:
                    yield format_task_sse_event("task_failed", {
                        "task_id": task_id,
                        "error": "SSE 连接超时"
                    })
                    break

        except asyncio.CancelledError:
            logger.info(
                "sse_connection_cancelled",
                extra={"trace_id": trace_id, "task_id": task_id, "component": "api"}
            )
        except Exception as e:
            logger.error(
                "sse_connection_error",
                extra={"trace_id": trace_id, "task_id": task_id, "error": str(e), "component": "api"}
            )
            yield format_task_sse_event("task_failed", {
                "task_id": task_id,
                "error": str(e)
            })
        finally:
            # 清理
            try:
                await subscriber.unsubscribe(channel_name)
                await subscriber.close()
            except Exception:
                pass
            
            logger.info(
                "sse_connection_closed",
                extra={"trace_id": trace_id, "task_id": task_id, "component": "api"}
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# =============================================================================
#
# =============================================================================

@router.post("/approve", response_model=ApproveResponse)
async def approve_endpoint(
    request: Request,
    approve_request: ApproveRequest,
    token_info: TokenPayload = Depends(verify_token),
) -> ApproveResponse:
    """
    审核操作端点
    处理用户对Agent操作的审批或拒绝
    """
    import time
    start_time = time.perf_counter()

    session_id = approve_request.session_id
    user_token = approve_request.user_token
    action = approve_request.action

    trace_id = getattr(request.state, "trace_id", "")
    user_id = token_info.user_id

    # ─────────────────────────────────────────
    # 开启链路追踪
    # ─────────────────────────────────────────
    try:
        from observability import get_trace_collector
        collector = get_trace_collector()
        collector.start_trace(
            query=f"approve:{session_id}:{action}",
            user_id=user_id,
            metadata={"session_id": session_id, "action": action}
        )
    except ImportError:
        collector = None

    logger.info(
        "approve_request",
        extra={
            "trace_id": trace_id,
            "session_id": session_id,
            "user_id": user_id,
            "action": action,
            "component": "api",
        }
    )

    graph = get_agent_graph()
    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }

    try:
        saved_state = graph.get_state(config)
        if not saved_state or not saved_state.values:
            raise HTTPException(status_code=404, detail=f"未找到会话 {session_id} 的状态。")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="无法恢复会话状态。"㈠浼氳瘽鐘舵€併€?)

    if action == "approve":
       
        action_payload = saved_state.values.get("action_payload", {})
        action_type = action_payload.get("action_type", "")
        if action_type:
            check_action_permission(token_info, action_type)

        try:
            resumed_state = graph.invoke(
                Command(resume={"approved": True, "user_token": user_token}),
                config=config,
            )

            messages = resumed_state.get("messages", [])
            result_msg = messages[-1].content if messages else "操作已完成。"

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(
                "approve_success",
                extra={
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "action_type": action_type,
                    "duration_ms": duration_ms,
                    "component": "api",
                }
            )

            return ApproveResponse(
                session_id=session_id,
                status="success",
                message=result_msg,
            )

        except Exception as e:
            logger.error(
                "approve_resume_error",
                extra={
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "error": str(e),
                    "component": "api",
                }
            )
            return ApproveResponse(
                session_id=session_id,
                status="error",
                message=f"执行恢复失败：{str(e)}，请稍后重试。",鎭㈠澶辫触锛歿str(e)}锛岃绋嶅悗閲嶈瘯銆?,
            )

    else:
        from langchain_core.messages import AIMessage

        new_messages = list(saved_state.values.get("messages", []))
        new_messages.append(
            AIMessage(content="操作已取消。如需重新发起，请输入新的请求。")
        )

        graph.update_state(config, {"messages": new_messages, "interrupted": False})

        logger.info(
            "reject_success",
            extra={
                "trace_id": trace_id,
                "session_id": session_id,
                "user_id": user_id,
                "component": "api",
            }
        )

        return ApproveResponse(
            session_id=session_id,
            status="rejected",
            message="操作已取消。",
        )

