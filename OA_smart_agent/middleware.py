# -*- coding: utf-8 -*-
"""
中间件模块 (middleware.py)

包含：
1. StructuredLoggingMiddleware - 结构化日志中间件
"""

import logging
import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("oa_agent.middleware")


# =============================================================================
# 结构化日志中间件
# =============================================================================

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    结构化日志中间件

    功能：
    1. 为每个请求生成唯一的 trace_id
    2. 记录请求开始时间和详细信息
    3. 计算请求处理耗时
    4. 记录请求结束信息
    5. 在响应头中返回 trace_id
    6. 捕获异常，记录错误日志
    7. 同步 trace_id 到 observability 链路追踪上下文
    """

    async def dispatch(self, request: Request, call_next):
        # 1. 生成唯一的 trace_id
        trace_id = str(uuid.uuid4())

        # 将 trace_id 保存到请求状态中
        request.state.trace_id = trace_id

        # ─────────────────────────────────────────
        # 同步到 observability ContextVar（用于链路追踪）
        # ─────────────────────────────────────────
        try:
            from observability import set_trace_context
            set_trace_context(trace_id=trace_id)
        except ImportError:
            pass  # observability 模块未安装

        # 2. 记录开始时间
        start_time = time.perf_counter()

        # 3. 记录请求开始日志
        logger.info(
            "request_start",
            extra={
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("User-Agent", ""),
                "component": "middleware",
            }
        )

        try:
            # 4. 调用下一个处理器
            response = await call_next(request)

            # 5. 计算请求处理耗时
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # 6. 根据响应状态码决定日志级别
            log_level = "info"
            if response.status_code >= 500:
                log_level = "error"
            elif response.status_code >= 400:
                log_level = "warning"

            # 7. 记录请求结束日志
            getattr(logger, log_level)(
                "request_end",
                extra={
                    "trace_id": trace_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "component": "middleware",
                }
            )

            # 8. 在响应头中注入 trace_id
            response.headers["X-Trace-ID"] = trace_id

            return response

        except Exception as e:
            # 计算耗时
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # 记录错误日志
            logger.error(
                "request_error",
                extra={
                    "trace_id": trace_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "duration_ms": duration_ms,
                    "component": "middleware",
                }
            )

            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "detail": "服务器内部错误，请稍后重试。",
                    "trace_id": trace_id,
                },
                headers={"X-Trace-ID": trace_id}
            )
