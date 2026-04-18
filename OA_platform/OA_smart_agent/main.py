"""
应用入口模块 (main.py)

作用：
    这是整个 OA 智能 Agent 的"大门"，所有的 API 请求都从这里进入。
    就像一个公司前台，负责接待来访者（HTTP请求）并分配到正确的部门（各个功能模块）。

主要功能：
1. 配置 FastAPI 应用（web 框架）
2. 注册各种路由（API 端点）
3. 配置中间件（日志、CORS等）
4. 管理应用生命周期（启动和关闭时做什么）

模块依赖：
- config: 配置管理
- api: Agent 业务逻辑路由
- middleware: 结构化日志
- metrics: 监控指标
- canary: 健康检查
- auth: 用户鉴权
- mcp: MCP 协议服务
"""

# 导入 FastAPI 框架
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys

# 导入项目内部模块
from config import settings
from api import router as agent_router  # Agent 业务路由
from middleware import StructuredLoggingMiddleware  # 结构化日志中间件
from metrics import metrics_endpoint  # Prometheus 监控指标端点
from canary import get_readiness_probe, get_feature_flag_manager  # 健康检查和功能开关
from auth import cleanup_session  # 会话清理
from mcp.server import mcp  # MCP 服务器


# =============================================================================
# 应用生命周期管理
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期钩子，在应用启动和关闭时执行特定操作。

    启动时做的事情：
    1. 配置日志格式（JSON格式，便于日志系统分析）
    2. 初始化功能开关管理器
    3. 检查 PostgreSQL 数据库连接
    4. 初始化 MCP 服务器
    5. 记录启动日志

    关闭时做的事情：
    1. 清理资源
    2. 记录关闭日志
    """
    # ─────────────────────────────────────────
    # 启动阶段
    # ─────────────────────────────────────────

    # 配置 JSON 格式的日志（机器可读，便于日志分析系统处理）
    if settings.log_format == "json":
        from loguru import logger as loguru_logger
        # 移除默认的日志处理器
        loguru_logger.remove()
        # 添加新的处理器，输出到控制台
        loguru_logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
            level=settings.log_level.upper(),
            serialize=True,  # 输出 JSON 格式
        )

    # 初始化功能开关管理器（用于灰度发布、AB测试等）
    get_feature_flag_manager()

    # ─────────────────────────────────────────
    # 初始化 LlamaIndex 可观测性（链路追踪 + 日志）
    # ─────────────────────────────────────────
    try:
        from observability import setup_observability
        handler = setup_observability(
            log_file="./logs/observability.jsonl",
            log_level=settings.log_level.upper(),
        )
        logging.info(
            "LlamaIndex 可观测性初始化成功",
            extra={"component": "startup", "observability": "enabled"}
        )

        # 初始化全局 Settings.llm（LlamaIndex LLM 单例）
        from llama_index.core import Settings as li_settings
        if li_settings.llm is None:
            from prompts import get_prompt_llm
            li_settings.llm = get_prompt_llm()
            logging.info(
                "LlamaIndex Settings.llm 初始化成功",
                extra={"component": "startup", "llm_model": str(li_settings.llm.model)}
            )
    except Exception as e:
        logging.warning(
            "可观测性初始化失败，继续运行",
            extra={"error": str(e), "component": "startup"}
        )

    # 初始化健康检查探针
    probe = get_readiness_probe()

    # ─────────────────────────────────────────
    # 初始化 MCP 服务器
    # MCP（Model Context Protocol）用于标准化 Agent 调用外部工具的方式
    # ─────────────────────────────────────────
    if settings.mcp_enabled:
        try:
            await mcp.initialize()
            logging.info(
                "MCP服务器初始化成功",
                extra={
                    "server_name": settings.mcp_server_name,
                    "component": "startup",
                }
            )
        except Exception as e:
            logging.error(
                "MCP服务器初始化失败",
                extra={"error": str(e), "component": "startup"}
            )

    # 检查 PostgreSQL 数据库连接
    try:
        from sqlalchemy import create_engine, text
        # 获取数据库连接地址
        url = settings.postgres_checkpointer_url or settings.history_db_url
        if not url:
            raise ValueError("未配置 PostgreSQL 连接地址")
        # 创建数据库引擎
        engine = create_engine(url, connect_timeout=5)
        # 测试连接
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        # 更新健康检查状态
        probe.update_check("postgres", True)
        logging.info("PostgreSQL数据库连接成功", extra={"component": "startup"})
    except Exception as e:
        probe.update_check("postgres", False)
        logging.warning("PostgreSQL数据库不可用", extra={"error": str(e), "component": "startup"})

    # 记录应用启动日志
    logging.info(
        "应用启动完成",
        extra={
            "app_name": settings.app_name,
            "environment": settings.environment,
            "version": settings.app_version,
            "component": "startup",
        }
    )

    yield  # 应用运行中...

    # ─────────────────────────────────────────
    # 关闭阶段
    # ─────────────────────────────────────────
    logging.info("应用正在关闭", extra={"component": "startup"})


# =============================================================================
# FastAPI 应用初始化
# =============================================================================

# 创建 FastAPI 应用实例
app = FastAPI(
    title=settings.app_name,  # 应用名称（显示在 API 文档中）
    description="""
    智能服务台 Agent 微服务

    核心功能：
    - 政策咨询：基于混合检索 RAG 回答公司政策、制度、流程规定等知识性问题
    - 动作执行：通过人工确认机制执行请假、报销等操作

    技术特点：
    - 支持多轮对话，记住对话上下文
    - 基于用户权限返回个性化答案
    - 提供完整的监控和日志
    """,
    version=settings.app_version,  # 版本号
    docs_url="/docs",  # Swagger API 文档地址
    redoc_url="/redoc",  # ReDoc API 文档地址
    lifespan=lifespan,  # 生命周期钩子
)


# =============================================================================
# CORS 中间件配置
# =============================================================================
# CORS（跨域资源共享）配置，控制哪些网站可以访问我们的 API
# 白名单模式：只有列表中的网站才能访问，提高安全性

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,  # 从配置文件读取允许的网站
    allow_credentials=True,  # 是否允许携带凭证（cookies）
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # 允许的 HTTP 方法
    allow_headers=["Authorization", "Content-Type", "X-Trace-ID"],  # 允许的请求头
)


# =============================================================================
# 结构化日志中间件
# =============================================================================
# 每个请求都会自动记录日志，包含请求路径、耗时、状态码等信息
# 便于排查问题和监控服务运行状态

app.add_middleware(StructuredLoggingMiddleware)


# =============================================================================
# 注册 Agent 业务路由
# =============================================================================
# 核心业务逻辑在这里，包括聊天接口、审批接口等

app.include_router(agent_router)


# =============================================================================
# Mock Java 后端接口（仅用于开发调试）
# =============================================================================
# 在没有真正 Java 后端的情况下，模拟返回结果，方便前端开发和测试

from fastapi import APIRouter
from pydantic import BaseModel

# 创建 Mock 路由
mock_router = APIRouter(prefix="/mock/java", tags=["Mock Java 后端（开发调试用）"])


class MockExecuteRequest(BaseModel):
    """模拟执行请求的数据模型"""
    action_type: str  # 操作类型，如请假、报销等
    params: dict  # 操作参数
    request_id: str  # 请求ID，用于追踪
    metadata: dict  # 额外元数据


@mock_router.post("/api/execute")
async def mock_java_execute(request: MockExecuteRequest) -> dict:
    """
    通用执行接口（模拟）

    根据 action_type 返回不同的模拟响应。
    """
    action_type = request.action_type
    params = request.params
    request_id = request.request_id

    logging.info(
        "收到Mock Java请求",
        extra={
            "action_type": action_type,
            "request_id": request_id,
            "component": "mock_java",
        }
    )

    success = True
    message = ""

    # 根据不同操作类型返回不同的模拟消息
    if action_type == "leave_request":
        # 请假申请
        message = f"请假申请已提交成功！日期：{params.get('start_date', '未知')} 至 {params.get('end_date', '未知')}，原因：{params.get('reason', '未填写')}"
    elif action_type == "password_reset":
        # 密码重置
        message = "密码重置邮件已发送到您的企业邮箱，请在 24 小时内查收。"
    elif action_type == "expense_reimburse":
        # 费用报销
        message = f"报销申请已提交，金额：{params.get('amount', 0)} 元，预计 3 个工作日内到账。"
    else:
        # 其他操作
        message = f"操作 {action_type} 已执行成功。"

    return {
        "success": success,
        "message": message,
        "request_id": request_id,
        "timestamp": "2026-04-07T00:00:00Z",
    }


@mock_router.post("/api/leave/submit")
async def mock_leave_submit(request: MockExecuteRequest) -> dict:
    """
    请假申请接口（模拟）

    专门处理请假申请的接口
    """
    params = request.params
    logging.info(
        "收到请假申请（Mock）",
        extra={"request_id": request.request_id, "params": params}
    )
    return {
        "success": True,
        "message": f"请假申请已提交！日期：{params.get('start_date', '未知')} 至 {params.get('end_date', '未知')}，类型：{params.get('leave_type', '未知')}",
        "request_id": request.request_id,
    }


@mock_router.post("/api/expense/submit")
async def mock_expense_submit(request: MockExecuteRequest) -> dict:
    """
    费用报销接口（模拟）

    专门处理费用报销的接口
    """
    params = request.params
    logging.info(
        "收到报销申请（Mock）",
        extra={"request_id": request.request_id, "params": params}
    )
    return {
        "success": True,
        "message": f"报销申请已提交，金额：{params.get('amount', 0)} 元，类型：{params.get('expense_type', '未知')}",
        "request_id": request.request_id,
    }


@mock_router.post("/api/password/reset")
async def mock_password_reset(request: MockExecuteRequest) -> dict:
    """
    密码重置接口（模拟）

    专门处理密码重置的接口
    """
    params = request.params
    logging.info(
        "收到密码重置请求（Mock）",
        extra={"request_id": request.request_id, "params": params}
    )
    return {
        "success": True,
        "message": f"密码重置邮件已发送到您的企业邮箱，系统：{params.get('system_name', '未知')}",
        "request_id": request.request_id,
    }


@mock_router.post("/api/permission/open")
async def mock_permission_open(request: MockExecuteRequest) -> dict:
    """
    权限开通接口（模拟）

    专门处理权限开通的接口
    """
    params = request.params
    logging.info(
        "收到权限开通请求（Mock）",
        extra={"request_id": request.request_id, "params": params}
    )
    return {
        "success": True,
        "message": f"权限开通申请已提交，系统：{params.get('system_name', '未知')}，级别：{params.get('permission_level', '普通')}",
        "request_id": request.request_id,
    }


@mock_router.get("/health")
async def mock_java_health() -> dict:
    """
    健康检查接口（模拟）

    返回 Java 后端的服务状态
    """
    return {"status": "ok", "service": "mock-java-backend"}


# 将 Mock 路由注册到应用
app.include_router(mock_router)


# =============================================================================
# Token 鉴权路由（登出）
# =============================================================================
# 处理用户登出，清除会话信息

auth_router = APIRouter(prefix="/api/v1/auth", tags=["鉴权"])


@auth_router.post("/logout")
async def logout(session_id: str):
    """
    登出接口

    功能：
    1. 清理 PostgreSQL 中对应的会话状态
    2. 返回成功响应

    前端在用户点击登出时调用此接口。
    """
    cleanup_session(session_id)
    logging.info(
        "用户登出",
        extra={"session_id": session_id, "component": "auth"}
    )
    return {"message": "登出成功"}


app.include_router(auth_router)


# =============================================================================
# Prometheus 监控指标端点
# =============================================================================
# 暴露服务运行指标，供 Prometheus 监控系统抓取

@app.get("/metrics", tags=["监控"])
async def get_metrics():
    """
    GET /metrics

    返回 Prometheus 标准格式的监控指标，供监控服务器抓取。

    包含的指标：
    - oa_agent_llm_*：AI 模型调用次数/耗时
    - oa_agent_rag_*：知识库检索指标
    - oa_agent_java_api_*：Java 后端 API 调用指标
    - oa_agent_circuit_breaker_*：断路器状态
    - oa_agent_active_sessions_total：当前活跃会话数
    """
    return await metrics_endpoint()


# =============================================================================
# 功能开关管理接口
# =============================================================================
# 用于动态控制功能开启/关闭，支持灰度发布

feature_router = APIRouter(prefix="/api/v1/feature-flags", tags=["功能开关"])


@feature_router.get("/")
async def list_feature_flags():
    """
    获取所有功能开关状态

    返回每个功能开关的当前状态（开启/关闭/灰度比例）
    """
    fm = get_feature_flag_manager()
    return fm.get_all_flags()


@feature_router.put("/{flag_name}/rollout")
async def update_rollout(flag_name: str, percentage: float):
    """
    动态更新灰度比例

    无需重启服务，即可调整功能的开启比例。
    例如：设置 percentage=10，表示只有 10% 的用户能使用该功能
    """
    fm = get_feature_flag_manager()
    fm.update_rollout(flag_name, percentage)
    return {"message": f"功能开关 {flag_name} 灰度比例已更新为 {percentage}%"}


app.include_router(feature_router)


# =============================================================================
# 根路由
# =============================================================================

@app.get("/", tags=["根路由"])
async def root():
    """
    根路由，返回服务基本信息

    提供快速了解服务状态的接口
    """
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "running",
        "endpoints": {
            "chat_async_submit": "/api/v1/chat/async/stream",  # 异步聊天提交接口
            "chat_async_sse": "/api/v1/chat/async/stream/{task_id}",  # SSE 流式接收
            "approve": "/api/v1/approve",  # 审批接口
            "session_status": "/api/v1/session/{session_id}/status",  # 会话状态
            "logout": "/api/v1/auth/logout",  # 登出
            "feature_flags": "/api/v1/feature-flags/",  # 功能开关
            "metrics": "/metrics",  # 监控指标
            "docs": "/docs",  # API 文档
        }
    }


# =============================================================================
# 健康检查端点
# =============================================================================

@app.get("/health/ready", tags=["健康检查"])
async def health_ready():
    """
    GET /health/ready

    就绪探针，检查所有依赖服务是否就绪。

    用于 Kubernetes 判断 Pod 是否准备好接收流量。
    如果返回 unhealthy，Kubernetes 会把 Pod 从 Service 中摘除。

    返回状态：
    - healthy：所有依赖正常
    - degraded：部分依赖不可用，但 Agent 可以降级运行
    - unhealthy：核心依赖不可用，无法处理请求
    """
    probe = get_readiness_probe()
    is_ready, status = probe.is_ready()

    return {
        "status": status,
        "checks": probe.get_status(),
        "service": settings.app_name,
        "version": settings.app_version,
    }


# =============================================================================
# 应用启动入口
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # 打印启动信息
    print(f"\n{'='*60}")
    print(f"  {settings.app_name}")
    print(f"  版本：{settings.app_version} | 环境：{settings.environment}")
    print(f"{'='*60}")
    print(f"  主服务：http://127.0.0.1:8000")
    print(f"  监控指标：http://127.0.0.1:8000/metrics")
    print(f"  健康检查：http://127.0.0.1:8000/health/ready")
    print(f"  API 文档：http://127.0.0.1:8000/docs")
    print(f"{'='*60}\n")

    # 启动 uvicorn 服务器
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # 监听所有网络接口
        port=8000,  # 端口号
        reload=settings.debug,  # 调试模式下自动重载
        workers=1 if settings.debug else 4,  # 生产环境多进程运行
        log_level=settings.log_level.lower(),  # 日志级别
    )
