"""
监控指标模块 (metrics.py)

只监控 Agent 核心指标：
1. LLM：调用量、耗时 P90/P99、错误率、并发数
2. RAG：检索成功率、检索耗时、文档召回数
3. Java API：各业务 action 耗时、错误率、调用量
4. 断路器：状态告警
5. 业务会话：活跃会话、待审批积压 Gauge

暴露方式：GET /metrics（Prometheus 标准格式）
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)
from starlette.responses import Response


# =============================================================================
# 1. LLM AI 模型调用指标
# =============================================================================

llm_requests_total = Counter(
    "oa_agent_llm_requests_total",
    "AI 模型请求总次数",
    ["model", "status"],  # status: success, error
)

llm_request_duration_seconds = Histogram(
    "oa_agent_llm_request_duration_seconds",
    "AI 模型请求耗时分布（秒）",
    ["model", "node"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],  # P50/P90/P99
)

llm_requests_in_flight = Gauge(
    "oa_agent_llm_requests_in_flight",
    "当前正在进行的 AI 请求数",
    ["model"],
)


# =============================================================================
# 2. RAG 知识库检索指标
# =============================================================================

rag_retrieval_total = Counter(
    "oa_agent_rag_retrieval_total",
    "知识库检索总次数",
    ["status"],  # status: success, error
)

rag_retrieval_duration_seconds = Histogram(
    "oa_agent_rag_retrieval_duration_seconds",
    "RAG 检索耗时分布（秒）",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)

rag_retrieved_docs_count = Histogram(
    "oa_agent_rag_retrieved_docs_count",
    "RAG 检索返回文档数量分布",
    buckets=[0, 1, 2, 3, 5, 10],
)


# =============================================================================
# 3. Java 后端 API 调用指标
# =============================================================================

java_api_requests_total = Counter(
    "oa_agent_java_api_requests_total",
    "Java 后端 API 请求总次数",
    ["action_type", "status"],  # status: success, error
)

java_api_duration_seconds = Histogram(
    "oa_agent_java_api_duration_seconds",
    "Java 后端 API 请求耗时分布（秒）",
    ["action_type"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],  # P99
)


# =============================================================================
# 4. 断路器指标
# =============================================================================

circuit_breaker_state = Gauge(
    "oa_agent_circuit_breaker_state",
    "断路器状态（0=关闭, 1=打开, 2=半开）",
    ["backend"],
)


# =============================================================================
# 5. 业务会话指标
# =============================================================================

active_sessions_total = Gauge(
    "oa_agent_active_sessions_total",
    "当前活跃会话数（等待用户确认）",
)

approval_pending_total = Gauge(
    "oa_agent_approval_pending_total",
    "当前待审批会话数",
)

total_sessions_created = Counter(
    "oa_agent_total_sessions_created_total",
    "历史创建的会话总数",
)


# =============================================================================
# 辅助函数
# =============================================================================

def record_llm_call(model: str, node: str, duration: float, success: bool):
    """记录 AI 模型调用"""
    status = "success" if success else "error"
    llm_requests_total.labels(model=model, status=status).inc()
    llm_request_duration_seconds.labels(model=model, node=node).observe(duration)


def increment_llm_in_flight(model: str, delta: int = 1):
    """增加/减少正在进行的 LLM 请求数"""
    llm_requests_in_flight.labels(model=model).inc(delta)


def record_rag_retrieval(duration: float, doc_count: int, success: bool):
    """记录知识库检索"""
    status = "success" if success else "error"
    rag_retrieval_total.labels(status=status).inc()
    rag_retrieval_duration_seconds.observe(duration)
    rag_retrieved_docs_count.observe(doc_count)


def record_java_api_call(action_type: str, duration: float, success: bool):
    """记录 Java 后端 API 调用"""
    status = "success" if success else "error"
    java_api_requests_total.labels(action_type=action_type, status=status).inc()
    java_api_duration_seconds.labels(action_type=action_type).observe(duration)


def set_circuit_breaker_state(backend: str, state: int):
    """设置断路器状态（0=关闭, 1=打开, 2=半开）"""
    circuit_breaker_state.labels(backend=backend).set(state)


def set_active_sessions(count: int):
    """设置活跃会话数"""
    active_sessions_total.set(count)


def set_approval_pending(count: int):
    """设置待审批会话数"""
    approval_pending_total.set(count)


def increment_sessions_created():
    """增加会话创建计数"""
    total_sessions_created.inc()


# =============================================================================
# Prometheus 指标端点
# =============================================================================

async def metrics_endpoint() -> Response:
    """GET /metrics - 返回 Prometheus 标准格式指标"""
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )
