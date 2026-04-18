"""
本地链路追踪和日志调试模块 (observability.py)

═══════════════════════════════════════════════════════════════════════════════
📚 核心能力
═══════════════════════════════════════════════════════════════════════════════

使用 LlamaIndex CallbackManager 实现全链路追踪：

1. LLM 调用追踪
   - 每次 LLM chat/complete 调用，自动记录：
     - Prompt 模板名称 + 变量注入情况
     - Token 消耗（prompt_tokens / completion_tokens / total_tokens）
     - 响应延迟（latency_ms）
     - 模型名称
     - 追踪 ID（trace_id / span_id）

2. 检索链路追踪
   - 向量检索（Milvus / Chroma / PGVector）
   - BM25 检索
   - 检索结果数量和质量

3. Rerank 后处理追踪
   - 输入文档数 / 输出文档数
   - 重排序耗时
   - 使用的模型名称

4. 端到端 RAG 链路追踪
   - 从 query 输入到最终答案的全流程
   - 每个节点的耗时占比
   - 错误链路标记

5. Structured Logging（日志结构化）
   - JSON 格式输出（兼容 ELK / Loki / Datadog）
   - 链路 ID 贯穿始终（方便 grep / 聚合分析）

═══════════════════════════════════════════════════════════════════════════════
🔧 核心组件
═══════════════════════════════════════════════════════════════════════════════

1. OACallbackHandler（继承自 CallbackHandler）
   - LlamaIndex 全局回调处理器
   - 注入到 Settings.callback_manager

2. OATraceCollector
   - 内存中的链路数据收集器
   - 支持按 trace_id 查询
   - 支持导出为 JSON / DataFrame

3. StructuredLogger
   - 结构化日志（JSON）
   - 自动附加 trace_id / span_id / user_id

═══════════════════════════════════════════════════════════════════════════════
📖 使用示例
═══════════════════════════════════════════════════════════════════════════════

```python
# 方式 1：在 main.py 中全局注册（推荐）
from observability import setup_observability
setup_observability()

# 之后所有 LlamaIndex / LangChain 调用自动追踪
from rag import build_llama_rag_pipeline
pipeline = build_llama_rag_pipeline()
results = pipeline.retrieve("年假怎么算")

# 方式 2：单独获取 CallbackHandler
from observability import get_oa_callback_handler
handler = get_oa_callback_handler()
llm.chat(messages, callbacks=[handler])

# 查询最近链路
from observability import get_trace_collector
collector = get_trace_collector()
traces = collector.get_recent_traces(limit=10)
```

═══════════════════════════════════════════════════════════════════════════════
🖥️ 链路追踪界面
═══════════════════════════════════════════════════════════════════════════════

追踪数据通过 loguru 输出为 JSON，每条日志包含：

{
  "event": "llm_chat",
  "trace_id": "abc123",
  "span_id": "def456",
  "parent_span_id": null,
  "timestamp": "2026-04-14T10:30:00.000Z",
  "latency_ms": 1250,
  "model": "gpt-4o-mini",
  "prompt_tokens": 512,
  "completion_tokens": 128,
  "total_tokens": 640,
  "component": "oa_agent",
  "level": "INFO"
}

查看方式：
  # 查看最近 10 条链路
  tail -f logs/observability.jsonl | jq .

  # 按 trace_id 聚合
  cat logs/observability.jsonl | jq -s 'group_by(.trace_id) | .[:5]'
```
"""

import functools
import json
import logging
import os
import time
import traceback
import uuid
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import (
    MISSING,
    asdict,
    dataclass,
    field,
)
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    TypeVar,
)
from weakref import WeakValueDictionary

from llama_index.core.callbacks import (
    BaseCallbackHandler,
    CBEventType,
    EventContext,
)
from llama_index.core.settings import Settings

logger = logging.getLogger("oa_agent.observability")

# =============================================================================
# Context Variables（链路 ID 跨函数传递）
# =============================================================================

_current_trace_id: ContextVar[Optional[str]] = ContextVar(
    "current_trace_id", default=None
)
_current_span_id: ContextVar[Optional[str]] = ContextVar(
    "current_span_id", default=None
)
_current_user_id: ContextVar[Optional[str]] = ContextVar(
    "current_user_id", default=None
)


def get_current_trace_id() -> Optional[str]:
    return _current_trace_id.get()


def get_current_span_id() -> Optional[str]:
    return _current_span_id.get()


def get_current_user_id() -> Optional[str]:
    return _current_user_id.get()


def set_trace_context(
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    """设置当前链路上下文（用于手动启动一条链路）"""
    if trace_id:
        _current_trace_id.set(trace_id)
    if span_id:
        _current_span_id.set(span_id)
    if user_id:
        _current_user_id.set(user_id)


def new_trace(
    user_id: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    开启一条新链路，返回 trace_id。

    用法：
        trace_id = new_trace(user_id="user123")
        # 后续所有操作自动关联到此 trace_id
    """
    trace_id = str(uuid.uuid4())[:12]
    _current_trace_id.set(trace_id)
    _current_user_id.set(user_id)
    _current_span_id.set(None)

    _log_structured(
        event="trace_start",
        trace_id=trace_id,
        user_id=user_id,
        metadata=metadata or {},
    )
    return trace_id


# =============================================================================
# 结构化日志（兼容 loguru + JSON）
# =============================================================================

_loguru_available = False
try:
    from loguru import logger as _loguru

    _loguru_available = True
except ImportError:
    _loguru = None


def _log_structured(
    level: str = "INFO",
    event: str = "",
    message: str = "",
    **kwargs,
) -> None:
    """
    输出结构化 JSON 日志（兼容 loguru）。

    自动附加：
    - timestamp: ISO 格式时间戳
    - trace_id / span_id: 链路追踪 ID
    - user_id: 当前用户
    - component: 固定为 "oa_agent"
    """
    trace_id = _current_trace_id.get()
    span_id = _current_span_id.get()
    user_id = _current_user_id.get()

    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "trace_id": trace_id,
        "span_id": span_id,
        "user_id": user_id,
        "component": "oa_agent",
        **kwargs,
    }

    # 附加 message（如果有）
    if message:
        log_data["message"] = message

    log_str = json.dumps(log_data, ensure_ascii=False, default=str)

    if _loguru_available:
        log_func = getattr(_loguru, level.lower(), _loguru.info)
        log_func(log_str)
    else:
        std_logger = logging.getLogger("oa_agent")
        std_level = getattr(logging, level.upper(), logging.INFO)
        std_logger.log(std_level, log_str)


# =============================================================================
# 链路数据收集器
# =============================================================================

@dataclass
class LLMSpan:
    """LLM 调用记录"""
    span_id: str = ""
    trace_id: str = ""
    parent_span_id: Optional[str] = None
    event_type: str = ""  # llm_chat / llm_complete
    model: str = ""
    prompt: str = ""
    response: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["latency_ms"] = round(d["latency_ms"], 2)
        return d


@dataclass
class RetrievalSpan:
    """检索调用记录"""
    span_id: str = ""
    trace_id: str = ""
    retriever_type: str = ""  # vector / bm25 / hybrid
    query: str = ""
    top_k: int = 0
    returned_count: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["latency_ms"] = round(d["latency_ms"], 2)
        return d


@dataclass
class RerankSpan:
    """重排序调用记录"""
    span_id: str = ""
    trace_id: str = ""
    model: str = ""
    input_count: int = 0
    output_count: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["latency_ms"] = round(d["latency_ms"], 2)
        return d


@dataclass
class Trace:
    """完整的链路记录"""
    trace_id: str = ""
    user_id: str = ""
    query: str = ""
    start_time: str = ""
    end_time: str = ""
    total_latency_ms: float = 0.0
    llm_spans: List[LLMSpan] = field(default_factory=list)
    retrieval_spans: List[RetrievalSpan] = field(default_factory=list)
    rerank_spans: List[RerankSpan] = field(default_factory=list)
    error: Optional[str] = None
    final_answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "query": self.query[:200] if self.query else "",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "llm_call_count": len(self.llm_spans),
            "total_prompt_tokens": sum(s.prompt_tokens for s in self.llm_spans),
            "total_completion_tokens": sum(s.completion_tokens for s in self.llm_spans),
            "total_tokens": sum(s.total_tokens for s in self.llm_spans),
            "retrieval_count": len(self.retrieval_spans),
            "rerank_count": len(self.rerank_spans),
            "error": self.error,
            "final_answer_preview": self.final_answer[:200] if self.final_answer else "",
            "metadata": self.metadata,
        }

    @property
    def total_llm_cost(self) -> float:
        """估算 LLM 成本（按 GPT-4o-mini 价格：$0.15/1M input, $0.6/1M output）"""
        prompt_cost = sum(s.prompt_tokens for s in self.llm_spans) / 1_000_000 * 0.15
        completion_cost = sum(s.completion_tokens for s in self.llm_spans) / 1_000_000 * 0.6
        return round(prompt_cost + completion_cost, 6)


class OATraceCollector:
    """
    内存链路收集器。

    功能：
    - 按 trace_id 存储链路
    - 按 user_id 查询
    - 导出 JSON / DataFrame
    - 计算链路耗时占比
    """

    def __init__(self, max_traces: int = 1000):
        self._traces: Dict[str, Trace] = {}
        self._max_traces = max_traces
        self._access_order: List[str] = []  # LRU 清理

    def _new_span_id(self) -> str:
        return str(uuid.uuid4())[:12]

    def _ensure_trace(self, trace_id: str) -> Trace:
        if trace_id not in self._traces:
            self._traces[trace_id] = Trace(trace_id=trace_id)
            self._access_order.append(trace_id)
            # LRU 清理
            if len(self._access_order) > self._max_traces:
                old = self._access_order.pop(0)
                self._traces.pop(old, None)
        return self._traces[trace_id]

    # ── LLM 追踪 ────────────────────────────────────────────────────────────

    def record_llm_start(
        self,
        event_type: str,
        model: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """LLM 调用开始，返回 span_id"""
        trace_id = _current_trace_id.get() or "no_trace"
        span_id = self._new_span_id()

        trace = self._ensure_trace(trace_id)
        span = LLMSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=_current_span_id.get(),
            event_type=event_type,
            model=model,
            prompt=prompt[:500],  # 截断避免内存
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        trace.llm_spans.append(span)
        _current_span_id.set(span_id)

        _log_structured(
            event="llm_start",
            event_type=event_type,
            model=model,
            span_id=span_id,
            prompt_preview=prompt[:100],
        )
        return span_id

    def record_llm_end(
        self,
        span_id: str,
        response: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        """LLM 调用结束"""
        trace_id = _current_trace_id.get() or "no_trace"
        trace = self._traces.get(trace_id)
        if not trace:
            return

        # 找到对应的 span
        span = next((s for s in trace.llm_spans if s.span_id == span_id), None)
        if not span:
            return

        span.response = response[:500]
        span.prompt_tokens = prompt_tokens
        span.completion_tokens = completion_tokens
        span.total_tokens = prompt_tokens + completion_tokens
        span.latency_ms = latency_ms
        span.error = error

        # 更新 total_latency
        trace.total_latency_ms += latency_ms

        _log_structured(
            event="llm_end",
            span_id=span_id,
            model=span.model,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=span.total_tokens,
            error=error,
        )

    # ── 检索追踪 ──────────────────────────────────────────────────────────

    def record_retrieval(
        self,
        retriever_type: str,
        query: str,
        top_k: int,
        returned_count: int,
        latency_ms: float,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """记录一次检索调用"""
        trace_id = _current_trace_id.get() or "no_trace"
        span_id = self._new_span_id()

        trace = self._ensure_trace(trace_id)
        span = RetrievalSpan(
            span_id=span_id,
            trace_id=trace_id,
            retriever_type=retriever_type,
            query=query[:200],
            top_k=top_k,
            returned_count=returned_count,
            latency_ms=latency_ms,
            error=error,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        trace.retrieval_spans.append(span)
        trace.total_latency_ms += latency_ms

        _log_structured(
            event="retrieval_end",
            retriever_type=retriever_type,
            top_k=top_k,
            returned_count=returned_count,
            latency_ms=round(latency_ms, 2),
            error=error,
        )
        return span_id

    # ── 重排序追踪 ─────────────────────────────────────────────────────────

    def record_rerank(
        self,
        model: str,
        input_count: int,
        output_count: int,
        latency_ms: float,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """记录一次重排序调用"""
        trace_id = _current_trace_id.get() or "no_trace"
        span_id = self._new_span_id()

        trace = self._ensure_trace(trace_id)
        span = RerankSpan(
            span_id=span_id,
            trace_id=trace_id,
            model=model,
            input_count=input_count,
            output_count=output_count,
            latency_ms=latency_ms,
            error=error,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        trace.rerank_spans.append(span)
        trace.total_latency_ms += latency_ms

        _log_structured(
            event="rerank_end",
            model=model,
            input_count=input_count,
            output_count=output_count,
            latency_ms=round(latency_ms, 2),
            error=error,
        )
        return span_id

    # ── 链路生命周期 ────────────────────────────────────────────────────────

    def start_trace(
        self,
        query: str,
        user_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """开启新链路"""
        trace_id = new_trace(user_id=user_id)
        trace = self._ensure_trace(trace_id)
        trace.query = query
        trace.user_id = user_id
        trace.start_time = datetime.now(timezone.utc).isoformat()
        if metadata:
            trace.metadata.update(metadata)
        return trace_id

    def end_trace(
        self,
        final_answer: str = "",
        error: Optional[str] = None,
    ) -> Optional[Trace]:
        """结束当前链路"""
        trace_id = _current_trace_id.get()
        if not trace_id or trace_id not in self._traces:
            return None

        trace = self._traces[trace_id]
        trace.end_time = datetime.now(timezone.utc).isoformat()
        trace.final_answer = final_answer
        trace.error = error

        _log_structured(
            event="trace_end",
            trace_id=trace_id,
            total_latency_ms=round(trace.total_latency_ms, 2),
            llm_call_count=len(trace.llm_spans),
            total_tokens=sum(s.total_tokens for s in trace.llm_spans),
            llm_cost_usd=trace.total_llm_cost,
            error=error,
        )
        return trace

    # ── 查询接口 ────────────────────────────────────────────────────────────

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        return self._traces.get(trace_id)

    def get_recent_traces(self, limit: int = 10) -> List[Trace]:
        """获取最近的链路（按 end_time 倒序）"""
        traces = [t for t in self._traces.values() if t.end_time]
        traces.sort(key=lambda t: t.end_time, reverse=True)
        return traces[:limit]

    def get_traces_by_user(self, user_id: str, limit: int = 10) -> List[Trace]:
        """按用户查询链路"""
        traces = [
            t for t in self._traces.values()
            if t.user_id == user_id and t.end_time
        ]
        traces.sort(key=lambda t: t.end_time, reverse=True)
        return traces[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        traces = list(self._traces.values())
        completed = [t for t in traces if t.end_time]
        errors = [t for t in completed if t.error]
        return {
            "total_traces": len(traces),
            "completed_traces": len(completed),
            "error_traces": len(errors),
            "total_llm_calls": sum(len(t.llm_spans) for t in traces),
            "total_tokens": sum(
                s.total_tokens for t in traces for s in t.llm_spans
            ),
            "avg_latency_ms": (
                sum(t.total_latency_ms for t in completed) / len(completed)
                if completed else 0
            ),
            "total_llm_cost_usd": sum(t.total_llm_cost for t in traces),
        }

    def export_json(self, filepath: str) -> None:
        """导出链路为 JSONL"""
        with open(filepath, "w", encoding="utf-8") as f:
            for trace in self._traces.values():
                f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"导出 {len(self._traces)} 条链路到 {filepath}")

    def clear(self) -> None:
        self._traces.clear()
        self._access_order.clear()


# =============================================================================
# LlamaIndex CallbackHandler
# =============================================================================

class OACallbackHandler(BaseCallbackHandler):
    """
    LlamaIndex 全局回调处理器。

    自动追踪：
    - LLM chat/complete 调用
    - 检索操作
    - Rerank 后处理
    - Chunking / Parsing

    用法：
        Settings.callback_manager.add_handler(OACallbackHandler(collector))
    """

    def __init__(
        self,
        collector: Optional[OATraceCollector] = None,
        trace_collector: Optional[OATraceCollector] = None,
    ):
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[],
        )
        self._collector = trace_collector or collector or OATraceCollector()

    @property
    def collector(self) -> OATraceCollector:
        return self._collector

    # ── LLM Events ─────────────────────────────────────────────────────────

    def on_chat_model_start(
        self,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        messages = []
        if payload:
            messages = payload.get("messages", [])
        prompt_text = self._messages_to_text(messages)

        model = (
            payload.get("model", "")
            if payload
            else kwargs.get("model", "")
        )
        self._collector.record_llm_start(
            event_type="llm_chat",
            model=model,
            prompt=prompt_text,
        )

    def on_chat_model_end(
        self,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        span_id = _current_span_id.get() or ""
        response = ""
        if payload:
            response = str(payload.get("response", ""))

        # 从 response 对象提取 token 用量
        prompt_tokens = 0
        completion_tokens = 0
        raw = payload.get("response") if payload else None
        if raw:
            try:
                usage = getattr(raw, "raw", {}) or {}
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("usage", {}).get("prompt_tokens", 0)
                    completion_tokens = usage.get("usage", {}).get("completion_tokens", 0)
            except Exception:
                pass

        self._collector.record_llm_end(
            span_id=span_id,
            response=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=kwargs.get("latency_ms", 0.0),
        )

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        pass  # 流式 token 追踪（可选）

    # ── Retrieval Events ─────────────────────────────────────────────────

    def on_retrieve(
        self,
        query: str,
        top_k: int,
        **kwargs,
    ) -> None:
        retriever_type = kwargs.get("retriever_name", "unknown")
        self._collector.record_retrieval(
            retriever_type=retriever_type,
            query=query,
            top_k=top_k,
            returned_count=0,  # 结束时更新
            latency_ms=0.0,
        )

    def onRetrieve(
        self,
        nodes: List[Any],
        query_bundle: Any = None,
        **kwargs,
    ) -> None:
        span_id = _current_span_id.get() or ""
        self._collector.record_retrieval(
            retriever_type=kwargs.get("retriever_name", "vector"),
            query=str(query_bundle) if query_bundle else "",
            top_k=len(nodes),
            returned_count=len(nodes),
            latency_ms=kwargs.get("latency_ms", 0.0),
        )

    # ── Rerank Events ────────────────────────────────────────────────────

    def on_postprocess_start(
        self,
        nodes: List[Any],
        query: str,
        **kwargs,
    ) -> None:
        pass

    def on_postprocess_end(
        self,
        nodes: List[Any],
        query: str,
        **kwargs,
    ) -> None:
        model = kwargs.get("model_name", "unknown")
        self._collector.record_rerank(
            model=model,
            input_count=kwargs.get("input_count", 0),
            output_count=len(nodes),
            latency_ms=kwargs.get("latency_ms", 0.0),
        )

    # ── Error Handling ───────────────────────────────────────────────────

    def on_error(self, error: Exception, **kwargs) -> None:
        span_id = _current_span_id.get() or ""
        _log_structured(
            event="error",
            level="ERROR",
            span_id=span_id,
            error=str(error),
            traceback=traceback.format_exc(),
        )

    # ── 辅助方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _messages_to_text(messages: Any) -> str:
        """将 messages 对象转换为文本（用于日志）"""
        if not messages:
            return ""
        try:
            if isinstance(messages, list):
                parts = []
                for msg in messages:
                    if hasattr(msg, "content"):
                        parts.append(str(msg.content))
                    elif isinstance(msg, dict):
                        parts.append(str(msg.get("content", "")))
                return "\n".join(parts)
        except Exception:
            pass
        return str(messages)[:500]


# =============================================================================
# 全局单例 & 便捷函数
# =============================================================================

_callback_handler: Optional[OACallbackHandler] = None
_trace_collector: Optional[OATraceCollector] = None


def get_trace_collector() -> OATraceCollector:
    """获取链路收集器全局单例"""
    global _trace_collector
    if _trace_collector is None:
        _trace_collector = OATraceCollector()
    return _trace_collector


def get_oa_callback_handler() -> Optional[OACallbackHandler]:
    """获取 LlamaIndex CallbackHandler 全局单例"""
    global _callback_handler
    if _callback_handler is None:
        _callback_handler = OACallbackHandler(
            collector=get_trace_collector()
        )
    return _callback_handler


def setup_observability(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    enable_console: bool = True,
) -> OACallbackHandler:
    """
    一键初始化全局可观测性。

    在 main.py 或 FastAPI startup 事件中调用。

    参数：
    - log_file: JSONL 日志输出路径（可选）
    - log_level: 日志级别（默认 INFO）
    - enable_console: 是否输出到控制台（默认 True）

    用法：
        from observability import setup_observability
        handler = setup_observability(log_file="./logs/observability.jsonl")
    """
    global _callback_handler

    collector = get_trace_collector()
    _callback_handler = OACallbackHandler(collector=collector)

    # 注册到 LlamaIndex Settings
    try:
        Settings.callback_manager.add_handler(_callback_handler)
        logger.info("OACallbackHandler 已注册到 LlamaIndex Settings.callback_manager")
    except Exception as e:
        logger.warning(f"无法注册 CallbackHandler 到 Settings: {e}")

    # 配置 loguru JSON 输出
    if _loguru_available and log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            _loguru.add(
                log_file,
                format="{message}",
                serialize=True,
                rotation="100 MB",
                retention="7 days",
                level=log_level,
            )
            if enable_console:
                _loguru.add(
                    lambda msg: print(msg, end=""),
                    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                           "<level>{level: <8}</level> | "
                           "<cyan>{extra[event]}</cyan> | "
                           "<level>{extra[message]}</level>",
                    level=log_level,
                )
        except Exception as e:
            logger.warning(f"loguru 配置失败: {e}")

    return _callback_handler


def trace_llm_call(
    event_type: str = "llm_chat",
    model: Optional[str] = None,
) -> Callable:
    """
    装饰器：为任意函数自动添加 LLM 调用追踪。

    用法：
        @trace_llm_call(event_type="answer_generate")
        def generate_answer(query, context):
            return llm.chat(messages)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 提取 prompt（假设第一个或第二个参数是 messages）
            prompt = ""
            for arg in list(args)[:3]:
                if isinstance(arg, str) and len(arg) > 10:
                    prompt = arg[:200]
                    break
            prompt = kwargs.get("prompt", prompt)

            start = time.perf_counter()
            span_id = collector.record_llm_start(
                event_type=event_type,
                model=model or "unknown",
                prompt=prompt,
            )
            try:
                result = func(*args, **kwargs)
                latency_ms = (time.perf_counter() - start) * 1000
                collector.record_llm_end(
                    span_id=span_id,
                    response=str(result)[:500] if result else "",
                    latency_ms=latency_ms,
                )
                return result
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                collector.record_llm_end(
                    span_id=span_id,
                    response="",
                    latency_ms=latency_ms,
                    error=str(e),
                )
                raise

        return wrapper
    collector = get_trace_collector()
    return decorator


# =============================================================================
# 导出清单
# =============================================================================

__all__ = [
    # 链路上下文
    "get_current_trace_id",
    "get_current_span_id",
    "get_current_user_id",
    "set_trace_context",
    "new_trace",
    # 收集器
    "OATraceCollector",
    "OACallbackHandler",
    "get_trace_collector",
    "get_oa_callback_handler",
    # 初始化
    "setup_observability",
    # 装饰器
    "trace_llm_call",
    # 数据类
    "LLMSpan",
    "RetrievalSpan",
    "RerankSpan",
    "Trace",
]
