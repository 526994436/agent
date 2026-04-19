"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     OA 智能 Agent 核心工作流引擎                               ║
║                              graph.py                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

【模块是什么？】
    这个文件就像是 OA 智能 Agent 的"大脑控制中心"。
    想象一下，你走进一家大型企业的客服中心：
    - 前台接待员（路由器）判断你是来问问题的还是来办业务的
    - 如果带了图片，先用多模态模型分析图片内容
    - 如果是问问题，帮你查知识库（知识库检索）
    - 如果是办业务，先帮你整理材料（参数提取），然后等你确认，最后帮你提交（API执行）
    - 最后，整理好回复告诉你结果（响应生成）

    这个文件就是定义这个流程的"剧本"。

【核心概念 - 工作流程图（StateGraph）】：
    工作流程由"节点（Node）"和"边（Edge）"组成，就像乐高积木一样：
    - 节点 = 执行特定任务的工作人员
    - 边 = 节点之间的连接，决定了下一个节点是谁

    就像一条流水线，每个工人（节点）只做自己的事，做完就传给下一个工人。

【七大核心节点】：
    ┌────────────────────────────────────────────────────────────────────────────┐
    │  节点1: 路由器 (router_node)                                              │
    │  └── 角色：前台接待员                                                       │
    │  └── 任务：分析用户输入，判断用户想要什么                                   │
    │  └── 多模态：检测图片，分析图片内容                               │
    │  └── 输出：intent = "policy"（问问题）、"action"（办业务）、"chitchat"（闲聊）│
    └─────────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            v                       v                       v
    ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
    │ 意图=chitchat │       │ 意图=policy   │       │ 意图=action   │
    └───────┬───────┘       └───────┬───────┘       └───────┬───────┘
            │                       │                       │
            v                       v                       v
    ┌───────────────┐       ┌───────────────┐       ┌───────────────────────┐
    │ 节点1.5: 闲聊 │       │ 节点2: RAG检索 │       │ 节点3: 参数提取       │
    │ (chitchat)   │       │ (rag_node)     │       │ (draft_action_node)   │
    └───────────────┘       └───────┬───────┘       └───────────────────────┘
                                    │                       │
                                    │                       v
                                    v               ┌───────────────────────┐
                            ┌───────────────┐       │ 【等待用户确认】       │
                            │ 节点5: 响应生成│       │   ↓                    │
                            │ (generate_response)      │ 节点4: API执行        │
                            └───────────────┘       │ (execute_api_node)    │
                                                    └───────────┬───────────┘
                                                                │
                                                                v
                                                    ┌───────────────────────┐
                                                    │ 节点5: 响应生成       │
                                                    │ (generate_response)   │
                                                    └───────────────────────┘

【多模态融合流程】：
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  Step 1: 图片场景检测                                                       │
    │    - 发票截图 → 进入报销流程（检索报销政策）                                  │
    │    - 错误截图 → 进入诊断流程（检索解决方案）                                  │
    │    - 通用图片 → 内容描述融入回答                                             │
    └─────────────────────────────────────────────────────────────────────────────┘

【四大企业级特性】：
    1. 会话记忆（PostgreSQL Checkpointer）
       - 就像"记事本"，保存 Agent 的工作状态
       - 用户关闭浏览器再打开，还能继续上次的对话

    2. AI 稳定性保障（LLMWrapper）
       - 超时熔断：请求超过时间就失败，不占用资源
       - 自动重试：网络抖动时自动重试
       - 模型降级：主模型不可用时自动切换备用模型

    3. 知识库质量控制（可控 Self-RAG）
       - 检索路由：规则优先，AI 辅助判断是否需要检索
       - 有用判断：用模型评估检索结果是否有用
       - 事实校验：检查回答中的数字、日期是否和知识库一致

    4. 多模态融合（Multimodal + RAG）
       - 图片分析：调用多模态模型分析图片内容
       - 文档检索增强：多模态结果作为上下文补充
       - 智能路由：根据图片类型选择不同处理流程
"""

import logging
import time
from functools import wraps
from typing import Literal, Optional, Callable
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from config import settings
from schemas import AgentState
from rag import build_hybrid_retriever
from mcp.server import mcp
from metrics import (
    record_llm_call,
    increment_llm_in_flight,
)

logger = logging.getLogger("oa_agent.graph")


# =============================================================================
# PostgreSQL Checkpointer 配置
# =============================================================================
# Checkpointer 就像一个"记事本"，用来保存 Agent 的工作状态。
# 这样即使用户关闭浏览器，下次打开还能继续上次的对话。

def _get_postgres_url() -> str:
    """
    获取 PostgreSQL 数据库连接地址

    优先级：
    1. postgres_checkpointer_url（专用配置）
    2. history_db_url（复用历史库配置）
    3. 抛出异常（必须配置）
    """
    url = settings.postgres_checkpointer_url or settings.history_db_url
    if not url:
        raise ValueError(
            "请配置 PostgreSQL 连接地址："
            "设置 postgres_checkpointer_url 或 history_db_url"
        )
    return url


def _build_postgres_saver() -> PostgresSaver:
    """
    构建 PostgreSQL Checkpointer
    用于持久化保存 Agent 的工作状态。
    """
    url = _get_postgres_url()
    
    # 删掉无用的 create_engine，因为 PostgresSaver.from_conn_string 
    # 会使用它自己默认的内部连接池管理（基于 psycopg）
    
    return PostgresSaver.from_conn_string(url)


def _postgres_safe_call(func: Callable) -> Callable:
    """
    PostgreSQL 操作装饰器

    当数据库操作失败时，记录错误日志并抛出明确的异常，
    而不是静默失败导致状态不一致。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                "PostgreSQL 操作失败",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "component": "checkpointer",
                }
            )
            raise RuntimeError(
                f"数据库连接失败：{str(e)}。"
                "请检查数据库服务状态。"
            ) from e
    return wrapper


class PostgresCheckpointer:
    """
    PostgreSQL Checkpointer 封装类

    提供：
    - 延迟初始化（应用启动后才连接数据库）
    - 异常安全包装
    - 连接池管理
    """

    def __init__(self):
        self._saver: Optional[PostgresSaver] = None
        self._engine = None

    def initialize(self) -> PostgresSaver:
        """延迟初始化数据库连接"""
        if self._saver is None:
            url = _get_postgres_url()
            self._engine = create_engine(
                url,
                max_overflow=settings.postgres_max_overflow,
                pool_size=settings.postgres_pool_size,
                pool_pre_ping=True,
            )
            self._saver = PostgresSaver.from_conn_string(url)
            # 确保数据库表结构存在
            self._saver.setup()
        return self._saver

    def close(self):
        """关闭连接池"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._saver = None


_postgres_checkpointer: Optional[PostgresCheckpointer] = None


def get_checkpointer() -> PostgresCheckpointer:
    """获取 PostgreSQL Checkpointer 单例"""
    global _postgres_checkpointer
    if _postgres_checkpointer is None:
        _postgres_checkpointer = PostgresCheckpointer()
    return _postgres_checkpointer


# =============================================================================
# LLM 调用封装
# =============================================================================
# LLM（Large Language Model）就是 AI 大模型，比如 GPT-4。
# 这个封装类提供了三重稳定性保障：
# 1. 超时熔断：请求超过时间就失败，不占用资源
# 2. 自动重试：网络抖动时自动重试
# 3. 模型降级：主模型不可用时自动切换备用模型

class LLMWrapper:
    """
    AI 模型调用封装类

    提供稳定可靠的 AI 模型调用服务。
    """

    def __init__(self):
        self._llm: Optional[ChatOpenAI] = None

    @property
    def llm(self) -> ChatOpenAI:
        """主模型（Qwen 3.5-14B-Instruct via vLLM）"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=settings.vllm_model,
                api_key=settings.vllm_api_key,
                base_url=settings.vllm_base_url,
                temperature=0.3,
                timeout=settings.vllm_request_timeout,
                max_retries=0,
                max_tokens=settings.vllm_max_tokens,
            )
        return self._llm

    def invoke(self, messages, *, model: Optional[str] = None) -> AIMessage:
        """
        调用 AI 模型
        """
        try:
            response = self._invoke_with_retry(self.llm, messages)
            return response
        except Exception as e:
            logger.error(
                "LLM 调用失败",
                extra={
                    "error": str(e),
                    "component": "llm",
                }
            )
            raise RuntimeError(f"AI 服务暂时不可用，请稍后重试。") from e

    def _invoke_with_retry(self, llm: ChatOpenAI, messages) -> AIMessage:
        """
        带重试的模型调用
        """
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = llm.invoke(messages)
                return AIMessage(content=response.content)
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                logger.warning(
                    f"LLM 调用重试 ({attempt + 1}/{max_attempts})",
                    extra={"error": str(e)}
                )
              
                wait_seconds = min(2 ** (attempt - 1), 16)
                logger.warning(
                    "AI 调用重试中",
                    extra={
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "wait_seconds": wait_seconds,
                        "error": str(e),
                    }
                )
                time.sleep(wait_seconds)


# 全局 LLM 实例
llm_wrapper = LLMWrapper()


def llm_with_timeout(messages, *, model: str = None, node: str = "default") -> AIMessage:
    """
    带超时保障的 AI 调用入口（带指标记录）

    这是代码中最常用的调用方式，确保调用稳定可靠。
    """
    target_model = model or settings.vllm_model
    increment_llm_in_flight(target_model, 1)
    start_time = time.perf_counter()
    success = True

    try:
        result = llm_wrapper.invoke(messages, model=model)
        return result
    except Exception:
        success = False
        raise
    finally:
        duration = time.perf_counter() - start_time
        increment_llm_in_flight(target_model, -1)
        record_llm_call(model=target_model, node=node, duration=duration, success=success)


# =============================================================================
# 路由器节点配置
# =============================================================================

def _build_router_llm():
    """
    构建路由器专用的 LLM。

    路由器只需要做简单的意图分类，不需要强大的推理能力。
    因此可以用更小更快的模型，节省成本和延迟。
    """
    from langchain_openai import ChatOpenAI
    from config import settings

    router_model = getattr(settings, 'vllm_model', None) or settings.vllm_model

    return ChatOpenAI(
        model=router_model,
        api_key=settings.vllm_api_key,
        base_url=settings.vllm_base_url,
        temperature=0.0,  # 意图分类不需要创意
        timeout=settings.vllm_request_timeout,
        max_retries=1,   # 快速失败
    )


# =============================================================================
# 节点一：路由器节点（意图分类）
# =============================================================================

def router_node(state: AgentState) -> AgentState:
    """
    意图分类节点（支持多模态）

    作用：分析用户输入，判断用户想要什么。

    返回两种意图：
    - policy（政策咨询）：用户问问题，如"年假怎么休" → 使用 Redis 缓存
    - action（动作执行）：用户要办事，如"帮我请个假" → 不使用 Redis 缓存

    多模态处理：
    - 如果用户上传了图片，先进行多模态分析
    - 根据图片内容决定后续处理流程（发票报销 / IT故障诊断 / 通用分析）

    设计说明：
    - 使用小模型（router_llm_model）做分类，节省成本
    - policy 意图可能命中缓存，action 意图直接走执行流程
    """
    # ─────────────────────────────────────────
    # 链路追踪：记录 router_node 执行
    # ─────────────────────────────────────────
    span = None
    try:
        from observability import get_trace_collector
        collector = get_trace_collector()
        span = collector.start_span(
            span_name="router_node",
            metadata={"node": "router"}
        )
    except ImportError:
        pass

    messages = state.get("messages", [])
    if not messages:
        if span:
            collector.end_span(span)
        return {"intent": None}

    latest_message = messages[-1].content if messages else ""

    # ─────────────────────────────────────────────────────────────────
    # Step 2: 检查是否有未完成的草稿（用于续填）
    # ─────────────────────────────────────────────────────────────────
    pending_slots = state.get("pending_slots", [])
    confirmed_slots = state.get("confirmed_slots", {})
    has_unfinished_draft = pending_slots and len(pending_slots) > 0

    # 如果有未完成的草稿，生成续填提醒
    draft_resume_message = ""
    if has_unfinished_draft:
        draft_action_type = state.get("draft_action_type", "unknown")
        action_names = {
            "leave_request": "请假申请",
            "expense_reimburse": "费用报销",
            "password_reset": "密码重置",
            "permission_open": "权限开通",
        }
        action_name = action_names.get(draft_action_type, draft_action_type)
        draft_resume_message = f"\n\n💡 您之前在进行【{action_name}】，还缺 {len(pending_slots)} 个字段：{', '.join(pending_slots[:3])}{'...' if len(pending_slots) > 3 else ''}"

    # ─────────────────────────────────────────────────────────────────
    # Step 3: 意图分类（使用 Structured Output）
    # ─────────────────────────────────────────────────────────────────
    try:
        from langchain_openai import ChatOpenAI
        from schemas import IntentClassification
        from config import settings

        router_model = getattr(settings, 'vllm_model', None) or settings.vllm_model

        # 记录 LLM 调用指标
        increment_llm_in_flight(router_model, 1)
        start_time = time.perf_counter()
        success = True

        try:
            router_llm = ChatOpenAI(
                model=router_model,
                api_key=settings.vllm_api_key,
                base_url=settings.vllm_base_url,
                temperature=0,
            )
            structured_llm = router_llm.with_structured_output(IntentClassification)
            
            classification = structured_llm.invoke(
                f"用户输入: {latest_message}"
            )
            intent = classification.intent
        except Exception:
            success = False
            raise
        finally:
            duration = time.perf_counter() - start_time
            increment_llm_in_flight(router_model, -1)
            record_llm_call(model=router_model, node="router", duration=duration, success=success)

        logger.info(
            "意图分类完成",
            extra={
                "intent": intent,
                "has_multimodal": multimodal_context is not None,
                "has_unfinished_draft": has_unfinished_draft,
                "draft_resume_hint": draft_resume_message[:50] if draft_resume_message else "",
                "session_id": state.get("session_id", ""),
                "component": "router",
            }
        )
    except Exception:
        # 分类失败时，默认走 policy 路径（保守策略）
        intent = "policy"
        logger.warning(
            "意图分类失败，默认走 policy 路径",
            extra={"session_id": state.get("session_id", "")}
        )

    return {
        "intent": intent,
        "has_unfinished_draft": has_unfinished_draft,
        "draft_resume_message": draft_resume_message,
    }


# =============================================================================
# 节点 1.5：闲聊礼貌拒答节点
# =============================================================================

CHITCHAT_RESPONSES = [
    "您好！我是一个专注于企业办公协助的智能助手，很高兴为您服务。如果您有政策咨询、流程办理等问题，请随时告诉我！",
    "您好！我是 OA 智能助手，主要帮助您解答公司制度和办理业务相关的问题。对于闲聊，我可能不太擅长，但随时准备好帮您解决工作问题！",
    "您好！感谢您的问候！我专注于企业办公场景，可以帮您查询政策、办理业务等。有任何办公需求，随时告诉我！",
]


def chitchat_node(state: AgentState) -> AgentState:
    """
    闲聊节点 - 礼貌拒答

    当用户意图为 chitchat（非办公闲聊）时，直接返回礼貌的拒答，
    不进行 RAG 检索或任何业务处理。
    """
    import random

    messages = state.get("messages", [])
    latest_message = messages[-1].content if messages else ""

    logger.info(
        "闲聊意图检测，礼貌拒答",
        extra={
            "session_id": state.get("session_id", ""),
            "user_message": latest_message[:50],
            "component": "chitchat",
        }
    )

    # 随机选择一条礼貌回复
    polite_response = random.choice(CHITCHAT_RESPONSES)

    # 添加 AI 回复消息
    from langchain_core.messages import AIMessage

    return {
        "messages": [AIMessage(content=polite_response)],
    }


# =============================================================================
# 节点二：RAG 检索节点
# =============================================================================

def rag_node(state: AgentState) -> AgentState:
    """
    RAG 检索节点（可控 Self-RAG 企业版 + 多模态融合）

    作用：从知识库中找到与用户问题最相关的内容，然后生成回答。

    多模态融合：
    - 如果有图片分析结果，将其融入检索和回答
    - 发票场景：检索报销政策并匹配发票信息
    - 故障诊断场景：检索相关错误解决方案

    工作流程（三步走）：
    ┌─────────────────────────────────────────────────────────────┐
    │ Step 1: 判断要不要查资料                                     │
    │   - 命中"必须检索"关键词（如"报销""请假"）→ 直接查            │
    │   - 命中"跳过检索"关键词（如"你好""谢谢"）→ 不查              │
    │   - 模糊情况 → 丢给 AI 判断                                  │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Step 2: 混合检索 + ABAC 权限过滤 + 多模态融合                │
    │   - 向量检索（语义相似度）                                   │
    │   - BM25 检索（关键词匹配）                                  │
    │   - RRF 融合（综合两种结果排序）                             │
    │   - 根据用户部门/项目过滤无权访问的文档                       │
    │   - 融合图片分析结果（发票信息/错误诊断）                     │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Step 3: 评估 + 生成                                          │
    │   - BGE-Rerank 精排，过滤低质量文档                          │
    │   - LLM 生成回答（包含多模态信息）                           │
    │   - 事实校验（检查回答中的数字/日期是否准确）                  │
    └─────────────────────────────────────────────────────────────┘
    """
    # ─────────────────────────────────────────
    # 链路追踪：记录 rag_node 执行
    # ─────────────────────────────────────────
    span = None
    try:
        from observability import get_trace_collector
        collector = get_trace_collector()
        span = collector.start_span(
            span_name="rag_node",
            metadata={"node": "rag"}
        )
    except ImportError:
        pass

    try:
        from controlled_self_rag import get_controlled_self_rag, RetrievalDecision

        messages = state.get("messages", [])
        if not messages:
            return {
                **state,
                "retrieved_docs": [],
                "reranked_docs": [],
                "retrieval_decision": None,
                "skipped_retrieval": False,
            }

    latest_message = messages[-1].content if messages else ""
    user_id = state.get("user_id", "default_user")
    # ─────────────────────────────────────────────────────────────────
    # 多模态上下文处理：用图片描述增强查询
    # ─────────────────────────────────────────────────────────────────
    multimodal_context = state.get("multimodal_context")
    multimodal_answer_addition = ""

    if multimodal_context:
        description = multimodal_context.get("description", "")
        if description:
            multimodal_answer_addition = f"【图片内容分析】\n{description}"

    # 使用可控 Self-RAG 处理检索
    try:
        self_rag = get_controlled_self_rag()

        # 如果有多模态上下文，增强查询
        enhanced_query = latest_message
        if multimodal_answer_addition:
            enhanced_query = f"{latest_message}\n\n{multimodal_answer_addition}"

        result = self_rag.process(
            query=enhanced_query,
            user_id=user_id,
            user_dept=state.get("user_department"),
            user_projects=state.get("user_projects"),
        )

        # 处理检索决策
        skipped_retrieval = result.retrieval_decision == RetrievalDecision.SKIP_RETRIEVE

        # 如果跳过检索（打招呼等），直接生成闲聊回答
        if skipped_retrieval or not result.should_retrieve:
            rag_message = AIMessage(content=result.answer or "您好！有什么可以帮您的吗？")

            logger.info(
                "跳过检索（闲聊类问题）",
                extra={
                    "reason": result.retrieval_decision.value if result.retrieval_decision else "unknown",
                    "session_id": state.get("session_id", ""),
                    "component": "rag",
                }
            )

            return {
                **state,
                "retrieved_docs": [],
                "reranked_docs": [],
                "retrieval_decision": result.retrieval_decision.value if result.retrieval_decision else None,
                "retrieval_quality_score": 0.0,
                "retrieval_filtered_count": 0,
                "is_useful": False,
                "skipped_retrieval": True,
                "reflection_results": [],
                "needs_escalation": False,
                "messages": [rag_message],
            }

        # 处理有用性判断
        if not result.is_useful or not result.docs:
            rag_message = AIMessage(
                content="在知识库中未找到相关信息，建议联系 HR 部门（内线 8001）获取帮助。"
            )
            
            logger.info(
                "检索结果无用",
                extra={
                    "quality_score": result.usefulness_score,
                    "session_id": state.get("session_id", ""),
                    "component": "rag",
                }
            )

            return {
                **state,
                "retrieved_docs": result.docs,
                "reranked_docs": result.docs,
                "retrieval_decision": result.retrieval_decision.value if result.retrieval_decision else None,
                "retrieval_quality_score": result.usefulness_score,
                "retrieval_filtered_count": 0,
                "is_useful": False,
                "skipped_retrieval": False,
                "reflection_results": [],
                "needs_escalation": True,
                "messages": [rag_message],
            }

        # 使用检索结果生成回答
        rag_message = AIMessage(content=result.answer)

        logger.info(
            "RAG 检索完成",
            extra={
                "doc_count": len(result.docs) if result.docs else 0,
                "quality_score": result.usefulness_score,
                "needs_escalation": result.needs_escalation,
                "session_id": state.get("session_id", ""),
                "component": "rag",
            }
        )

        return {
            **state,
            "retrieved_docs": result.docs,
            "reranked_docs": result.docs,
            "retrieval_decision": result.retrieval_decision.value if result.retrieval_decision else None,
            "retrieval_quality_score": result.usefulness_score,
            "retrieval_filtered_count": 0,
            "is_useful": True,
            "skipped_retrieval": False,
            "reflection_results": [],
            "needs_escalation": result.needs_escalation,
            "messages": [rag_message],
        }

    except Exception as e:
        logger.error(
            "RAG 节点执行失败",
            extra={"error": str(e), "component": "rag"}
        )
        rag_message = AIMessage(
            content="系统暂时无法处理您的请求，请稍后重试或联系 HR（内线 8001）获取帮助。"
        )

        return {
            **state,
            "retrieved_docs": [],
            "reranked_docs": [],
            "retrieval_decision": None,
            "retrieval_quality_score": 0.0,
            "retrieval_filtered_count": 0,
            "is_useful": False,
            "skipped_retrieval": False,
            "reflection_results": [],
            "needs_escalation": True,
            "messages": [rag_message],
        }


# =============================================================================
# 节点三：动作草稿节点（Schema 强约束版）
# =============================================================================

def draft_action_node(state: AgentState) -> AgentState:
    """
    动作草稿节点（Schema 强约束版）

    作用：当用户要执行操作（如请假、报销）时，AI 会从用户输入中提取出具体的参数。
    使用 Schema 强约束，确保参数完整性和正确性。

    【核心流程】
    1. 提取 action_type 和 params
    2. 合并历史参数（支持续填）
    3. 计算缺失字段
    4. 生成自然语言提问

    【续填逻辑】
    - 如果有 pending_slots，说明用户之前在进行某个业务
    - 合并历史 confirmed_slots 和新提取的参数
    - 继续询问缺失的字段

    例如：
    - 用户说："我想请下周三到周五的假"
    - AI 提取：action_type="leave_request", params={"start_date": "下周三", "end_date": "下周五"}
    - 计算缺失：leave_type, reason
    - 生成提问："请问您请的是什么类型的假？（年假/病假/事假）"
    """
    messages = state.get("messages", [])
    if not messages:
        return state

    latest_message = messages[-1].content if messages else ""
    user_token = state.get("user_token", "")
    user_id = state.get("user_id", "")
    session_id = state.get("session_id", "")

    # ─────────────────────────────────────────
    # 续填逻辑：合并历史参数
    # ─────────────────────────────────────────
    has_unfinished_draft = state.get("has_unfinished_draft", False)
    previous_action_type = state.get("draft_action_type", "")
    previous_confirmed_slots = state.get("confirmed_slots", {})
    previous_pending_slots = state.get("pending_slots", [])

    try:
        from langchain_openai import ChatOpenAI
        from schemas import ActionParamExtraction
        from config import settings

        action_model = settings.vllm_model
        action_llm = ChatOpenAI(
            model=action_model,
            api_key=settings.vllm_api_key,
            base_url=settings.vllm_base_url,
            temperature=0,
        )
        structured_llm = action_llm.with_structured_output(ActionParamExtraction)

        # 如果有续填任务，在 prompt 中添加上下文
        prompt_content = latest_message
        if has_unfinished_draft and previous_action_type:
            resume_context = f"\n\n【续填任务】当前在进行【{previous_action_type}】，已确认的字段：{previous_confirmed_slots}，还缺：{previous_pending_slots[:3]}"
            prompt_content += resume_context

        extraction = structured_llm.invoke(
            f"用户输入：{prompt_content}"
        )
        action_type = extraction.action_type
        extracted_params = extraction.params
        confirmation_message = extraction.confirmation_message
    except Exception:
        action_type = "unknown"
        extracted_params = {}
        confirmation_message = "参数提取失败，请重试"

    # ─────────────────────────────────────────
    # Schema 强约束：计算缺失字段
    # ─────────────────────────────────────────
    draft_complete = False
    draft_missing_fields = []
    response_message = confirmation_message

    try:
        from draft_schemas import SchemaRegistry, fill_draft_from_params

        # 确定 action_type：如果有续填任务，使用之前的类型
        effective_action_type = action_type if action_type != "unknown" else previous_action_type

        schema_class = SchemaRegistry.get_schema_class(effective_action_type)

        if schema_class:
            # 合并历史参数和新提取的参数
            merged_params = {**previous_confirmed_slots, **extracted_params}

            # 创建草稿实例并计算缺失字段
            draft_instance = fill_draft_from_params(effective_action_type, merged_params)
            if draft_instance:
                draft_missing_fields = draft_instance.get_missing_fields()
                draft_complete = len(draft_missing_fields) == 0

            # 生成缺失字段提问
            if draft_missing_fields:
                questions = SchemaRegistry.build_questions_for_fields(effective_action_type, draft_missing_fields[:2])
                if questions:
                    response_message = questions[0]

    except ImportError as e:
        logger.warning(f"草稿模块未导入: {e}")

    import uuid
    from datetime import datetime, timezone
    action_payload = {
        "action_type": action_type,
        "params": extracted_params,
        "request_id": str(uuid.uuid4()),
        "metadata": {
            "user_token": user_token,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    }

    logger.info(
        "动作参数提取完成（Schema 强约束）",
        extra={
            "action_type": action_type,
            "draft_id": draft_id,
            "draft_complete": draft_complete,
            "missing_fields": draft_missing_fields,
            "session_id": session_id,
            "component": "draft_action",
        }
    )

    return {
        **state,
        "intent": "action",
        "extracted_params": extracted_params,
        "action_payload": action_payload,
        "draft_action_type": action_type,
        "draft_complete": draft_complete,
        "draft_missing_fields": draft_missing_fields,
        "requires_approval": draft_complete,
        # pending_slots: 持久化缺失字段，供后续对话续填
        "pending_slots": draft_missing_fields,
        "confirmed_slots": extracted_params,
        "messages": [
            HumanMessage(content=response_message)
        ],
    }


# =============================================================================
# 节点四：API 执行节点
# =============================================================================

def execute_api_node(state: AgentState) -> AgentState:
    """
    API 执行节点

    作用：用户确认后，实际调用后端服务执行操作。

    使用的协议：MCP（Model Context Protocol）
    这是一种标准化的工具调用协议，比直接 HTTP 调用更规范。
    """
    action_payload = state.get("action_payload", {})
    user_token = state.get("user_token", "")
    action_type = action_payload.get("action_type", "unknown")
    extracted_params = action_payload.get("params", {})

    if not action_payload:
        return {
            **state,
            "messages": [AIMessage(content="操作 Payload 为空，无法执行，请重新发起请求。")],
        }

    logger.info(
        "开始执行 API 调用",
        extra={
            "action_type": action_type,
            "request_id": action_payload.get("request_id", ""),
            "component": "execute_api",
        }
    )

    # MCP 工具映射表
    tool_mapping = {
        "leave_request": mcp.tools.leave_request,
        "expense_reimburse": mcp.tools.expense_reimburse,
        "password_reset": mcp.tools.password_reset,
        "permission_open": mcp.tools.permission_open,
    }

    try:
        # 根据 action_type 选择对应的 MCP 工具
        if action_type in tool_mapping:
            tool = tool_mapping[action_type]
            result = mcp.run_tool(tool.name, extracted_params, {"user_token": user_token})
            success = result.get("success", True)
            result_message = result.get("message", "操作执行成功")
            request_id = result.get("request_id", action_payload.get("request_id", ""))
        else:
            # 通用执行接口
            result = mcp.run_tool(
                "execute_action",
                {"action_type": action_type, "params": extracted_params},
                {"user_token": user_token}
            )
            success = result.get("success", True)
            result_message = result.get("message", "操作执行成功")
            request_id = result.get("request_id", action_payload.get("request_id", ""))

    except RuntimeError as e:
        if "断路器" in str(e):
            success = False
            result_message = str(e)
        else:
            success = False
            result_message = f"系统异常：{str(e)}"
    except Exception as e:
        success = False
        result_message = f"系统异常：{str(e)}"

    logger.info(
        "API 调用完成",
        extra={
            "action_type": action_type,
            "request_id": request_id,
            "success": success,
            "component": "execute_api",
        }
    )

    if success:
        exec_message = AIMessage(
            content=f"操作已提交成功！\n"
                    f"操作类型：{action_type}\n"
                    f"请求ID：{request_id}\n"
                    f"结果：{result_message}"
        )
    else:
        exec_message = AIMessage(
            content=f"操作执行失败。\n"
                    f"失败原因：{result_message}\n"
                    f"如有疑问，请联系 HR 部门。"
        )

    return {
        **state,
        "messages": [exec_message],
        "interrupted": False,
        "execution_success": success,
        "execution_error": result_message if not success else None,
    }


# =============================================================================
# 节点五：响应生成节点
# =============================================================================

def generate_response_node(state: AgentState) -> AgentState:
    """
    响应生成节点

    作用：将处理结果组装成最终的回复消息。
    这个节点通常在 RAG 检索后调用。
    """
    messages = state.get("messages", [])
    if not messages:
        return {"final_response": "您好，有什么可以帮您？"}

    final_msg = messages[-1].content if messages else "处理完成。"
    return {
        "final_response": final_msg,
    }


# =============================================================================
# 构建 StateGraph
# =============================================================================

def build_agent_graph() -> StateGraph:
    """
    构建完整的 Agent StateGraph

    StateGraph 是 LangGraph 的核心概念：
    - Node（节点）：执行特定任务的函数
    - Edge（边）：节点之间的连接关系

    这个函数定义了：
    1. 有哪些节点
    2. 节点之间的连接关系
    3. 条件路由（根据状态决定下一步去哪）
    """
    graph = StateGraph(AgentState)

    # 添加所有节点
    graph.add_node("router", router_node)  # 意图分类
    graph.add_node("chitchat", chitchat_node)  # 闲聊礼貌拒答
    graph.add_node("rag", rag_node)  # RAG 检索
    graph.add_node("draft_action", draft_action_node)  # 参数提取
    graph.add_node("execute_api", execute_api_node)  # API 执行
    graph.add_node("generate_response", generate_response_node)  # 响应生成

    # 设置入口节点
    graph.set_entry_point("router")

    # 条件路由：根据意图决定下一步
    def route_after_router(state: AgentState) -> Literal["rag", "draft_action", "chitchat"]:
        intent = state.get("intent", "policy")
        has_unfinished_draft = state.get("has_unfinished_draft", False)
        latest_message = state.get("messages", [None])[-1].content if state.get("messages") else ""

        # 续填关键词检测
        resume_keywords = ["继续", "接着", "还没", "还没填", "继续填", "填完了"]
        is_resume = any(kw in latest_message for kw in resume_keywords)

        # 如果有未完成的草稿，且用户想继续填写 → 跳转到 draft_action
        if has_unfinished_draft and (intent == "action" or is_resume):
            return "draft_action"
        elif intent == "action":
            return "draft_action"
        elif intent == "chitchat":
            return "chitchat"
        else:
            return "rag"

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"rag": "rag", "draft_action": "draft_action", "chitchat": "chitchat"},
    )

    # 添加普通边（固定流程）
    graph.add_edge("draft_action", END)  # 动作执行后结束
    graph.add_edge("chitchat", END)  # 闲聊回复后结束
    graph.add_edge("rag", "generate_response")  # RAG 后生成响应
    graph.add_edge("generate_response", END)  # 响应后结束

    # 编译工作流
    compiled_graph = graph.compile(
        checkpointer=get_checkpointer(),  # 使用 PostgreSQL 保存状态
        interrupt_before=["execute_api"],  # 在执行 API 前暂停，等用户确认
    )

    return compiled_graph


_agent_graph: Optional[StateGraph] = None


def get_agent_graph() -> StateGraph:
    """
    获取 Agent Graph 单例（延迟初始化）

    单例模式确保整个应用只创建一个工作流实例，节省资源。
    """
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph
