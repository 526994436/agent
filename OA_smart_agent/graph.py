"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     OA 智能 Agent 核心工作流引擎 (ReAct 模式)                  ║
║                              graph.py                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

【架构说明】
采用 ReAct (Reasoning + Acting) 模式，让大模型自主决定：
1. 是否需要调用工具
2. 调用哪个工具
3. 如何处理工具返回结果

【流程图】
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌──────────┐     ┌──────────────────────────────────────────────────┐     │
│  │          │     │                    MCP Tools                       │     │
│  │    LLM   │────▶│  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │     │
│  │(+Tools)  │     │  │ rag_retrieve│  │leave_request│  │ expense │ │     │
│  └──────────┘     │  │ (知识库)    │  │  (请假)     │  │(报销)   │ │     │
│       ▲           │  └─────────────┘  └─────────────┘  └─────────┘ │     │
│       │           └──────────────────────────────────────────────────┘     │
│       │                                   │                                │
│       │         (工具调用前自动中断)        ▼                                │
│       │                          ┌───────────────┐                        │
│       └──────────────────────────│  Human Loop   │                        │
│                                  │  (审批/拒绝)  │                        │
│                                  └───────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘

【核心优势】
1. 意图识别：大模型自己决定是否调用工具
2. 参数提取：大模型原生理解工具 Schema
3. 工具解耦：新增工具无需改代码
4. Human-in-the-loop：interrupt_before 框架层实现
"""

import logging
import time
from functools import wraps
from typing import Literal, Optional, Callable, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from config import settings
from schemas import AgentState

logger = logging.getLogger("oa_agent.graph")


# =============================================================================
# PostgreSQL Checkpointer 配置
# =============================================================================

def _get_postgres_url() -> str:
    """获取 PostgreSQL 数据库连接地址"""
    url = settings.postgres_checkpointer_url or settings.history_db_url
    if not url:
        raise ValueError(
            "请配置 PostgreSQL 连接地址："
            "设置 postgres_checkpointer_url 或 history_db_url"
        )
    return url


def _build_postgres_saver() -> PostgresSaver:
    """构建 PostgreSQL Checkpointer"""
    url = _get_postgres_url()
    return PostgresSaver.from_conn_string(url)


def _postgres_safe_call(func: Callable) -> Callable:
    """PostgreSQL 操作装饰器"""
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
    """PostgreSQL Checkpointer 封装类"""

    def __init__(self):
        self._saver: Optional[PostgresSaver] = None

    def initialize(self) -> PostgresSaver:
        """延迟初始化数据库连接"""
        if self._saver is None:
            url = _get_postgres_url()
            self._saver = PostgresSaver.from_conn_string(url)
            self._saver.setup()
        return self._saver

    def close(self):
        """关闭连接池"""
        if self._saver:
            self._saver = None


_postgres_checkpointer: Optional[PostgresCheckpointer] = None


def get_checkpointer() -> PostgresCheckpointer:
    """获取 PostgreSQL Checkpointer 单例"""
    global _postgres_checkpointer
    if _postgres_checkpointer is None:
        _postgres_checkpointer = PostgresCheckpointer()
    return _postgres_checkpointer


# =============================================================================
# LLM 配置
# =============================================================================

def _create_llm(temperature: float = 0.3) -> ChatOpenAI:
    """
    创建 LLM 实例（带指标记录）
    
    支持从 state 中获取 user_token 并注入到请求头
    """
    return ChatOpenAI(
        model=settings.vllm_model,
        api_key=settings.vllm_api_key,
        base_url=settings.vllm_base_url,
        temperature=temperature,
        timeout=settings.vllm_request_timeout,
        max_retries=1,
        max_tokens=settings.vllm_max_tokens,
    )


# =============================================================================
# 工具加载（MCP 集成）
# =============================================================================

def _load_mcp_tools() -> Sequence[BaseTool]:
    """
    从 MCP Server 加载工具列表
    
    返回的每个工具都会被 ToolNode 包装，
    大模型可以自主决定是否调用、调用哪个工具。
    """
    try:
        from mcp.server import mcp
        tools = mcp.tools
        logger.info(f"已加载 {len(tools)} 个 MCP 工具")
        return tools
    except Exception as e:
        logger.warning(f"MCP 工具加载失败: {e}，将使用空工具列表")
        return []


# =============================================================================
# 节点一：LLM 节点（带工具绑定）
# =============================================================================

def create_llm_node() -> Callable:
    """
    创建 LLM 节点工厂函数
    
    这个节点是 ReAct 模式的核心：
    - 大模型接收对话历史
    - 基于用户输入，决定是否调用工具
    - 如果需要调用，将生成 tool_calls
    - 如果不需要，生成普通文本回复
    
    Returns:
        可调用节点函数
    """
    # 加载 MCP 工具
    mcp_tools = _load_mcp_tools()
    
    # 创建绑定了工具的 LLM
    llm = _create_llm(temperature=0.3)
    if mcp_tools:
        llm_with_tools = llm.bind_tools(mcp_tools)
    else:
        llm_with_tools = llm

    def llm_node(state: AgentState) -> AgentState:
        """
        LLM 节点：处理对话，决定是否调用工具
        
        Args:
            state: AgentState，包含 messages, user_token 等
            
        Returns:
            更新后的 state，包含 messages（追加 AI 回复）
        """
        from metrics import record_llm_call, increment_llm_in_flight
        
        # 链路追踪
        span = None
        try:
            from observability import get_trace_collector
            collector = get_trace_collector()
            span = collector.start_span(
                span_name="llm_node",
                metadata={"node": "llm", "tools_count": len(mcp_tools)}
            )
        except ImportError:
            pass

        messages = state.get("messages", [])
        if not messages:
            return {"messages": [AIMessage(content="您好，有什么可以帮您？")]}

        # 记录指标
        target_model = settings.vllm_model
        increment_llm_in_flight(target_model, 1)
        start_time = time.perf_counter()
        success = True

        try:
            # 调用 LLM（大模型自己决定是否调用工具）
            response = llm_with_tools.invoke(messages)
            
            logger.info(
                "LLM 调用完成",
                extra={
                    "has_tool_calls": bool(response.tool_calls),
                    "tool_count": len(response.tool_calls) if response.tool_calls else 0,
                    "content_length": len(response.content) if response.content else 0,
                    "session_id": state.get("session_id", ""),
                    "component": "llm",
                }
            )

            return {
                "messages": [response],
            }
        except Exception as e:
            success = False
            logger.error(
                "LLM 调用失败",
                extra={"error": str(e), "component": "llm"}
            )
            return {
                "messages": [AIMessage(content=f"抱歉，系统处理时出现错误：{str(e)}")],
            }
        finally:
            duration = time.perf_counter() - start_time
            increment_llm_in_flight(target_model, -1)
            record_llm_call(model=target_model, node="llm", duration=duration, success=success)
            
            if span:
                try:
                    collector.end_span(span)
                except Exception:
                    pass

    return llm_node


# =============================================================================
# 节点二：工具节点（LangGraph 原生 ToolNode）
# =============================================================================

def create_tool_node() -> ToolNode:
    """
    创建工具节点
    
    ToolNode 会自动：
    1. 检查 LLM 返回的 tool_calls
    2. 调用对应的 MCP 工具
    3. 将工具结果作为 ToolMessage 添加到 messages
    
    当大模型准备好调用工具时，会在这里被 LangGraph 中断（interrupt），
    等待前端用户点击"同意"后才会真正执行。
    """
    mcp_tools = _load_mcp_tools()
    
    if not mcp_tools:
        logger.warning("没有加载到 MCP 工具，ToolNode 将不执行任何操作")
        # 返回一个空工具节点
        def empty_node(state: AgentState) -> AgentState:
            return {}
        return empty_node

    tool_node = ToolNode(mcp_tools)
    
    logger.info(f"ToolNode 已创建，包含 {len(mcp_tools)} 个工具")
    return tool_node


# =============================================================================
# 辅助函数：路由决策
# =============================================================================

# 不需要审批的工具类型（查询类）
TOOLS_NO_APPROVAL = {
    "rag_retrieve",      # 知识库检索
    "search_knowledge",   # 知识搜索（备用）
}

# 需要审批的工具类型（动作类）
TOOLS_NEED_APPROVAL = {
    "leave_request",          # 请假申请
    "expense_reimburse",      # 费用报销
    "password_reset",          # 密码重置
    "permission_open",         # 权限开通
    "execute_action",          # 通用动作执行
}


def should_continue(state: AgentState) -> Literal["llm", "tools", "tools_auto", END]:
    """
    判断下一步应该去哪里

    核心逻辑：
    1. 如果最后一条消息包含 tool_calls → 判断工具类型
       - 查询类工具（rag_retrieve）→ 直接执行
       - 动作类工具（leave_request 等）→ 中断等待审批
    2. 如果没有 tool_calls 且有普通回复 → 结束
    3. 如果是 ToolMessage（工具执行结果）→ 回到 LLM

    LangGraph 会自动处理 ToolMessage，
    工具执行完后会回到 LLM 节点继续对话。
    """
    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]

    # 检查是否有 tool_calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_name = last_message.tool_calls[0].get("name", "")

        # 判断工具是否需要审批
        if tool_name in TOOLS_NO_APPROVAL:
            # 查询类工具：直接执行，不中断
            return "tools_auto"
        else:
            # 动作类工具：返回 tools，触发 interrupt_before 中断
            return "tools"

    # 检查是否是 ToolMessage（工具执行结果）
    if hasattr(last_message, "type") and last_message.type == "tool":
        return "llm"

    # 普通回复，结束
    return END


# =============================================================================
# 构建 StateGraph
# =============================================================================

def build_agent_graph() -> StateGraph:
    """
    构建完整的 Agent StateGraph（ReAct 模式 + 智能审批）

    架构：
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  ┌──────────┐     ┌───────────────────────────────────────────────┐     │
    │  │          │     │              MCP Tools                          │     │
    │  │   LLM   │────▶│  • rag_retrieve (知识库)                       │     │
    │  │(+Tools) │     │  • leave_request (请假)  ←── 需要审批          │     │
    │  └──────────┘     │  • expense_reimburse (报销) ← 需要审批         │     │
    │       ▲           │  • password_reset (密码)                        │     │
    │       │           │  • permission_open (权限)                       │     │
    │       │           └───────────────────────────────────────────────┘     │
    │       │                            │                                   │
    │       │           ┌────────────────┼────────────────┐                   │
    │       │           │                │                │                   │
    │       │           ▼                ▼                ▼                   │
    │       │    ┌────────────┐  ┌────────────┐  ┌────────────┐            │
    │       │    │ tools_auto │  │  Human     │  │   END      │            │
    │       │    │(无审批)    │  │  Loop     │  │            │            │
    │       │    └────────────┘  │(需审批)    │  └────────────┘            │
    │       │           │         │            │                             │
    │       │           │         │(审批/拒绝) │                             │
    │       │           │         └─────┬──────┘                             │
    │       │           │               │                                    │
    └───────┼───────────┼───────────────┼────────────────────────────────────┘
            │           │               │
            │    (直接执行)       (中断等待审批)

    审批规则：
    - 查询类工具（rag_retrieve）：无需审批，直接执行
    - 动作类工具（leave_request, expense_reimburse 等）：需要审批
    """
    graph = StateGraph(AgentState)

    # 创建节点
    llm_node = create_llm_node()
    tool_node = create_tool_node()

    # 注册节点
    graph.add_node("llm", llm_node)          # LLM 对话节点（核心）
    graph.add_node("tools", tool_node)         # 工具节点（需审批）
    graph.add_node("tools_auto", tool_node)   # 工具节点（无审批）

    # 设置入口节点
    graph.set_entry_point("llm")

    # 条件路由
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "llm": "llm",           # 工具结果回来，继续对话
            "tools": "tools",         # 动作类工具：需要审批
            "tools_auto": "tools_auto",  # 查询类工具：直接执行
            END: END,                 # 结束
        }
    )

    # 工具执行完后回到 LLM
    graph.add_edge("tools", "llm")
    graph.add_edge("tools_auto", "llm")

    # 编译工作流
    # 注意：只有 "tools" 节点会触发中断，"tools_auto" 不会
    compiled_graph = graph.compile(
        checkpointer=get_checkpointer().initialize(),
        interrupt_before=["tools"],  # 只有动作类工具调用前中断
    )

    logger.info("Agent Graph 构建完成（ReAct 模式 + 智能审批）")
    return compiled_graph


# =============================================================================
# 单例模式
# =============================================================================

_agent_graph: Optional[StateGraph] = None


def get_agent_graph() -> StateGraph:
    """
    获取 Agent Graph 单例（延迟初始化）
    """
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


# =============================================================================
# 工具列表导出（供外部使用）
# =============================================================================

def get_available_tools() -> list:
    """获取所有可用的 MCP 工具列表"""
    tools = _load_mcp_tools()
    return [
        {
            "name": t.name,
            "description": t.description,
            "args": t.args_schema.model_json_schema() if hasattr(t, "args_schema") else {},
        }
        for t in tools
    ]
