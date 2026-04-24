"""
数据模型定义模块 (schemas.py)

这个文件定义了系统中的所有"数据结构"，就像一个数据库的表结构。

主要包含两大类：
1. Pydantic 模型 - 用于 API 请求/响应的数据验证
2. AgentState (TypedDict) - 用于 LangGraph 内部状态管理

为什么需要这个文件？
- 确保前端传给后端的数据格式正确
- 自动转换数据类型（如 JSON -> Python 对象）
- 提供清晰的字段说明和默认值
"""

# ========== 引入基础类型和工具 ==========
from typing import Optional, Literal, List, Any, TypedDict, Annotated  # typing 工具箱
from pydantic import BaseModel, Field  # Pydantic 数据验证
from langgraph.graph import add_messages  # LangGraph 消息追加策略



# =============================================================================
# 第一部分：FastAPI 接口层的 Pydantic 模型
# =============================================================================
# 这一部分的模型用于前端和后端之间的数据交换


class FileUploadData(BaseModel):
    """
    文件上传数据 - 用户上传的文件内容
    
    支持的文件格式：PDF、Word (DOCX/DOC)、Excel (XLSX/XLS/CSV)、PowerPoint (PPTX)、纯文本 (TXT)
    """
    file_name: str = Field(..., description="文件名，包含扩展名，如 document.pdf")
    file_content: str = Field(..., description="Base64 编码的文件内容")
    file_type: Optional[str] = Field(None, description="文件 MIME 类型，如 application/pdf")


class ChatRequest(BaseModel):
    """
    聊天请求 - 用户发给 AI 的消息
    
    就像填写一张表单，每个字段都有要求：
    """
    session_id: str = Field(..., description="会话唯一标识，UUID 格式")  # 会话 ID，必填
    user_token: str = Field(..., description="用户鉴权 Token，从请求头或 Body 传入")  # 身份令牌，必填
    query: str = Field(..., description="用户输入的查询内容", min_length=1, max_length=2000)  # 问题内容，必填，限制长度
    image_data: Optional[str] = Field(None, description="Base64 编码的图片，用于多模态处理")  # 图片，可选
    file_data: Optional[FileUploadData] = Field(None, description="上传的文件内容，支持 PDF、Word、Excel、PPTX、TXT")  # 文件上传，可选


class FileUploadData(BaseModel):
    """
    文件上传数据 - 用户上传的文件内容
    
    支持的文件格式：PDF、Word (DOCX)、Excel (XLSX/XLS)、PowerPoint (PPTX)、纯文本 (TXT)
    """
    file_name: str = Field(..., description="文件名，包含扩展名，如 document.pdf")
    file_content: str = Field(..., description="Base64 编码的文件内容")
    file_type: Optional[str] = Field(None, description="文件 MIME 类型，如 application/pdf")


class DraftAction(BaseModel):
    """
    待审批动作 - 当 AI 要执行操作时（如请假），展示给用户确认的结构
    
    就像一张"申请表"，包含：
    """
    action_type: str = Field(..., description="动作类型，如 leave_request（请假）, reset_password（重置密码）")  # 操作类型
    extracted_params: dict = Field(default_factory=dict, description="从对话中提取的参数，如 {时间: '2024-01-01', 原因: '出差'}")  # 提取的参数
    payload: dict = Field(..., description="将发送给 Java 后端的标准化 Payload")  # 完整数据
    confirmation_message: str = Field(..., description="展示给用户的确认话术，如'您确定要请假 3 天吗？'")  # 确认消息


class ChatResponse(BaseModel):
    """
    聊天响应 - AI 返回给用户的消息结构
    
    有两种情况：
    1. completed - 直接给出答案
    2. awaiting_approval - 需要用户确认后才能执行
    """
    session_id: str  # 会话 ID
    status: Literal["completed", "awaiting_approval"] = "completed"  # 状态：完成 或 等待审批
    message: Optional[str] = Field(None, description="最终回复话术或中断提示")  # AI 说的话
    requires_approval: bool = Field(False, description="是否需要用户审批")  # 是否需要审批
    draft_action: Optional[DraftAction] = Field(None, description="待审批动作详情，只有 requires_approval=True 时才有值")  # 审批卡片


class ApproveRequest(BaseModel):
    """
    审批请求 - 用户确认或拒绝操作
    
    就像审批流程中的"同意"或"驳回"按钮：
    """
    session_id: str = Field(..., description="会话唯一标识")  # 要审批哪个会话
    user_token: str = Field(..., description="用户鉴权 Token")  # 身份验证
    action: Literal["approve", "reject"] = Field(..., description="审批动作：approve=同意执行, reject=取消操作")  # 操作：同意或拒绝


class ApproveResponse(BaseModel):
    """
    审批响应 - 审批结果返回给前端
    
    告诉前端审批是否成功，以及结果消息：
    """
    session_id: str  # 会话 ID
    status: Literal["success", "rejected", "error"] = "success"  # 结果：成功/被拒绝/错误
    message: str = Field(..., description="执行结果描述或拒绝原因")  # 结果消息


# =============================================================================
# 第二部分：LangGraph 内部 State 定义（TypedDict）
# =============================================================================
# 这一部分是 AI 内部的"工作记忆"，记录每个节点处理过程中的数据

class AgentState(TypedDict, total=False):
    """
    LangGraph 工作流的状态定义（ReAct 模式）。

    重构说明：
    - 移除了 intent、extracted_params、draft_action_type 等草稿相关字段
    - 移除了 pending_slots、confirmed_slots 等 Slot Filling 字段
    - 移除了 draft_complete、draft_missing_fields 等草稿状态字段
    - 保留 messages 作为核心状态
    - 新增 session_id、user_id 用于追踪和日志

    状态流转：
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │   LLM    │────▶│  Tools   │────▶│   LLM    │ ← 循环
    │(+Tools)  │     │ (MCP)    │     │(ToolMsg) │
    └──────────┘     └──────────┘     └──────────┘
         ▲                                  │
         │                                  │
         └──────────────────────────────────┘
                      (tool_calls)

    Human-in-the-loop：
    - interrupt_before=["tools"] 确保工具调用前暂停
    - 前端可以通过 graph.invoke(NULL, resume=True) 继续执行
    """

    # ─────────────────────────────────────────
    # 核心字段
    # ─────────────────────────────────────────

    # 对话历史列表，add_messages 策略确保新消息追加而非覆盖
    # 包含 HumanMessage、AIMessage、ToolMessage
    messages: Annotated[list, add_messages]

    # 用户鉴权 Token（从 API 层传入，用于 MCP 工具调用）
    user_token: Optional[str]

    # 会话 ID（用于追踪和日志）
    session_id: Optional[str]

    # 用户 ID（用于 ABAC 权限过滤）
    user_id: Optional[str]

    # 用户部门（用于 ABAC 权限过滤）
    user_department: Optional[str]

    # 用户项目组（用于 ABAC 权限过滤）
    user_projects: Optional[list]

    # ─────────────────────────────────────────
    # 多模态处理字段（可选）
    # ─────────────────────────────────────────

    # 用户上传的图片数据（Base64 编码）
    image_data: Optional[str]

    # 多模态处理结果（图片分析、OCR 识别等）
    multimodal_context: Optional[dict]

    # ─────────────────────────────────────────
    # RAG 检索字段（保留，供 MCP 工具使用）
    # ─────────────────────────────────────────

    # RAG 检索结果
    retrieved_docs: Optional[List[dict]]

    # 检索路由决策：must_retrieve / skip_retrieve / maybe_retrieve
    retrieval_decision: Optional[str]

    # 检索结果质量评分
    retrieval_quality_score: Optional[float]

    # ─────────────────────────────────────────
    # 状态标志
    # ─────────────────────────────────────────

    # 图是否处于中断状态（用于 API 层判断是否需要返回审批卡片）
    interrupted: bool

    # 是否需要升级人工处理
    needs_escalation: bool

    # 工具调用前的待审批信息（由 LangGraph 自动设置）
    pending_tool_calls: Optional[List[dict]]
