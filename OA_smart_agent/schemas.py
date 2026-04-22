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
# 意图分类 Structured Output 模型
# =============================================================================

class IntentClassification(BaseModel):
    """意图分类结果 - 用于 Router 节点的结构化输出"""
    intent: Literal["policy", "action", "chitchat"] = Field(
        description="用户意图类型。policy=政策咨询（年假、报销流程、考勤规定等）, action=业务办理（请假、报销、密码重置等）, chitchat=闲聊"
    )
    reasoning: str = Field(default="", description="简短推理过程")


class ActionParamExtraction(BaseModel):
    """动作参数提取结果 - 用于 Draft Action 节点的结构化输出"""
    action_type: Literal["leave_request", "expense_reimburse", "password_reset", "permission_open", "其他"] = Field(
        description="动作类型。leave_request=请假, expense_reimburse=报销, password_reset=密码重置, permission_open=权限开通, 其他=无法识别"
    )
    params: dict = Field(default_factory=dict, description="提取的参数，键值对形式，如 {start_date: \"2026-04-10\", end_date: \"2026-04-12\", reason: \"旅游\"}")
    confirmation_message: str = Field(default="", description="确认话术，如'您正在申请 2026-04-10 至 2026-04-12 的请假，共3天，原因是旅游'。参数不完整时说明缺失信息")


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
    LangGraph 工作流的状态定义。

    这个类定义了 AI 处理问题时的"思考过程记录"，
    就像一个会诊记录本，每个科室的医生都会在上面添加信息。

    状态流转说明：
    1. messages 记录完整对话历史，每经过一个 Node 追加新消息
    2. intent 由 Router_Node 写入，后继节点读取以决定路由
    3. extracted_params 和 action_payload 由 Draft_Action_Node 生成
    4. requires_approval 标记图是否需要中断等待人工审批
    5. user_token 从 Request 中传入，在 Execute_API_Node 中使用
    """

    # ─────────────────────────────────────────
    # 核心字段（Phase 1）- 最基本的信息
    # ─────────────────────────────────────────
    
    # 对话历史列表，add_messages 策略确保新消息追加而非覆盖
    # 就像一个聊天记录本，每轮对话都加进去
    messages: Annotated[list, add_messages]

    # 用户意图：policy（政策咨询）或 action（动作执行）或 chitchat（闲聊）
    # AI 分析用户问题后判断：是问问题、想办事、还是闲聊
    intent: Optional[str]

    # 从对话中提取的动作参数（如请假时间、申请人等）
    # 当用户说"我想请假3天"，这里就记录 {天数: 3}
    extracted_params: Optional[dict]

    # 是否需要人工审批（Draft_Action_Node 设为 True 后触发中断）
    # 就像打卡机：需要审批时机器会"暂停"
    requires_approval: bool

    # 标准化 Payload，携带操作类型和参数，将发给 Java 后端
    # 格式统一，方便后端处理
    action_payload: Optional[dict]

    # 用户鉴权 Token（从 API 层传入，在 Execute_API_Node 中使用）
    # 用于调用 Java 后端 API 时验证身份
    user_token: Optional[str]

    # 图是否处于中断状态（用于 API 层判断是否需要返回审批卡片）
    interrupted: bool

    # RAG 检索结果（用于 RAG_Node 输出，供话术生成使用）
    # AI 找到的参考资料
    retrieved_docs: Optional[List[dict]]

    # 最终输出话术（由 Generate_Response_Node 写入）
    final_response: Optional[str]

    # ─────────────────────────────────────────
    # Re-Ranking 字段（Phase 2.1）- 精排优化
    # ─────────────────────────────────────────
    
    # Cross-Encoder 重排后的文档列表（更准确的排序）
    reranked_docs: Optional[List[dict]]

    # ─────────────────────────────────────────
    # 可控 Self-RAG 字段（Phase 2 企业版）- 检索增强
    # ─────────────────────────────────────────
    
    # 检索路由决策：must_retrieve / skip_retrieve / maybe_retrieve
    retrieval_decision: Optional[str]
    
    # 检索结果质量评分（BGE-Rerank 平均分数）
    # 用于监控和日志分析，暂未在业务逻辑中使用
    retrieval_quality_score: Optional[float]
    
    # 被过滤的低分文档数
    # 可用于统计检索效率，暂未在业务逻辑中使用
    retrieval_filtered_count: Optional[int]
    
    # 有用性判断结果
    # 由 RAG 节点设置，用于判断检索结果是否可用
    is_useful: Optional[bool]
    
    # 是否跳过检索（闲聊/问候等不需要检索的情况）
    skipped_retrieval: bool

    # ─────────────────────────────────────────
    # 升级处理字段（用于需要人工介入的场景）
    # ─────────────────────────────────────────
    
    # 自我反思结果列表（包含 IS_RELATED / IS_SUPPORTED / IS_USEFUL 评估）
    reflection_results: Optional[List[dict]]

    # 是否需要升级人工处理
    needs_escalation: bool

    # 压缩后的对话上下文（用于长对话的 Token 节省）
    compressed_context: Optional[str]

    # 待填充的参数槽位列表（Slot Filling）
    pending_slots: Optional[List[str]]

    # 已确认的参数槽位字典（Slot Filling）
    confirmed_slots: Optional[dict]

    # ─────────────────────────────────────────
    # 多模态处理字段（Phase 5）
    # ─────────────────────────────────────────
    
    # 用户上传的图片数据（Base64 编码）
    image_data: Optional[str]

    # 多模态处理结果（图片分析、OCR 识别等）
    multimodal_context: Optional[dict]

    # ─────────────────────────────────────────
    # 草稿箱字段（Schema 强约束 + 任务挂起）
    # ─────────────────────────────────────────

    # 当前业务操作类型（如 leave_request, expense_reimburse）
    # 用于识别用户正在办理什么业务
    draft_action_type: Optional[str]

    # 草稿 ID（用于关联 Redis 中的草稿数据）
    draft_id: Optional[str]

    # 当前草稿是否完整（所有必填字段是否已填）
    draft_complete: bool

    # 缺失的必填字段列表（用于 LLM 生成提问）
    draft_missing_fields: Optional[List[str]]

    # 任务挂起栈是否为空（用于判断是否有被中断的任务）
    has_suspended_tasks: bool

    # 待恢复任务的摘要信息
    suspended_task_summary: Optional[dict]
