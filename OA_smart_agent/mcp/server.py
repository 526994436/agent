"""
MCP Server - 合同审计工具

使用 FastMCP 实现标准化工具调用。

提供的工具（Tools）：
- contract_parse: 合同结构解析（LLM 提取关键字段）
- contract_audit: 合同审计（RAG 查规范 + tool 规则校验 + 输出报告）
- rag_retrieve: 企业知识库检索
"""

import asyncio
import json
from typing import Any, Optional

from fastmcp import FastMCP

from ..utils.logging import logger


# ─────────────────────────────────────────────
# MCP Server 实例
# ─────────────────────────────────────────────
mcp = FastMCP(
    name="OAEnterpriseTools",
    description="企业 OA 系统智能工具集（合同审计 + 知识库检索）",
    dependencies=["httpx"],
)


# ─────────────────────────────────────────────
# MCP Tools 定义
# ─────────────────────────────────────────────

# ── 工具一：合同结构解析 ──────────────────────────────────────────────
@mcp.tool()
async def contract_parse(
    contract_text: str,
) -> dict:
    """
    解析合同文本结构，提取关键字段和条款。

    使用场景：
    - 用户上传合同后，首先调用此工具解析合同结构
    - 合同审计流程的第一步

    参数:
        contract_text: 合同原始文本内容（从 PDF/Word 提取的文字）

    返回:
        包含合同结构化解析结果
    """
    from pydantic import BaseModel
    from typing import List as TList

    try:
        from prompts import contract_struct_parse
        from graph import llm_with_timeout

        messages = contract_struct_parse(contract_text=contract_text)

        class ContractStructOutput(BaseModel):
            contract_name: str
            contract_type: str
            contract_no: Optional[str] = None
            signing_date: Optional[str] = None
            effective_date: Optional[str] = None
            expiry_date: Optional[str] = None
            party_a: Optional[str] = None
            party_b: Optional[str] = None
            total_amount: Optional[str] = None
            payment_terms: Optional[str] = None
            key_obligations_a: TList[str] = []
            key_obligations_b: TList[str] = []
            penalty_clauses: Optional[str] = None
            termination_clauses: Optional[str] = None
            dispute_resolution: Optional[str] = None
            governing_law: Optional[str] = None
            special_clauses: Optional[str] = None
            main_clauses: TList[dict] = []

        result = await asyncio.to_thread(
            llm_with_timeout.with_structured_output,
            messages,
            ContractStructOutput,
        )

        struct_dict = result.model_dump()
        return {
            "success": True,
            "contract_name": struct_dict.get("contract_name", ""),
            "contract_type": struct_dict.get("contract_type", ""),
            "structure": struct_dict,
            "message": "合同结构解析完成",
        }

    except Exception as e:
        logger.error(f"合同结构解析失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "合同结构解析失败，请检查合同文本格式",
        }


# ── 工具二：合同审计（完整流程）──────────────────────────────────────
@mcp.tool()
async def contract_audit(
    contract_text: str,
    user_token: str = "",
) -> dict:
    """
    对合同进行完整审计：解析结构 + RAG 查公司规范 + 风险校验 + 生成报告。

    审计流程：
    1. LLM 解析合同结构
    2. RAG 检索公司合同管理制度和规范
    3. Tool 做规则校验（对照法律法规 + 公司规范）
    4. 输出审计报告（风险点 + 修改建议 + 是否通过）

    使用场景：
    - 用户上传合同后，对合同进行全面审计
    - 自动识别风险点并生成修改建议

    参数:
        contract_text: 合同原始文本内容
        user_token: 用户认证令牌（用于 RAG 权限过滤）

    返回:
        包含完整审计报告（风险点 + 修改建议 + 是否通过）
    """
    from pydantic import BaseModel
    from typing import List as TList

    try:
        from prompts import (
            contract_struct_parse,
            contract_risk_check,
            format_docs,
        )
        from controlled_self_rag import get_controlled_self_rag
        from auth import decode_jwt_token
        from graph import llm_with_timeout

        # ── Step 1: LLM 解析合同结构 ─────────────────────────────────
        struct_messages = contract_struct_parse(contract_text=contract_text)

        class ContractStructOutput(BaseModel):
            contract_name: str
            contract_type: str
            contract_no: Optional[str] = None
            signing_date: Optional[str] = None
            effective_date: Optional[str] = None
            expiry_date: Optional[str] = None
            party_a: Optional[str] = None
            party_b: Optional[str] = None
            total_amount: Optional[str] = None
            payment_terms: Optional[str] = None
            key_obligations_a: TList[str] = []
            key_obligations_b: TList[str] = []
            penalty_clauses: Optional[str] = None
            termination_clauses: Optional[str] = None
            dispute_resolution: Optional[str] = None
            governing_law: Optional[str] = None
            special_clauses: Optional[str] = None
            main_clauses: TList[dict] = []

        contract_struct = await asyncio.to_thread(
            llm_with_timeout.with_structured_output,
            struct_messages,
            ContractStructOutput,
        )
        struct_dict = contract_struct.model_dump()

        # ── Step 2: RAG 查公司规范 ──────────────────────────────────
        self_rag = get_controlled_self_rag()
        user_dept = None
        user_projects: list = []
        if user_token:
            try:
                token_info = decode_jwt_token(user_token)
                user_dept = token_info.departments[0] if token_info.departments else None
                user_projects = token_info.projects or []
            except Exception:
                pass

        rules_result = await asyncio.to_thread(
            self_rag.process,
            query="合同签订规范 公司合同管理制度 法务审核要求 合同风险控制 采购规范 销售规范",
            user_dept=user_dept,
            user_projects=user_projects,
        )
        rules_text = format_docs(rules_result.docs or []) if rules_result.docs else "（未找到相关公司规范，以法律法规为主要校验依据）"

        # ── Step 3: Tool 规则校验 ───────────────────────────────────
        risk_messages = contract_risk_check(
            contract_text=contract_text,
            contract_struct=json.dumps(struct_dict, ensure_ascii=False, indent=2),
            company_rules=rules_text,
        )

        class RiskCheckOutput(BaseModel):
            audit_passed: bool
            audit_level: str
            overall_score: float
            total_risks: int
            high_risks: int
            medium_risks: int
            low_risks: int
            risk_items: TList[dict] = []
            summary: str
            audit_suggestions: TList[str] = []

        risk_result = await asyncio.to_thread(
            llm_with_timeout.with_structured_output,
            risk_messages,
            RiskCheckOutput,
        )
        risk_dict = risk_result.model_dump()

        # ── Step 4: 构造最终审计报告 ───────────────────────────────
        return {
            "success": True,
            # 审计结论
            "audit_passed": risk_dict["audit_passed"],
            "audit_level": risk_dict["audit_level"],
            "overall_score": risk_dict["overall_score"],
            "total_risks": risk_dict["total_risks"],
            "high_risks": risk_dict["high_risks"],
            "medium_risks": risk_dict["medium_risks"],
            "low_risks": risk_dict["low_risks"],
            # 风险点明细
            "risk_items": risk_dict["risk_items"],
            # 建议
            "summary": risk_dict["summary"],
            "audit_suggestions": risk_dict["audit_suggestions"],
            # 合同结构
            "contract_struct": struct_dict,
            # 辅助信息
            "rules_count": len(rules_result.docs) if rules_result.docs else 0,
            "message": (
                f"审计完成：结论={risk_dict['audit_level']}，"
                f"综合评分={risk_dict['overall_score']}，"
                f"风险点 {risk_dict['total_risks']} 个"
                f"（高:{risk_dict['high_risks']} 中:{risk_dict['medium_risks']} 低:{risk_dict['low_risks']}）"
            ),
        }

    except Exception as e:
        logger.error(f"合同审计失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "audit_passed": False,
            "audit_level": "fail",
            "overall_score": 0,
            "total_risks": 0,
            "high_risks": 0,
            "medium_risks": 0,
            "low_risks": 0,
            "risk_items": [],
            "summary": "审计过程中出现错误",
            "audit_suggestions": ["请稍后重试或联系法务部门"],
            "contract_struct": None,
            "rules_count": 0,
            "message": f"合同审计失败: {e}",
        }


# ── 工具三：企业知识库检索 ───────────────────────────────────────────
@mcp.tool()
async def rag_retrieve(
    query: str,
    top_k: int = 5,
    user_token: str = "",
) -> dict:
    """
    检索企业知识库，获取与问题最相关的文档内容。

    使用场景：
    - 用户询问政策、制度、流程等问题时
    - 需要查阅公司文档才能回答的问题

    参数:
        query: 用户的查询问题
        top_k: 返回的最相关文档数量，默认5篇
        user_token: 用户认证令牌（用于 ABAC 权限过滤）

    返回:
        包含检索到的文档列表和生成的回答
    """
    try:
        from controlled_self_rag import get_controlled_self_rag, RetrievalDecision
        from auth import decode_jwt_token

        self_rag = get_controlled_self_rag()

        user_dept = None
        user_projects: list = []
        if user_token:
            try:
                token_info = decode_jwt_token(user_token)
                user_dept = token_info.departments[0] if token_info.departments else None
                user_projects = token_info.projects or []
            except Exception:
                pass

        result = await asyncio.to_thread(
            self_rag.process,
            query=query,
            user_dept=user_dept,
            user_projects=user_projects,
        )

        if result.retrieval_decision == RetrievalDecision.SKIP_RETRIEVE.value:
            return {
                "success": True,
                "should_retrieve": False,
                "is_useful": True,
                "answer": result.answer or "您好！有什么可以帮您的吗？",
                "docs": [],
                "message": "跳过检索（闲聊类问题）",
            }

        if not result.retrieved or not result.is_useful or not result.docs:
            return {
                "success": True,
                "should_retrieve": True,
                "is_useful": False,
                "answer": result.answer or "在知识库中未找到相关信息，建议联系 HR 部门（内线 8001）获取帮助。",
                "docs": [],
                "message": "检索结果无用",
            }

        return {
            "success": True,
            "should_retrieve": True,
            "is_useful": True,
            "answer": result.answer,
            "docs": [
                {
                    "content": doc.get("text", "")[:500],
                    "score": doc.get("score", 0),
                    "metadata": doc.get("metadata", {}),
                }
                for doc in (result.docs or [])[:top_k]
            ],
            "quality_score": result.usefulness_score,
            "needs_escalation": result.needs_escalation,
            "message": f"成功检索到 {len(result.docs)} 篇相关文档",
        }

    except Exception as e:
        logger.error(f"RAG 检索失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "answer": "系统暂时无法处理您的请求，请稍后重试或联系 HR（内线 8001）获取帮助。",
            "docs": [],
        }


# ─────────────────────────────────────────────
# 工具列表导出（方便调试和查看）
# ─────────────────────────────────────────────
java_backend_tools = [
    {
        "name": "contract_parse",
        "description": "解析合同文本结构，提取合同名称、类型、双方当事人、关键条款等关键字段。用于合同审计流程的第一步。",
        "input_schema": {
            "type": "object",
            "properties": {
                "contract_text": {
                    "type": "string",
                    "description": "合同原始文本内容（从 PDF/Word 提取的文字）",
                },
            },
            "required": ["contract_text"],
        },
    },
    {
        "name": "contract_audit",
        "description": "对合同进行完整审计：LLM 解析结构 + RAG 查公司规范 + 规则校验 + 输出审计报告（风险点 + 修改建议 + 是否通过）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "contract_text": {
                    "type": "string",
                    "description": "合同原始文本内容",
                },
                "user_token": {
                    "type": "string",
                    "description": "用户认证令牌（用于 RAG 权限过滤）",
                },
            },
            "required": ["contract_text"],
        },
    },
    {
        "name": "rag_retrieve",
        "description": "检索企业知识库，获取与问题最相关的文档内容。用于政策咨询、制度查询等场景。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户的查询问题"},
                "top_k": {"type": "integer", "description": "返回的最相关文档数量，默认5篇", "default": 5},
                "user_token": {"type": "string", "description": "用户认证令牌（用于 ABAC 权限过滤）"},
            },
            "required": ["query"],
        },
    },
]
