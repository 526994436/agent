"""
Prompt 模板统一管理模块 (prompts.py)

═══════════════════════════════════════════════════════════════════════════════
📚 设计目标
═══════════════════════════════════════════════════════════════════════════════

将所有散落在业务代码中的 LLM Prompt 模板集中管理：

1. 使用 LlamaIndex PromptTemplate 实现变量注入类型安全
2. 支持多轮对话模板（ChatPromptTemplate）
3. 支持 few-shot 示例注入
4. 支持模板继承（基础模板 + 业务扩展）
5. 所有模板集中在一个文件，方便审计、迭代、A/B 测试

═══════════════════════════════════════════════════════════════════════════════
📋 模板分类
═══════════════════════════════════════════════════════════════════════════════

1. SELF_RAG    — 可控 Self-RAG 链路（校验、纠错、生成、优化）
2. SHARED      — 跨模块共享的基础模板片段

═══════════════════════════════════════════════════════════════════════════════
📖 使用示例
═══════════════════════════════════════════════════════════════════════════════

```python
from prompts import (
    # Self-RAG
    QUALITY_CHECK_WITH_DOCS_PROMPT,
    ANSWER_GENERATE_PROMPT,
    STRICT_GENERATE_PROMPT,
)

# 变量注入（自动类型安全）
prompt = ANSWER_GENERATE_PROMPT.format(context="...", query="请假流程是什么？")
messages = ANSWER_GENERATE_PROMPT.format_messages(context="...", query="...")

# 获取 LLM（自动使用 Settings.llm）
llm = get_prompt_llm()
response = llm.chat(messages)
```
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

from llama_index.core.base.llms.base import LLM
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from llama_index.core.prompts import ChatPromptTemplate, PromptTemplate
from llama_index.llms.openai import OpenAI

logger = logging.getLogger("oa_agent.prompts")


# =============================================================================
# 共享模板片段（可复用的部分）
# =============================================================================

SHARED_SYSTEM_PROMPT = SystemMessage(
    content="""你是一个专业的企业 OA 助手。

【槽位填充规则】
当存在缺失字段（pending_slots）时，必须主动向用户询问：
1. 每次只问 1-2 个最关键的字段，不要一次问完
2. 用自然语言提问，不要直接列出字段名
3. 结合已确认的参数自然衔接，例如：已确认"请年假"，则问"请问您想从哪天开始？"
4. 直到所有必填字段都确认完毕，才进入执行阶段"""
)


# =============================================================================
# 1. Self-RAG 模板（controlled_self_rag.py）
# =============================================================================

# ── 答案质量检测员（有文档）───────────────────────────────────────────────

QUALITY_CHECK_WITH_DOCS_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        SystemMessage(content="你是一个严格的质量检测员，只输出 JSON。"),
        HumanMessage(content="""你是企业 OA 助手的【答案质量检测员】。

【用户问题】
{query}

【检索到的文档】
{docs_text}

【AI 生成的答案】
{answer}

【任务】
严格对照文档，判断答案是否有以下 4 类致命错误：

1. FACTUAL_ERROR（事实错误）：
   - 答案中的报销比例、请假天数、流程节点与文档不符
   - 答案声称的限制、人数、金额与文档不一致

2. HALLUCINATION（纯幻觉）：
   - 答案提到不存在的制度、规定、流程
   - 答案提到不存在的审批人姓名、职位
   - 答案编造了文档中完全没有的内容

3. NO_EVIDENCE（无依据）：
   - 答案完全没引用任何文档内容
   - 答案的关键信息在文档中找不到对应

4. OUT_OF_SCOPE（越权/违规）：
   - 泄露了其他部门/员工的私有数据
   - 答案涉及非 OA 范畴的内容（如八卦、隐私等）

【输出格式】（JSON）
{{
  "is_acceptable": true或false,
  "error_types": ["FACTUAL_ERROR"/"HALLUCINATION"/"NO_EVIDENCE"/"OUT_OF_SCOPE"]的数组,
  "details": ["具体错误描述"]的数组,
  "reasoning": "简短推理"
}}

只输出 JSON，不要有其他文字。"""),
    ],
)

# ── 答案生成（兜底）───────────────────────────────────────────────────────

ANSWER_GENERATE_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        SHARED_SYSTEM_PROMPT,
        HumanMessage(content="""基于以下检索结果，回答用户问题。

重要规则：
1. 只基于文档内容回答，不要编造
2. 注明引用来源（格式：参考来源[1]）
3. 如果信息不足，明确说明

【参考文档】
{context}

【用户问题】
{query}

请给出清晰、准确的回答。"""),
    ],
)

# ── 答案优化员 ────────────────────────────────────────────────────────────

ANSWER_REFINE_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        SystemMessage(content="你是一个专业的 OA 助手，只输出改写后的答案。"),
        HumanMessage(content="""你是企业 OA 助手的【答案优化员】。

任务：对已有答案进行精简、规范改写，不要重新生成内容。

【用户问题】
{query}

【原始答案】
{answer}

【参考文档】
{docs_text if docs_text else "（无参考文档）"}

【要求】
1. 保持答案的核心意思不变
2. 精简啰嗦的表述
3. 规范格式和用词
4. 只改写表述，不添加新内容
5. 如需引用，请注明来源格式 [来源X]

请直接输出改写后的答案："""),
    ],
)

# ── 严格生成员（Level 2）────────────────────────────────────────────────

STRICT_GENERATE_LV2_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        SystemMessage(content="你是一个严格的 OA 助手，只输出基于文档的答案。"),
        HumanMessage(content="""你是企业 OA 助手的【严格生成员】。

任务：严格对照参考文档重新生成答案，禁止自由发挥。

【用户问题】
{query}

【参考文档】
{docs_text}

【重要规则】
1. 必须严格基于文档内容，禁止编造
2. 数字、标准、流程必须与文档一致
3. 如文档中没有的信息，明确说明"文档中未提及"
4. 答案要清晰、准确、完整

请生成答案："""),
    ],
)

# ── 查询重写/扩展（Query Writing，用于 Level 3 重检索前）────────────────────

QUERY_REWRITE_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        SystemMessage(content="你是一个专业的企业 OA 知识库检索优化专家，只输出符合要求的 JSON。"),
        HumanMessage(content="""任务：当首次检索效果不佳时，请分析原因并对原始查询进行优化重写，提高二次检索的命中率。

【原始查询】
{query}

【原始检索结果】（效果不好，被判定为无依据或有事实错误）
{old_docs_text}

【重写策略】（选择性组合使用）：
1. 查询扩展：添加同义词、相关术语（如"报销"→"报销/发票/贴票/财务审批"）
2. 查询分解：将复杂问题拆分为多个子查询
3. 查询泛化：从具体到抽象（如"年会礼品报销"→"公司报销制度/福利标准"）
4. 查询具体化：从抽象到具体（如"报销流程"→"差旅费报销流程/发票粘贴规范"）
5. 补充上下文：添加公司/部门/岗位相关的限定词

【输出格式】（严格遵循 JSON）
{{
  "failure_analysis": "简要分析为什么给出的原始检索结果无法解决原始查询的问题。",
  "rewritten_queries": [
    {{
      "query": "重写后的查询语句",
      "strategy": "使用的重写策略（expansion/decomposition/generalization/specialization/contextualization）",
      "reason": "为什么这样重写"
    }}
  ],
  "final_query": "综合多个策略后用于最终检索的主查询（注意：需适合混合检索，既要包含核心自然语言语义，也要补充关键实体词汇）"
}}

要求：
1. 先进行 failure_analysis 反思。
2. 生成 2-3 个不同的重写查询方案。
3. 最终选择 1 个最优查询用于检索，存入 final_query。
4. 只输出 JSON，不要有任何其他文字说明。"""),
    ],
)


# ── 严格生成员（Level 3）────────────────────────────────────────────────

STRICT_GENERATE_LV3_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        SystemMessage(content="你是一个严格的 OA 助手，只输出基于文档的答案。"),
        HumanMessage(content="""你是企业 OA 助手的【严格生成员】。

任务：严格基于新检索到的文档生成答案。

【用户问题】
{query}

【新检索到的文档】
{docs_text}

【重要规则】
1. 必须严格基于文档内容，禁止编造任何内容
2. 数字、标准、流程必须与文档完全一致
3. 如文档中没有的信息，明确说明"文档中未提及"
4. 如果文档完全不相关，请说明"未找到相关信息"

请生成答案："""),
    ],
)


# =============================================================================
# 2. 合同审计模板（contract_audit.py）
# =============================================================================

# ── 合同结构解析 ──────────────────────────────────────────────────────────

CONTRACT_STRUCT_PARSE_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        SystemMessage(content="你是一个专业的企业合同法务顾问，只输出符合要求的 JSON。"),
        HumanMessage(content="""你是企业 OA 助手的【合同结构解析员】。

任务：分析合同文本，提取关键结构信息和条款。

【合同文本】
{contract_text}

【输出格式】（严格 JSON）
{{
  "contract_name": "合同名称（从文本中提取）",
  "contract_type": "合同类型（采购/劳动/租赁/销售/服务/技术/其他）",
  "contract_no": "合同编号（如有）",
  "signing_date": "签订日期（如有）",
  "effective_date": "生效日期（如有）",
  "expiry_date": "到期日期（如有）",
  "party_a": "甲方（我方）名称",
  "party_b": "乙方（对方）名称",
  "total_amount": "合同总金额（如有）",
  "payment_terms": "付款条款摘要",
  "key_obligations_a": "甲方主要义务列表",
  "key_obligations_b": "乙方主要义务列表",
  "penalty_clauses": "违约责任条款摘要",
  "termination_clauses": "终止/解除条款摘要",
  "dispute_resolution": "争议解决方式",
  "governing_law": "适用法律",
  "special_clauses": "特殊条款（如有）",
  "main_clauses": [
    {{
      "clause_no": "条款编号/标题",
      "clause_text": "条款原文（完整引用）",
      "clause_summary": "条款内容摘要"
    }}
  ]
}}

规则：
1. 只输出 JSON，不要有其他文字
2. 所有字段尽量填充，无法提取则填 null
3. main_clauses 列出合同中最重要的 5-10 个条款
4. 条款原文必须一字不差引用原文"""),
    ],
)

# ── 合同风险规则校验 ────────────────────────────────────────────────────

CONTRACT_RISK_CHECK_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        SystemMessage(content="你是一个严格的企业合同法务审计员，只输出符合要求的 JSON。"),
        HumanMessage(content="""你是企业 OA 助手的【合同风险校验员】。

任务：对照公司规范和法律法规，对合同条款进行系统性风险校验，识别所有潜在风险点。

【合同文本】
{contract_text}

【合同结构解析】
{contract_struct}

【公司规范（知识库检索结果）】
{company_rules}

【重要背景】
- 你是为公司法务部门服务的审计助手
- 风险等级：high=高风险（可能导致重大损失或法律纠纷，必须修改）, medium=中风险（存在隐患，建议修改）, low=低风险（最佳实践，建议优化）
- 一份合同可能完全没有风险（返回空 risk_items）

【风险类别参考】（按此分类）
1. 格式条款（high）：提供方免除自身责任、加重对方责任、排除对方主要权利的格式条款
2. 违约责任缺失（high）：对方违约时无明确违约责任或违约成本极低
3. 违约责任过重（medium）：我方违约责任明显高于对方，权利义务不对等
4. 管辖权条款（high）：约定由对方所在地或对我方不利的地方管辖/仲裁
5. 知识产权归属（high）：合同成果的知识产权归属对方或约定不明
6. 保密条款缺失（medium）：涉及核心技术/商业秘密但无保密条款
7. 不可抗力条款缺失（medium）：无不可抗力条款或定义过窄
8. 合同金额/计价不合理（medium）：价格明显偏离市场、计价方式对我方不利
9. 履行期限不明确（low）：关键期限约定模糊可能导致履行争议
10. 解除权条款（high）：对方单方解除权过多，我方缺乏对等解除权
11. 连带责任（high）：约定了不合理的连带担保责任
12. 竞业限制（medium）：对我方员工约定不合理竞业限制
13. 税务风险（medium）：发票/税务条款不符合法规
14. 数据安全（high）：涉及用户数据但无数据安全/隐私保护条款
15. 授权范围（medium）：授权范围过宽可能带来超出预期的法律责任

【输出格式】（严格 JSON）
{{
  "audit_passed": true或false,
  "audit_level": "pass（通过）/pass_with_conditions（有条件通过）/fail（不通过）",
  "overall_score": 0-100的综合评分数字,
  "total_risks": 风险点总数,
  "high_risks": 高风险数,
  "medium_risks": 中风险数,
  "low_risks": 低风险数,
  "risk_items": [
    {{
      "risk_id": 1,
      "category": "风险类别",
      "severity": "high/medium/low",
      "clause_text": "涉及的合同条款原文（完整引用）",
      "risk_description": "风险描述",
      "legal_basis": "相关法律依据（如《民法典》第XXX条）",
      "suggestion": "修改建议"
    }}
  ],
  "summary": "总体评价（1-2句话）",
  "audit_suggestions": ["整体修改建议1", "整体修改建议2"]
}}

规则：
1. 只输出 JSON，不要有其他文字
2. 严格对照原文识别风险，clause_text 必须一字不差引用
3. 评分标准：90-100=pass, 60-89=pass_with_conditions, 0-59=fail
4. 如果合同无明显风险点，risk_items 为空数组，audit_passed=true, audit_level="pass"
5. 必须同时结合合同原文和公司规范进行校验"""),
    ],
)


# =============================================================================
# 便捷格式化函数
# =============================================================================

def format_docs(docs: List[Dict[str, Any]], prefix: str = "[文档{i}] ") -> str:
    """
    将文档列表格式化为字符串（兼容旧代码的 docs_text 变量）。

    用法：
        docs_text = format_docs(retrieved_docs, prefix="[文档{i}] ")
    """
    if not docs:
        return "（无相关文档）"
    lines = []
    for i, doc in enumerate(docs, 1):
        text = doc.get("text", "")
        if text:
            lines.append(f"{prefix.format(i=i)}{text}")
    return "\n\n".join(lines) if lines else "（文档内容为空）"


# ── Self-RAG 模板快捷调用 ──────────────────────────────────────────────────

def quality_check_with_docs(
    query: str, docs_text: str, answer: str
) -> List[BaseMessage]:
    """答案质量检测（有文档）"""
    return QUALITY_CHECK_WITH_DOCS_TEMPLATE.format_messages(
        query=query, docs_text=docs_text, answer=answer
    )


def answer_generate(context: str, query: str) -> List[BaseMessage]:
    """答案生成"""
    return ANSWER_GENERATE_TEMPLATE.format_messages(context=context, query=query)


def answer_refine(
    query: str, answer: str, docs_text: str = ""
) -> List[BaseMessage]:
    """答案优化"""
    return ANSWER_REFINE_TEMPLATE.format_messages(
        query=query, answer=answer, docs_text=docs_text
    )


def strict_generate_lv2(query: str, docs_text: str) -> List[BaseMessage]:
    """严格生成（Level 2）"""
    return STRICT_GENERATE_LV2_TEMPLATE.format_messages(
        query=query, docs_text=docs_text
    )


def strict_generate_lv3(query: str, docs_text: str) -> List[BaseMessage]:
    """严格生成（Level 3）"""
    return STRICT_GENERATE_LV3_TEMPLATE.format_messages(
        query=query, docs_text=docs_text
    )


def query_rewrite(
    query: str, old_docs_text: str
) -> List[BaseMessage]:
    """查询重写/扩展（用于 Level 3 重检索前）"""
    return QUERY_REWRITE_TEMPLATE.format_messages(
        query=query, old_docs_text=old_docs_text
    )


# ── 合同审计快捷调用 ─────────────────────────────────────────────────────

def contract_struct_parse(contract_text: str) -> List[BaseMessage]:
    """合同结构解析"""
    return CONTRACT_STRUCT_PARSE_TEMPLATE.format_messages(contract_text=contract_text)


def contract_risk_check(
    contract_text: str,
    contract_struct: str,
    company_rules: str,
) -> List[BaseMessage]:
    """合同风险校验"""
    return CONTRACT_RISK_CHECK_TEMPLATE.format_messages(
        contract_text=contract_text,
        contract_struct=contract_struct,
        company_rules=company_rules,
    )


# =============================================================================
# 全局 LLM 获取（自动使用 Settings.llm）
# =============================================================================

_llm: Optional[LLM] = None


def get_prompt_llm() -> LLM:
    """
    获取用于 Prompt 执行的 LLM 实例（全局单例）。

    自动从 config.settings 初始化 OpenAI LLM。
    优先使用 Settings.llm，兜底自行创建。
    """
    global _llm
    if _llm is not None:
        return _llm

    from llama_index.core import Settings as li_settings

    # 优先复用 Settings.llm
    if li_settings.llm is not None:
        _llm = li_settings.llm
        return _llm

    # 兜底：从 config 创建
    try:
        from config import settings as app_settings
        _llm = OpenAI(
            model=app_settings.vllm_model,
            api_key=app_settings.vllm_api_key,
            base_url=app_settings.vllm_base_url,
            temperature=0.0,
        )
        logger.info(f"Prompt LLM initialized: {_llm.model}")
    except Exception as e:
        logger.error(f"Failed to initialize prompt LLM: {e}")
        raise

    return _llm



# =============================================================================
# 模板导出清单（__all__）
# =============================================================================

__all__ = [
    # 原始模板
    "QUALITY_CHECK_WITH_DOCS_TEMPLATE",
    "ANSWER_GENERATE_TEMPLATE",
    "ANSWER_REFINE_TEMPLATE",
    "STRICT_GENERATE_LV2_TEMPLATE",
    "STRICT_GENERATE_LV3_TEMPLATE",
    "QUERY_REWRITE_TEMPLATE",
    # 合同审计模板
    "CONTRACT_STRUCT_PARSE_TEMPLATE",
    "CONTRACT_RISK_CHECK_TEMPLATE",
    # 工具函数
    "format_docs",
    "get_prompt_llm",
    # 快捷调用函数
    "quality_check_with_docs",
    "answer_generate",
    "answer_refine",
    "strict_generate_lv2",
    "strict_generate_lv3",
    "query_rewrite",
    # 合同审计快捷调用
    "contract_struct_parse",
    "contract_risk_check",
]
