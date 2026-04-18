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
2. ROUTER      — 意图分类（policy / action / chitchat）
3. ACTION      — 动作参数提取（请假 / 报销 / 权限等）
4. SHARED      — 跨模块共享的基础模板片段

═══════════════════════════════════════════════════════════════════════════════
📖 使用示例
═══════════════════════════════════════════════════════════════════════════════

```python
from prompts import (
    # Self-RAG
    FACTUAL_VERIFY_PROMPT,
    QUALITY_CHECK_WITH_DOCS_PROMPT,
    ANSWER_GENERATE_PROMPT,
    STRICT_GENERATE_PROMPT,
    # Router
    INTENT_CLASSIFY_PROMPT,
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

# ── 事实校验员 ────────────────────────────────────────────────────────────────

FACTUAL_VERIFY_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        SystemMessage(content="你是一个严格的事实校验员，只输出 JSON。"),
        HumanMessage(content="""你是企业 OA 助手的【事实校验员】。

核心原则：你只检查一致性，不做自由创作。不要重写答案，不要添加新内容。

【用户问题】
{query}

【检索到的文档】
{docs_text}

【LLM 生成的答案】
{answer}

【校验任务】
请检查答案中的关键信息是否与文档一致：

1. 数字检查：金额、标准、时间期限、数量等数字是否与文档一致
2. 步骤检查：操作步骤、命令、代码是否与文档一致
3. 规则检查：政策、制度、规定是否与文档描述一致
4. 幻觉检测：答案是否有文档中不存在的内容

【输出格式】（JSON）
{{
  "is_consistent": true或false,
  "errors": [
    {{
      "field": "金额",
      "answer_value": "500元",
      "doc_value": "300元",
      "reason": "答案中的金额与文档不符",
      "correction": "应该是300元"
    }}
  ],
  "warnings": ["任何需要提醒的内容"]
}}

只输出 JSON，不要输出其他内容。"""),
    ],
    namespace="self_rag",
)

# ── 答案质量检测员（无文档）───────────────────────────────────────────────

QUALITY_CHECK_NO_DOCS_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        SystemMessage(content="你是一个严格的质量检测员，只输出 JSON。"),
        HumanMessage(content="""你是企业 OA 助手的【答案质量检测员】。

【用户问题】
{query}

【AI 生成的答案】
{answer}

【任务】
判断这个答案是否有问题。只检测以下 4 类致命错误：

1. FACTUAL_ERROR（事实错误）：答案中的数字、标准与真实情况不符
2. HALLUCINATION（纯幻觉）：答案声称有某制度/流程/审批人，但完全不存在
3. NO_EVIDENCE（无依据）：答案完全没有基于任何参考资料
4. OUT_OF_SCOPE（越权/违规）：答案泄露了不该知道的信息，或答非 OA 范畴

【输出格式】（JSON）
{{
  "is_acceptable": true或false,
  "error_types": ["FACTUAL_ERROR"或"HALLUCINATION"或"NO_EVIDENCE"或"OUT_OF_SCOPE"]的数组,
  "details": ["具体错误描述"]的数组,
  "reasoning": "简短推理"
}}

只输出 JSON，不要有其他文字。"""),
    ],
    namespace="self_rag",
)

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
    namespace="self_rag",
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
    namespace="self_rag",
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
    namespace="self_rag",
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
    namespace="self_rag",
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
    namespace="self_rag",
)


# =============================================================================
# 2. Router 模板（graph.py）
# =============================================================================

INTENT_CLASSIFY_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        SystemMessage(content="""你是一个企业 OA 助手，需要判断用户的意图。

请分析用户最新输入，输出以下 JSON 格式（不要输出其他内容）：
{{
  "intent": "policy" 或 "action" 或 "chitchat",
  "reasoning": "简短推理过程"
}}

意图说明：
- policy: 用户询问公司政策、制度、流程规定等知识性问题（年假、报销流程、考勤规定等）
- action: 用户请求执行某个操作，如请假、报销、重置密码、开通权限、调休申请等
- chitchat: 用户进行与办公无关的闲聊，如"今天天气真好"、"你吃了吗"等

严格按 JSON 格式输出，不要有额外文字。"""),
        HumanMessage(content="用户输入: {query}"),
    ],
    namespace="router",
)


# =============================================================================
# 3. Action 模板（graph.py）
# =============================================================================

ACTION_PARAM_EXTRACT_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        SystemMessage(content="""你是一个企业 OA 动作参数提取器。请从用户输入中提取动作信息。

输出以下 JSON 格式（仅输出 JSON，不要其他内容）：
{{
  "action_type": "leave_request | expense_reimburse | password_reset | permission_open | 其他",
  "params": {{"具体参数键值对，如 start_date, end_date, reason 等"}},
  "confirmation_message": "生成一句确认话术，如'您正在申请 2026-04-10 至 2026-04-12 的请假，共3天，原因是旅游'"
}}

注意：
- 日期格式统一为 YYYY-MM-DD
- 如果参数不完整，params 中置空字符串，confirmation_message 中说明"以下信息缺失：xxx"
- action_type 必须使用上述标准类型之一"""),
    ],
    namespace="action",
)


# =============================================================================
# 便捷格式化函数（兼容旧接口，无需改动业务代码）
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

def factual_verify(query: str, docs_text: str, answer: str) -> List[BaseMessage]:
    """事实校验员 Prompt"""
    return FACTUAL_VERIFY_TEMPLATE.format_messages(
        query=query, docs_text=docs_text, answer=answer
    )


def quality_check_no_docs(query: str, answer: str) -> List[BaseMessage]:
    """答案质量检测（无文档）"""
    return QUALITY_CHECK_NO_DOCS_TEMPLATE.format_messages(
        query=query, answer=answer
    )


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


# ── Router 模板快捷调用 ──────────────────────────────────────────────────

def intent_classify(query: str) -> List[BaseMessage]:
    """意图分类"""
    return INTENT_CLASSIFY_TEMPLATE.format_messages(query=query)


def action_param_extract(query: str) -> List[BaseMessage]:
    """动作参数提取"""
    return ACTION_PARAM_EXTRACT_TEMPLATE.format_messages(query=query)


def api_intent_judge() -> List[BaseMessage]:
    """API 意图判断"""
    return API_INTENT_JUDGE_TEMPLATE.format_messages()


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


def chat_with_prompt(
    messages: Sequence[BaseMessage],
    llm: Optional[LLM] = None,
    **kwargs,
) -> str:
    """
    便捷函数：用指定 Prompt（messages）执行 LLM chat，返回响应文本。

    自动注入全局 CallbackHandler（链路追踪）。
    """
    from observability import get_oa_callback_handler

    model = llm or get_prompt_llm()
    handler = get_oa_callback_handler()

    chat_kwargs: Dict[str, Any] = {"messages": list(messages)}
    if handler:
        chat_kwargs["additional_kwargs"] = {"callbacks": [handler]}
    chat_kwargs.update(kwargs)

    try:
        response = model.chat(**chat_kwargs)
        return str(response)
    except Exception as e:
        logger.error(f"LLM chat failed: {e}")
        return ""


# =============================================================================
# 模板导出清单（__all__）
# =============================================================================

__all__ = [
    # 原始模板
    "FACTUAL_VERIFY_TEMPLATE",
    "QUALITY_CHECK_NO_DOCS_TEMPLATE",
    "QUALITY_CHECK_WITH_DOCS_TEMPLATE",
    "ANSWER_GENERATE_TEMPLATE",
    "ANSWER_REFINE_TEMPLATE",
    "STRICT_GENERATE_LV2_TEMPLATE",
    "STRICT_GENERATE_LV3_TEMPLATE",
    "INTENT_CLASSIFY_TEMPLATE",
    "ACTION_PARAM_EXTRACT_TEMPLATE",
    # 工具函数
    "format_docs",
    "get_prompt_llm",
    "chat_with_prompt",
    # 快捷调用函数
    "factual_verify",
    "quality_check_no_docs",
    "quality_check_with_docs",
    "answer_generate",
    "answer_refine",
    "strict_generate_lv2",
    "strict_generate_lv3",
    "intent_classify",
    "action_param_extract",
]
