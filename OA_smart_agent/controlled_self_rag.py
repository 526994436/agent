"""
================================================================================
可控 Self-RAG 模块 (controlled_self_rag.py)
================================================================================

【模块简介】
这是一个"智能检索增强"系统，专门为企业 OA 助手设计。
你可以把它想象成一个超级助理，能够自动判断什么时候需要查阅公司文档，
什么时候可以直接回答，并且确保回答的内容准确无误。

【核心比喻 - 理解整个系统】
想象你去医院看病的过程：
1. 分诊台（检索路由）：护士判断你是需要去专科检查，还是只是小问题
2. 检查室（Rerank 精排）：医生用专业仪器（评分系统）筛选出最相关的检查结果
3. 诊断室（有用性判断）：医生判断这些检查结果能不能支撑诊断
4. 处方核对（分级纠错）：药剂师按三级标准复核处方，发现问题分级处理

我们的系统就是用类似的方式处理用户问题！

【为什么叫"可控"Self-RAG？】

传统 RAG 的问题：
- AI 可能会瞎编答案（幻觉）
- 不知道什么时候该查资料，什么时候不需要
- 检索到的内容质量参差不齐

我们的解决方案（可控 Self-RAG）：
1. 【规则兜底】：明确的关键词规则处理 90% 的常见情况（比如问"报销"就一定检索）
2. 【量化评估】：用 BGE-Rerank 评分系统，精确判断文档质量
3. 【最小化 AI】：AI 只做两件事：处理模糊情况、校验事实一致性

【核心节点】

节点1：有用性判断（UsefulnessChecker）
├── 作用：判断"检索到的资料有没有用"
├── 比喻：就像论文查重系统
│         - 相似度 > 0.3 → 有参考价值，保留
│         - 相似度 < 0.3 → 不相关，丢弃
└── 特点：完全量化，可配置阈值，可监控

节点2：分级纠错（CorrectionPipeline）
├── 作用：检测并修复答案质量，从轻到重最多纠错 3 次
├── 比喻：就像药剂师处方的三级复核
│         - LLM 生成后，检测是否有 4 类致命错误
│         - 有问题则分级处理：无用→拒答，部分错→重生成，完全错→重新检索
└── 特点：从轻到重，最多重试 3 次，无效则拒答

【完整流程图】
┌─────────────────────────────────────────────────────────────────────────┐
│                          用户提问                                        │
│                    "年会礼品报销有限额吗？"                               │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 【第一站：ABAC-RAG 检索】                                                │
│                                                                          │
│  - 检查你的权限（能看哪些部门的文档？）                                    │
│  - 从向量数据库找到 50 篇候选文档                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 【第一站：BGE-Rerank 精排】                                               │
│                                                                          │
│  AI 评分排序：                                                            │
│  - 文档A（报销流程）: 0.82 分 ✓                                          │
│  - 文档B（礼品标准）: 0.75 分 ✓                                          │
│  - 文档C（团建费用）: 0.31 分 ✗ (低于阈值，丢弃)                          │
│                                                                          │
│  保留前 5 篇最相关的文档                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 【第二站：有用性判断】                                                     │
│                                                                          │
│  问：这些文档能回答我的问题吗？                                            │
│  评分：0.78 分 > 0.3 阈值 → 能用！                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 【第三站：LLM 生成答案】                                                  │
│                                                                          │
│  基于文档生成回答：                                                        │
│  "根据公司财务制度，年会礼品报销限额为 500 元/人..."                        │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 【第四站：分级纠错】                                                     │
│                                                                          │
│  LLM 质量检测：                                                          │
│  - FACTUAL_ERROR: 数字/标准与文档不符？ → 重生成                          │
│  - HALLUCINATION: 编造不存在的制度？ → 重新检索                          │
│  - NO_EVIDENCE: 无文档依据？ → 拒答或重检                               │
│                                                                          │
│  结论：回答通过质量检查，输出！                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    最终答案（带引用来源）

【配置说明】
所有阈值和关键词都从 config.py 的 settings 读取，方便：
- 调参优化（改一个数字就行）
- 监控告警（记录每次判断的分数）
- A/B 测试（不同部门用不同配置）

【使用示例】
>>> from controlled_self_rag import get_controlled_self_rag
>>> rag = get_controlled_self_rag()
>>> result = rag.process(
...     query="年会礼品报销有限额吗？",
...     user_dept="财务部"
... )
>>> print(result.answer)
"根据公司财务制度第三条，年会礼品报销限额为500元/人..."
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger("oa_agent.controlled_self_rag")


# =============================================================================
# 第一部分：数据结构定义（就像表格的表头）
# =============================================================================

class RetrievalDecision(str, Enum):
    """
    检索路由决策 - 判断"是否需要查资料"

    就像图书馆管理员判断读者的问题：
    - MUST_RETRIEVE: 必须检索（关键词明确指向 OA 知识库，如"年假"、"报销"）
    - SKIP_RETRIEVE: 跳过检索（闲聊、寒暄、无关问题）
    - MAYBE_RETRIEVE: 模糊情况，交由 LLM 二次判断
    """
    MUST_RETRIEVE = "must_retrieve"  # 必须检索
    SKIP_RETRIEVE = "skip_retrieve"  # 跳过检索
    MAYBE_RETRIEVE = "maybe_retrieve"  # 模糊判断


class UsefulnessDecision(str, Enum):
    """
    有用性判断决策 - 判断"检索到的内容有没有用"

    就像考试评分：
    - USEFUL: 及格（分数 >= 阈值），可以使用
    - NOT_USEFUL: 不及格（分数 < 阈值），需要人工处理
    """
    USEFUL = "useful"           # 有用
    NOT_USEFUL = "not_useful"   # 无用


class AnswerErrorType(str, Enum):
    """
    答案致命错误类型 - Self-RAG 判定"回答不行"的 4 类标准

    只盯这 4 类致命问题，其他小问题不纠结：
    1. FACTUAL_ERROR: 事实错误 - 报销比例、请假天数等与知识库不符
    2. HALLUCINATION: 纯幻觉 - 编不存在的制度、流程、审批人
    3. NO_EVIDENCE: 无依据 - 答案没引用任何 OA 知识库内容
    4. OUT_OF_SCOPE: 越权/违规 - 泄露其他部门数据、答非 OA 范畴
    """
    FACTUAL_ERROR = "factual_error"      # 事实错误
    HALLUCINATION = "hallucination"      # 纯幻觉
    NO_EVIDENCE = "no_evidence"          # 无依据
    OUT_OF_SCOPE = "out_of_scope"        # 越权/违规


class CorrectionLevel(int, Enum):
    """
    纠错级别 - 从轻到重的 3 级处理流程

    OA 场景只用到前 3 级：
    1 (LIGHT): 轻度纠错 - 仅改写，不重检索（答案意思对但措辞乱）
    2 (MEDIUM): 中度纠错 - 重生成，不重检索（部分事实错但文档对）
    3 (HEAVY): 重度纠错 - 重新检索 + 重排 + 再生（检索结果无关）
    """
    LIGHT = 1   # 轻度纠错
    MEDIUM = 2  # 中度纠错
    HEAVY = 3   # 重度纠错
    REJECT = 0  # 直接拒答（兜底）


@dataclass
class UsefulnessResult:
    """
    有用性判断结果 - 记录文档质量评估的详细信息
    
    属性说明：
    - decision: 最终决策（有用/无用）
    - score_threshold: 设定的阈值（参考线）
    - actual_score: 实际平均分（和阈值对比）
    - filtered_count: 被过滤掉的文档数量
    - remaining_count: 保留的文档数量
    - reason: 判断理由
    """
    decision: UsefulnessDecision
    score_threshold: float
    actual_score: float
    filtered_count: int
    remaining_count: int
    reason: str


@dataclass
class QualityCheckResult:
    """
    答案质量检测结果 - Self-RAG 判定"回答不行"的结果

    只检测 4 类致命问题：
    - error_types: 发现的错误类型列表
    - is_acceptable: 是否可以接受（False = 需要纠错）
    - details: 错误详情列表
    """
    is_acceptable: bool
    error_types: List[AnswerErrorType] = field(default_factory=list)
    details: List[str] = field(default_factory=list)
    reasoning: str = ""


class QualityCheckOutput(BaseModel):
    """用于 LLM structured output 的 Pydantic 模型"""
    is_acceptable: bool = Field(description="答案是否可接受")
    error_types: List[str] = Field(description="错误类型列表，如 factual_error, hallucination, no_evidence, out_of_scope")
    details: List[str] = Field(description="错误详情列表")
    reasoning: str = Field(description="判断理由")


@dataclass
class CorrectionResult:
    """
    纠错结果 - 分级纠错处理的结果

    属性说明：
    - success: 是否成功修复
    - level: 使用的纠错级别
    - corrected_answer: 纠错后的答案
    - original_answer: 原始答案
    - remaining_errors: 剩余未修复的错误
    - final_decision: 最终决策（输出/拒答）
    """
    success: bool
    level: CorrectionLevel
    corrected_answer: str
    original_answer: str
    remaining_errors: List[str] = field(default_factory=list)
    final_decision: str = "output"  # output 或 reject


@dataclass
class ControlledSelfRAGResult:
    """
    可控 Self-RAG 最终结果 - 整个系统的输出汇总

    包含：
    - retrieval_decision: 检索路由决策
    - 检索结果（查到了什么文档）
    - 有用性评估（文档质量如何）
    - 分级纠错结果（回答是否经过纠错）
    - 最终答案（给用户的回复）
    """
    # 检索路由决策
    retrieval_decision: str = "must_retrieve"  # must_retrieve / skip_retrieve / maybe_retrieve

    # 检索结果 - 实际查到了什么
    retrieved: bool = False
    docs: List[Dict[str, Any]] = field(default_factory=list)

    # 有用性判断 - 文档质量评估
    is_useful: bool
    usefulness_score: float = 0.0

    # 最终答案 - 给用户的回复
    answer: str = ""

    # 状态标记 - 是否需要人工介入
    needs_escalation: bool = False
    escalation_reason: str = ""


# =============================================================================
# 第二部分：节点1 - 有用性判断（UsefulnessChecker）
# =============================================================================

class UsefulnessChecker:
    """
    有用性判断节点 - 判断"检索到的文档有没有用"

    【重构后的逻辑】
    1. 严格阈值过滤：分数 < 阈值 → 直接丢弃
    2. 无兜底策略：如果过滤后为空 → 查询重写 → 重检索
    3. 快速失败：重检索仍无结果 → 判定为 NOT_USEFUL，快速失败

    【设计理念】
    想象你让助理去找相关资料：
    - 助理找回了 10 份文档
    - 但其中 7 份是"团建方案"，2 份是"年会预算"，只有 1 份是"礼品报销标准"
    - 你需要筛选出真正有用的那 1-2 份文档

    BGE-Rerank 就是这个"筛选过程"，它给每份文档打分：
    - 分数 >= 0.3 → 及格，是有用的文档
    - 分数 < 0.3 → 不及格，丢弃（不再兜底保留低分文档）
    """

    def __init__(
        self,
        score_threshold: float = 0.3,
    ):
        """
        初始化有用性检查器

        参数说明：
        - score_threshold: 分数阈值（默认 0.3），低于这个分数的文档会被丢弃
        """
        from config import settings

        self.score_threshold = score_threshold or settings.rerank_score_threshold

    def check(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        user_dept: Optional[str] = None,
        user_projects: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], UsefulnessResult]:
        """
        执行有用性判断（严格模式，无兜底）

        逻辑流程：
        1. 严格阈值过滤：分数 < 阈值 → 丢弃
        2. 如果过滤后为空 → 查询重写 → 重检索
        3. 如果重检索仍无结果 → NOT_USEFUL，快速失败

        参数：
        - query: 用户的问题
        - docs: Rerank 排序后的文档列表（每份文档有 rerank_score 字段）
        - user_dept: 用户部门（用于 ABAC 权限过滤）
        - user_projects: 用户项目列表（用于 ABAC 权限过滤）

        返回：(过滤后的文档列表, 判断结果)
        """
        # 如果没有文档 → 直接判定为"无用"
        if not docs:
            return [], UsefulnessResult(
                decision=UsefulnessDecision.NOT_USEFUL,
                score_threshold=self.score_threshold,
                actual_score=0.0,
                filtered_count=0,
                remaining_count=0,
                reason="文档列表为空",
            )

        # ─────────────────────────────────────────────────────────────────
        # Step 1: 严格阈值过滤（移除兜底逻辑）
        # ─────────────────────────────────────────────────────────────────
        filtered_docs = []
        low_score_count = 0

        for doc in docs:
            rerank_score = doc.get("rerank_score", doc.get("score", 0.0))
            if rerank_score >= self.score_threshold:
                filtered_docs.append(doc)
            else:
                low_score_count += 1

        # 严格模式：有文档达到阈值 → 通过
        if filtered_docs:
            avg_score = sum(
                d.get("rerank_score", d.get("score", 0))
                for d in filtered_docs
            ) / len(filtered_docs)

            logger.info(
                "usefulness_check_passed",
                extra={
                    "decision": UsefulnessDecision.USEFUL.value,
                    "threshold": self.score_threshold,
                    "avg_score": avg_score,
                    "filtered_count": low_score_count,
                    "remaining_count": len(filtered_docs),
                    "component": "controlled_self_rag",
                }
            )

            return filtered_docs, UsefulnessResult(
                decision=UsefulnessDecision.USEFUL,
                score_threshold=self.score_threshold,
                actual_score=avg_score,
                filtered_count=low_score_count,
                remaining_count=len(filtered_docs),
                reason=f"通过有用性检查（{len(filtered_docs)} 个文档 >= 阈值 {self.score_threshold}）",
            )

        # ─────────────────────────────────────────────────────────────────
        # Step 2: 过滤后为空 → 查询重写 + 重检索
        # ─────────────────────────────────────────────────────────────────
        logger.info(
            "usefulness_check_empty_after_filter",
            extra={
                "original_query": query[:50],
                "component": "controlled_self_rag",
            }
        )

        rewritten_query = self._query_rewrite(query, docs)

        # 重新检索
        new_docs = self._reretrieve(
            query=rewritten_query,
            user_dept=user_dept,
            user_projects=user_projects,
        )

        if not new_docs:
            # 重检索仍无结果 → 快速失败
            logger.warning("usefulness_check_reretrieve_empty")
            return [], UsefulnessResult(
                decision=UsefulnessDecision.NOT_USEFUL,
                score_threshold=self.score_threshold,
                actual_score=0.0,
                filtered_count=len(docs),
                remaining_count=0,
                reason="重检索后无相关文档",
            )

        # 对重检索结果再次过滤
        new_filtered = []
        new_low_score = 0
        for doc in new_docs:
            rerank_score = doc.get("rerank_score", doc.get("score", 0.0))
            if rerank_score >= self.score_threshold:
                new_filtered.append(doc)
            else:
                new_low_score += 1

        if not new_filtered:
            # 重检索后过滤仍为空 → 快速失败
            logger.warning("usefulness_check_reretrieve_filtered_empty")
            return [], UsefulnessResult(
                decision=UsefulnessDecision.NOT_USEFUL,
                score_threshold=self.score_threshold,
                actual_score=0.0,
                filtered_count=len(new_docs),
                remaining_count=0,
                reason="重检索后无相关文档",
            )

        # 重检索成功
        avg_score = sum(
            d.get("rerank_score", d.get("score", 0))
            for d in new_filtered
        ) / len(new_filtered)

        logger.info(
            "usefulness_check_reretrieve_success",
            extra={
                "original_query": query[:50],
                "rewritten_query": rewritten_query[:50],
                "remaining_count": len(new_filtered),
                "component": "controlled_self_rag",
            }
        )

        return new_filtered, UsefulnessResult(
            decision=UsefulnessDecision.USEFUL,
            score_threshold=self.score_threshold,
            actual_score=avg_score,
            filtered_count=new_low_score,
            remaining_count=len(new_filtered),
            reason=f"重检索后通过（{len(new_filtered)} 个文档 >= 阈值 {self.score_threshold}）",
        )

    def _query_rewrite(self, query: str, old_docs: List[Dict[str, Any]]) -> str:
        """
        查询重写（内部方法）

        基于原始查询和之前的低质量检索结果，对查询进行优化重写。
        """
        from prompts import query_rewrite, format_docs
        from graph import llm_with_timeout
        from pydantic import BaseModel
        from typing import List as TList

        old_docs_text = format_docs(old_docs)

        try:
            messages = query_rewrite(query=query, old_docs_text=old_docs_text)

            class QueryRewriteOutput(BaseModel):
                rewritten_queries: TList[dict]
                final_query: str

            result = llm_with_timeout.with_structured_output(
                messages, QueryRewriteOutput
            )

            logger.info(
                "usefulness_check_query_rewrite_done",
                extra={
                    "original_query": query[:50],
                    "rewritten_query": result.final_query[:50],
                    "candidate_count": len(result.rewritten_queries),
                }
            )

            return result.final_query
        except Exception as e:
            logger.warning(
                "usefulness_check_query_rewrite_failed",
                extra={"error": str(e)}
            )
            return query  # 失败时使用原始查询

    def _reretrieve(
        self,
        query: str,
        user_dept: Optional[str] = None,
        user_projects: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        from rag import build_llama_rag_pipeline
        pipeline = build_llama_rag_pipeline(
            abac_depts=[user_dept] if user_dept else [],
            abac_projects=user_projects or [],
        )
        docs = pipeline.retrieve(query=query, top_k=50, use_rerank=False)

        from reranker_llamaindex import get_llama_reranker
        reranker = get_llama_reranker()
        docs = reranker.rerank(query=query, documents=docs, top_k=5)

        return docs


# =============================================================================
# 答案质量检测器（AnswerQualityChecker）
# =============================================================================

class AnswerQualityChecker:
    """
    答案质量检测器 - Self-RAG 判定"回答不行"的核心模块

    只盯 4 类致命问题，其他小问题不纠结：
    1. FACTUAL_ERROR: 事实错误 - 报销比例、请假天数等与知识库不符
    2. HALLUCINATION: 纯幻觉 - 编不存在的制度、流程、审批人
    3. NO_EVIDENCE: 无依据 - 答案没引用任何 OA 知识库内容
    4. OUT_OF_SCOPE: 越权/违规 - 泄露其他部门数据、答非 OA 范畴
    """

    def __init__(self):
        pass

    def check(
        self,
        query: str,
        answer: str,
        docs: List[Dict[str, Any]],
    ) -> QualityCheckResult:
        """
        检测答案质量，判断是否需要纠错

        参数：
        - query: 用户问题
        - answer: LLM 生成的答案
        - docs: 检索到的文档（作为参考答案）

        返回：QualityCheckResult（是否可接受、错误类型列表）
        """
        if not answer or not answer.strip():
            return QualityCheckResult(
                is_acceptable=False,
                error_types=[AnswerErrorType.NO_EVIDENCE],
                details=["答案为空"],
                reasoning="答案为空，无法提供有效回复",
            )

        return self._check_with_docs(query, answer, docs)

    def _check_with_docs(
        self,
        query: str,
        answer: str,
        docs: List[Dict[str, Any]],
    ) -> QualityCheckResult:
        """有文档时检测 4 类致命错误"""
        from prompts import quality_check_with_docs, format_docs
        from graph import llm_wrapper

        docs_text = format_docs(docs)
        messages = quality_check_with_docs(query=query, docs_text=docs_text, answer=answer)

        try:
            result = llm_wrapper.with_structured_output(
                messages, QualityCheckOutput
            )
            error_types = [AnswerErrorType(e) for e in result.error_types]
            return QualityCheckResult(
                is_acceptable=result.is_acceptable,
                error_types=error_types,
                details=result.details,
                reasoning=result.reasoning,
            )
        except Exception as e:
            logger.warning("quality_check_failed", extra={"error": str(e)})

        # 检测失败时保守：有问题则标记
        return QualityCheckResult(
            is_acceptable=False,
            error_types=[AnswerErrorType.FACTUAL_ERROR],
            details=["质量检测失败"],
            reasoning="检测异常，保守标记为需要纠错",
        )


# =============================================================================
# 第五部分：可控 Self-RAG 主流程
# =============================================================================

class ControlledSelfRAG:
    """
    可控 Self-RAG 主流程 - 串联检索、有用性判断和分级纠错
    
    【整体流程】
    用户问题
    → [ABAC-RAG 检索] → 查资料
    → [BGE-Rerank 精排] → 排序筛选
    → [节点1：有用性判断] → 判断资料有没有用
    → [LLM 生成答案] → 撰写回复
    → [节点2：分级纠错闭环] → 检测并修复答案质量
    → 最终回答

    【类比】
    想象去医院看病：
    1. 挂号处（RAG）→ 拿出你的病历本和检查报告
    2. 检查室（Rerank）→ 用仪器分析哪份报告最重要
    3. 医生诊室（有用性判断）→ 看完报告，判断能得出什么结论
    4. 开药处（LLM）→ 开处方
    5. 药剂师核对（分级纠错）→ 检查处方和报告是否一致
    6. 取药回家（最终回答）
    """

    def __init__(
        self,
        usefulness_checker: Optional[UsefulnessChecker] = None,
    ):
        """
        初始化可控 Self-RAG 流程

        参数说明：
        - usefulness_checker: 有用性检查器（默认为新建实例）
        """
        from config import settings

        self.usefulness_checker = usefulness_checker or UsefulnessChecker()
        self.correction_pipeline = None  # 延迟初始化（在 process 中设置）
        self.enabled = settings.controlled_self_rag_enabled

    def process(
        self,
        query: str,
        user_dept: Optional[str] = None,
        user_projects: Optional[List[str]] = None,
        raw_docs: Optional[List[Dict[str, Any]]] = None,
    ) -> ControlledSelfRAGResult:
        """
        可控 Self-RAG 主流程入口

        输入用户问题，输出完整的处理结果

        参数：
        - query: 用户的问题
        - user_dept: 用户所属部门（用于 ABAC 权限过滤）
        - user_projects: 用户参与的项目列表（用于 ABAC 权限过滤）
        - raw_docs: 可选，外部传入的检索结果（如果有的话）

        返回：ControlledSelfRAGResult（包含所有中间结果和最终答案）
        """
        self._user_dept = user_dept
        self._user_projects = user_projects or []
        # 延迟初始化 CorrectionPipeline，注入 ABAC 权限参数
        if self.correction_pipeline is None:
            self.correction_pipeline = CorrectionPipeline(
                user_dept=user_dept,
                user_projects=user_projects or [],
            )

        # 如果功能被禁用 → 返回快速结果（跳过整个流程）
        if not self.enabled:
            return ControlledSelfRAGResult(
                retrieval_decision=RetrievalDecision.MUST_RETRIEVE.value,
                retrieved=False,
                is_useful=True,
                needs_escalation=False,
            )

        # ─────────────────────────────────────────────────────────────────
        # 第零步：关键词路由决策（SKIP / MUST / MAYBE）
        # ─────────────────────────────────────────────────────────────────
        retrieval_decision = self._route_by_keywords(query)
        if retrieval_decision == RetrievalDecision.SKIP_RETRIEVE:
            return ControlledSelfRAGResult(
                retrieval_decision=RetrievalDecision.SKIP_RETRIEVE.value,
                retrieved=False,
                is_useful=True,
                needs_escalation=False,
            )

        # ─────────────────────────────────────────────────────────────────
        # 第一步：ABAC-RAG 检索（如果没有外部传入结果）
        # ─────────────────────────────────────────────────────────────────
        # 从向量数据库检索相关文档
        docs = raw_docs or []  # 如果外部传入了结果，直接用；否则去检索
        
        # 需要检索的情况
        if not docs:
            from rag import build_llama_rag_pipeline, LlamaRAGPipeline

            pipeline = build_llama_rag_pipeline(
                abac_depts=[self._user_dept] if self._user_dept else [],
                abac_projects=self._user_projects or [],
            )
            docs = pipeline.retrieve(query=query, top_k=50, use_rerank=False)

        # ─────────────────────────────────────────────────────────────────
        # 第三步：Rerank 精排
        # ─────────────────────────────────────────────────────────────────
        from reranker_llamaindex import get_llama_reranker
        reranker = get_llama_reranker(
            use_cohere=settings.reranking_use_cohere if hasattr(settings, 'reranking_use_cohere') else False,
        )
        docs = reranker.rerank(
            query=query,
            documents=docs,
            top_k=5,
        )

        # ─────────────────────────────────────────────────────────────────
        # 第四步：有用性判断（BGE-Rerank 阈值过滤）
        # ─────────────────────────────────────────────────────────────────
        # 检查检索到的文档是否足够有用
        filtered_docs, usefulness_result = self.usefulness_checker.check(
            query=query,
            docs=docs,
            user_dept=self._user_dept,
            user_projects=self._user_projects,
        )
        
        # 如果有用性判断为"无用" → 标记需要人工介入
        if usefulness_result.decision == UsefulnessDecision.NOT_USEFUL:
            return ControlledSelfRAGResult(
                retrieval_decision=RetrievalDecision.MUST_RETRIEVE.value,
                retrieved=True,
                docs=docs,
                is_useful=False,
                usefulness_score=usefulness_result.actual_score,
                needs_escalation=True,
                escalation_reason="检索结果有用性不足，建议联系 HR 部门",
            )

        # ─────────────────────────────────────────────────────────────────
        # 第五步：LLM 生成答案
        # ─────────────────────────────────────────────────────────────────
        answer = self._generate_answer(query, filtered_docs)

        # ─────────────────────────────────────────────────────────────────
        # 第六步：分级纠错闭环（替换原事实校验）
        # ─────────────────────────────────────────────────────────────────
        # 先检测答案质量，如有致命问题则触发分级纠错
        correction_result = self.correction_pipeline.correct(
            query=query,
            answer=answer,
            docs=filtered_docs,
            retry_count=0,
        )

        # 使用纠错后的答案
        final_answer = correction_result.corrected_answer

        # 标记是否需要人工介入（拒答时）
        needs_escalation = correction_result.final_decision == "reject"

        # ─────────────────────────────────────────────────────────────────
        # 第七步：返回结果
        # ─────────────────────────────────────────────────────────────────
        return ControlledSelfRAGResult(
            retrieval_decision=RetrievalDecision.MUST_RETRIEVE.value,
            retrieved=True,
            docs=filtered_docs,
            is_useful=True,
            usefulness_score=usefulness_result.actual_score,
            answer=final_answer,
            needs_escalation=needs_escalation,
            escalation_reason="系统无法准确回答，建议联系 HR" if needs_escalation else "",
        )

    def _route_by_keywords(self, query: str) -> str:
        """
        基于关键词的检索路由决策。

        通过检查查询中是否包含必须检索或跳过检索的关键词，
        来快速判断是否需要查知识库。

        参数：
        - query: 用户的问题

        返回：RetrievalDecision 枚举值字符串
        """
        from config import settings

        query_lower = query.lower()

        skip_keywords = [
            "你好", "您好", "嗨", "hi", "hello", "hi there",
            "你是谁", "叫什么", "你能做什么", "介绍一下", "天气",
            "谢谢", "thanks", "再见", "拜拜", "bye",
        ]
        skip_keywords.extend(getattr(settings, "retrieval_skip_keywords", []))

        for kw in skip_keywords:
            if kw.lower() in query_lower:
                logger.info(
                    "retrieval_routing_skip",
                    extra={"query": query[:50], "matched_keyword": kw}
                )
                return RetrievalDecision.SKIP_RETRIEVE

        must_keywords = [
            "请假", "报销", "年假", "病假", "事假",
            "加班", "考勤", "审批", "费用",
            "密码", "权限", "账号", "系统",
            "制度", "政策", "流程", "规范", "标准",
            "怎么", "如何", "是什么", "多少",
            "合同", "协议", "规定", "规则",
            "福利", "奖金", "薪酬", "工资",
            "培训", "晋升", "考核",
        ]
        must_keywords.extend(getattr(settings, "retrieval_must_keywords", []))

        for kw in must_keywords:
            if kw.lower() in query_lower:
                logger.info(
                    "retrieval_routing_must",
                    extra={"query": query[:50], "matched_keyword": kw}
                )
                return RetrievalDecision.MUST_RETRIEVE

        return RetrievalDecision.MAYBE_RETRIEVE

    def _generate_answer(
        self,
        query: str,
        docs: List[Dict[str, Any]],
    ) -> str:
        """
        生成带引用的答案
        
        基于检索到的文档，让 AI 生成回答，并注明来源
        
        参数：
        - query: 用户的问题
        - docs: 筛选后的文档列表
        
        返回：AI 生成的回答（带引用标注）
        """
        from prompts import answer_generate, format_docs
        from graph import llm_with_timeout

        docs_text = format_docs(docs, prefix="[来源{i}] ")
        messages = answer_generate(context=docs_text, query=query)

        try:
            response = llm_with_timeout(messages)
            return response.content
        except Exception as e:
            # 生成失败 → 返回错误提示
            logger.error(
                "controlled_self_rag_answer_failed",
                extra={"error": str(e), "component": "controlled_self_rag"}
            )
            return "生成答案时遇到问题，请稍后重试。"


# =============================================================================
# 分级纠错闭环（CorrectionPipeline）
# =============================================================================

class CorrectionPipeline:
    """
    分级纠错闭环 - 从「轻量改写」到「重新检索」再到「直接拒答」
    """

    def __init__(
        self,
        user_dept: Optional[str] = None,
        user_projects: Optional[List[str]] = None,
    ):
        self._user_dept = user_dept
        self._user_projects = user_projects or []
        self.quality_checker = AnswerQualityChecker()

    def correct(
        self,
        query: str,
        answer: str,
        docs: List[Dict[str, Any]],
        retry_count: int = 0,
    ) -> CorrectionResult:
        """
        执行分级纠错闭环

        参数：
        - query: 用户问题
        - answer: 原始答案
        - docs: 当前检索到的文档
        - retry_count: 当前重试次数（用于控制循环）

        返回：CorrectionResult（纠错后的答案或拒答）
        """
        logger.info(
            "correction_pipeline_start",
            extra={
                "query": query[:50],
                "retry_count": retry_count,
                "doc_count": len(docs),
                "component": "correction_pipeline",
            }
        )

        # 安全限制：最多纠错 3 次，防止无限循环
        max_retries = 3
        if retry_count >= max_retries:
            logger.warning(
                "correction_max_retries_reached",
                extra={"retry_count": retry_count, "component": "correction_pipeline"}
            )
            return self._reject(
                answer,
                "系统已尝试多次优化，仍无法给出满意答案",
            )

        # Step 1: 检测答案质量
        quality_result = self.quality_checker.check(query, answer, docs)

        if quality_result.is_acceptable:
            logger.info(
                "answer_quality_acceptable",
                extra={"component": "correction_pipeline"}
            )
            return CorrectionResult(
                success=True,
                level=CorrectionLevel.HEAVY + 1,  # 特殊值表示无需纠错
                corrected_answer=answer,
                original_answer=answer,
                final_decision="output",
            )

        # Step 2: 根据错误类型决定纠错级别
        error_types = quality_result.error_types

        # OUT_OF_SCOPE 直接拒答（最严重）
        if AnswerErrorType.OUT_OF_SCOPE in error_types:
            logger.info(
                "out_of_scope_detected_rejecting",
                extra={"details": quality_result.details, "component": "correction_pipeline"}
            )
            return self._reject(answer, quality_result.details[0] if quality_result.details else "问题超出系统回答范围")

        # NO_EVIDENCE → 重度纠错（重新生成）
        if AnswerErrorType.NO_EVIDENCE in error_types:
            logger.info(
                "no_evidence_with_docs_heavy_correction",
                extra={"component": "correction_pipeline"}
            )
            return self._level_3_heavy_correction(query, answer, docs, retry_count)

        # FACTUAL_ERROR → 根据是否有相关文档决定中度或重度
        if AnswerErrorType.FACTUAL_ERROR in error_types:
            logger.info(
                "factual_error_detected",
                extra={"details": quality_result.details, "component": "correction_pipeline"}
            )
            return self._level_2_medium_correction(query, answer, docs, retry_count)

        # HALLUCINATION → 重度纠错
        if AnswerErrorType.HALLUCINATION in error_types:
            logger.info(
                "hallucination_detected_heavy_correction",
                extra={"details": quality_result.details, "component": "correction_pipeline"}
            )
            return self._level_3_heavy_correction(query, answer, docs, retry_count)

        # 兜底：默认轻度纠错
        logger.info(
            "default_light_correction",
            extra={"component": "correction_pipeline"}
        )
        return self._level_1_light_correction(query, answer, docs)

    def _reject(self, original_answer: str, reason: str) -> CorrectionResult:
        """直接拒答"""
        rejection_message = f"抱歉，系统无法准确回答您的问题。{reason}，建议联系 HR 部门（内线 8001）获取帮助。"
        return CorrectionResult(
            success=False,
            level=CorrectionLevel.REJECT,
            corrected_answer=rejection_message,
            original_answer=original_answer,
            remaining_errors=[reason],
            final_decision="reject",
        )

    def _level_1_light_correction(
        self,
        query: str,
        answer: str,
        docs: List[Dict[str, Any]],
    ) -> CorrectionResult:
        """
        第 1 级（轻度纠错）：仅改写，不重检索

        适用：答案意思对，但措辞啰嗦、不规范、格式乱
        """
        from prompts import answer_refine, format_docs
        from graph import llm_with_timeout

        docs_text = format_docs(docs)
        messages = answer_refine(query=query, answer=answer, docs_text=docs_text)

        try:
            response = llm_with_timeout(messages)
            corrected = response.content.strip()

            logger.info(
                "level_1_light_correction_done",
                extra={"original_len": len(answer), "corrected_len": len(corrected)}
            )

            return CorrectionResult(
                success=True,
                level=CorrectionLevel.LIGHT,
                corrected_answer=corrected,
                original_answer=answer,
                final_decision="output",
            )
        except Exception as e:
            logger.warning(
                "level_1_correction_failed",
                extra={"error": str(e)}
            )
            # 失败则返回原答案
            return CorrectionResult(
                success=True,
                level=CorrectionLevel.LIGHT,
                corrected_answer=answer,
                original_answer=answer,
                final_decision="output",
            )

    def _level_2_medium_correction(
        self,
        query: str,
        answer: str,
        docs: List[Dict[str, Any]],
        retry_count: int,
    ) -> CorrectionResult:
        """
        第 2 级（中度纠错）：重生成，不重检索

        适用：部分事实错、漏关键信息，但检索片段本身是对的
        最多重试 2 次
        """
        from prompts import strict_generate_lv2, format_docs
        from graph import llm_with_timeout

        docs_text = format_docs(docs)
        messages = strict_generate_lv2(query=query, docs_text=docs_text)

        try:
            response = llm_with_timeout(messages)
            corrected = response.content.strip()

            # 生成后再检测一次
            quality_result = self.quality_checker.check(query, corrected, docs)

            if quality_result.is_acceptable:
                logger.info(
                    "level_2_medium_correction_success",
                    extra={"retry_count": retry_count}
                )
                return CorrectionResult(
                    success=True,
                    level=CorrectionLevel.MEDIUM,
                    corrected_answer=corrected,
                    original_answer=answer,
                    final_decision="output",
                )
            else:
                # 仍未通过，继续升级
                logger.info(
                    "level_2_medium_correction_incomplete",
                    extra={"errors": quality_result.error_types}
                )
                # 如果重试次数未满，继续尝试
                if retry_count < 2:
                    return self._level_3_heavy_correction(query, answer, docs, retry_count + 1)
                else:
                    return self._reject(answer, "系统无法生成满意答案")

        except Exception as e:
            logger.warning(
                "level_2_correction_failed",
                extra={"error": str(e)}
            )
            if retry_count < 2:
                return self._level_3_heavy_correction(query, answer, docs, retry_count + 1)
            else:
                return self._reject(answer, "答案生成失败")

    def _level_3_heavy_correction(
        self,
        query: str,
        answer: str,
        docs: List[Dict[str, Any]],
        retry_count: int,
    ) -> CorrectionResult:
        """
        第 3 级（重度纠错）：查询重写 → 重新检索 → 重排 → 再生

        适用：检索结果无关、答案完全幻觉、无有效参考

        前置动作：Query Writing（查询重写/扩展）
        - 基于原始查询和之前的检索结果，对查询进行优化重写
        - 策略包括：查询扩展、查询分解、查询泛化、查询具体化
        - 提高二次检索的召回率
        """
        from prompts import query_rewrite, strict_generate_lv3, format_docs
        from graph import llm_with_timeout
        from pydantic import BaseModel
        from typing import List as TList

        logger.info(
            "level_3_heavy_correction_start",
            extra={"retry_count": retry_count, "query": query[:50]}
        )

        # ─────────────────────────────────────────────────────────────────
        # 前置动作：Query Writing（查询重写/扩展）
        # ─────────────────────────────────────────────────────────────────
        old_docs_text = format_docs(docs)
        rewritten_query = query  # 默认使用原始查询

        try:
            messages = query_rewrite(query=query, old_docs_text=old_docs_text)

            # 定义解析输出的 Pydantic 模型
            class QueryRewriteOutput(BaseModel):
                rewritten_queries: TList[dict]
                final_query: str

            result = llm_with_timeout.with_structured_output(
                messages, QueryRewriteOutput
            )
            rewritten_query = result.final_query

            logger.info(
                "query_rewrite_done",
                extra={
                    "original_query": query[:50],
                    "rewritten_query": rewritten_query[:50],
                    "candidate_count": len(result.rewritten_queries),
                }
            )
        except Exception as e:
            logger.warning(
                "query_rewrite_failed_fallback_to_original",
                extra={"error": str(e)}
            )
            # 查询重写失败时，保守使用原始查询继续流程

        # ─────────────────────────────────────────────────────────────────
        # Step 1: 重新检索（使用重写后的查询）
        # ─────────────────────────────────────────────────────────────────
        from rag import build_llama_rag_pipeline
        pipeline = build_llama_rag_pipeline(
            abac_depts=[self._user_dept] if self._user_dept else [],
            abac_projects=self._user_projects or [],
        )
        new_docs = pipeline.retrieve(query=rewritten_query, top_k=50, use_rerank=False)

        if not new_docs:
            logger.warning("level_3_no_new_docs")
            return self._reject(answer, "未找到相关信息")

        # ─────────────────────────────────────────────────────────────────
        # Step 2: BGE 重排（使用 LlamaIndex Reranker）
        # ─────────────────────────────────────────────────────────────────
        from reranker_llamaindex import get_llama_reranker
        reranker = get_llama_reranker()
        new_docs = reranker.rerank(query=rewritten_query, documents=new_docs, top_k=5)

        # ─────────────────────────────────────────────────────────────────
        # Step 3: 严格生成（使用重写后的查询）
        # ─────────────────────────────────────────────────────────────────
        docs_text = format_docs(new_docs)
        messages = strict_generate_lv3(query=rewritten_query, docs_text=docs_text)

        try:
            response = llm_with_timeout(messages)
            corrected = response.content.strip()

            # 最终检测
            quality_result = self.quality_checker.check(query, corrected, new_docs)

            if quality_result.is_acceptable:
                logger.info("level_3_heavy_correction_success")
                return CorrectionResult(
                    success=True,
                    level=CorrectionLevel.HEAVY,
                    corrected_answer=corrected,
                    original_answer=answer,
                    final_decision="output",
                )
            else:
                logger.info("level_3_heavy_correction_still_bad")
                return self._reject(answer, "系统无法准确回答您的问题")

        except Exception as e:
            logger.error("level_3_generation_failed", extra={"error": str(e)})
            return self._reject(answer, "答案生成失败")


# =============================================================================
# 第六部分：全局单例管理
# =============================================================================

# 全局变量：存储唯一的实例（避免重复创建）
_controlled_self_rag: Optional[ControlledSelfRAG] = None


def get_controlled_self_rag() -> ControlledSelfRAG:
    """
    获取可控 Self-RAG 单例
    
    单例模式：整个应用只创建一个实例，所有地方都用这个实例
    好处：节省资源、保持状态一致
    
    使用方式：
    >>> rag = get_controlled_self_rag()
    >>> result = rag.process("报销流程是什么？", "user123")
    """
    global _controlled_self_rag
    # 如果还没创建过 → 创建新实例
    if _controlled_self_rag is None:
        _controlled_self_rag = ControlledSelfRAG()
    # 返回已有的实例
    return _controlled_self_rag
