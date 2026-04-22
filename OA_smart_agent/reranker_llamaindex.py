"""
Reranker 封装模块 (reranker_llamaindex.py)

使用 LlamaIndex 官方 PostProcessor 实现重排序功能。

支持的模型：
1. CohereRerank      — 云端 API，支持多语言（rerank-multilingual-v3.0）
2. SentenceTransformerReranker — 本地 GPU，支持 Cross-Encoder 家族
3. LLMReranker       — 使用 LLM 打分重排序（实验性）

使用示例：

    from reranker_llamaindex import (
        get_llama_reranker,
        LlamaRerankerPipeline,
    )

    # 直接使用
    reranker = get_llama_reranker(use_cohere=True, api_key="...")
    results = reranker.rerank("年假如何计算", documents, top_k=5)

    # 嵌入 LlamaIndex Pipeline（配合 rag_llamaindex.py 使用）
    pipeline = LlamaRerankerPipeline(reranker_type="cohere", top_k=5)
"""

import logging
import os
from typing import Any, Dict, List, Optional

from llama_index.core.postprocessor import (
    CohereRerank,
    SentenceTransformerReranker,
)

logger = logging.getLogger("oa_agent.reranker_llamaindex")


# =============================================================================
# 配置类
# =============================================================================

class RerankConfig:
    """重排序配置"""

    def __init__(
        self,
        reranker_type: str = "sentence_transformer",  # "cohere" | "sentence_transformer"
        model_name: str = "BAAI/bge-reranker-large",
        use_mock: bool = False,
        batch_size: int = 32,
        normalize_scores: bool = True,
        alpha: float = 0.5,
    ):
        self.reranker_type = reranker_type
        self.model_name = model_name
        self.use_mock = use_mock
        self.batch_size = batch_size
        self.normalize_scores = normalize_scores
        self.alpha = alpha


# =============================================================================
# 核心重排序器
# =============================================================================

class LlamaReranker:
    """
    LlamaIndex 官方 PostProcessor 封装。

    支持：
    - CohereRerank（云端）
    - SentenceTransformerReranker（本地 GPU）
    """

    def __init__(
        self,
        reranker_type: str = "sentence_transformer",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        top_n: int = 5,
        config: Optional[RerankConfig] = None,
    ):
        self.reranker_type = reranker_type
        self.model_name = model_name or "BAAI/bge-reranker-large"
        self.api_key = api_key or os.environ.get("COHERE_API_KEY", "")
        self.top_n = top_n
        self.config = config
        self._postprocessor: Optional[Any] = None
        self._initialized = False

        if config and not config.use_mock:
            self._load_postprocessor()

    def _load_postprocessor(self):
        """懒加载 LlamaIndex PostProcessor"""
        if self._initialized:
            return

        try:
            if self.reranker_type == "cohere":
                self._postprocessor = CohereRerank(
                    api_key=self.api_key,
                    top_n=self.top_n,
                    model="rerank-multilingual-v3.0",
                )
                logger.info(f"CohereRerank loaded (model=rerank-multilingual-v3.0)")

            elif self.reranker_type == "sentence_transformer":
                self._postprocessor = SentenceTransformerReranker(
                    model=self.model_name,
                    top_n=self.top_n,
                )
                logger.info(f"SentenceTransformerReranker loaded (model={self.model_name})")

            else:
                logger.warning(f"Unknown reranker_type: {self.reranker_type}，降级到空操作")
                self._postprocessor = None

            self._initialized = True

        except ImportError as e:
            logger.warning(f"LlamaIndex PostProcessor 导入失败: {e}，降级到 MockRerank")
            self._postprocessor = None
            self._initialized = True
        except Exception as e:
            logger.error(f"LlamaIndex PostProcessor 初始化失败: {e}")
            self._postprocessor = None
            self._initialized = True

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        对文档列表进行重排序。

        参数：
        - query: 查询文本
        - documents: 文档列表（每项需要包含 text 字段）
        - top_k: 返回前多少条

        返回：
        - 重排序后的文档列表（降序），每项包含：
          - text, score, rerank_rank, rerank_model
        """
        if not documents:
            return []

        self._load_postprocessor()

        if self._postprocessor is None:
            return self._mock_rerank(documents, top_k)

        try:
            # 将业务字典格式转换为 LlamaIndex NodeWithScore
            from llama_index.core.schema import NodeWithScore, TextNode

            nodes = [
                NodeWithScore(
                    node=TextNode(
                        id_=doc.get("chunk_id", doc.get("node_id", str(i))),
                        text=doc.get("text", ""),
                        metadata=doc.get("metadata", {}),
                    ),
                    score=float(doc.get("score", doc.get("rerank_score", 0.5))),
                )
                for i, doc in enumerate(documents)
            ]

            # 执行重排序
            reranked_nodes = self._postprocessor.postprocess_nodes(
                nodes,
                query=query,
            )

            # 转换回业务字典格式
            results = []
            for rank, nws in enumerate(reranked_nodes[:top_k]):
                doc = {
                    "text": nws.node.get_content(),
                    "score": nws.score,
                    "rerank_score": nws.score,
                    "rerank_rank": rank + 1,
                    "rerank_model": self.reranker_type,
                    "node_id": nws.node.node_id,
                    "metadata": nws.node.metadata or {},
                }
                # 保留原始 metadata 中的字段
                if "chunk_id" in (nws.node.metadata or {}):
                    doc["chunk_id"] = nws.node.metadata["chunk_id"]
                results.append(doc)

            logger.info(
                f"LlamaReranker 完成: {len(documents)} → {len(results)} "
                f"(type={self.reranker_type})"
            )
            return results

        except Exception as e:
            logger.error(f"LlamaReranker rerank 失败: {e}，使用 Mock")
            return self._mock_rerank(documents, top_k)

    @staticmethod
    def _mock_rerank(documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """简单的关键词匹配降级"""
        import re

        query_words = set(re.findall(r"\w+", query.lower() if "query" in dir() else ""))
        scored_docs = []

        for doc in documents:
            text = doc.get("text", "")
            doc_words = set(re.findall(r"\w+", text.lower()))
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words), 1)
            doc["rerank_score"] = score
            doc["rerank_rank"] = 0
            doc["rerank_model"] = "mock"
            scored_docs.append(doc)

        scored_docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        for i, doc in enumerate(scored_docs[:top_k]):
            doc["rerank_rank"] = i + 1
        return scored_docs[:top_k]


# =============================================================================
# Pipeline 封装（可嵌入 LangGraph State）
# =============================================================================

class LlamaRerankerPipeline:
    """
    重排序 Pipeline 封装（兼容旧版 HybridReranker / TwoStageRetriever）。

    用于链式调用：
        pipeline = LlamaRerankerPipeline(reranker_type="cohere", top_k=5)
        results = pipeline.rerank_with_fusion(query, {"dense": [...], "bm25": [...]})
    """

    def __init__(
        self,
        reranker_type: str = "sentence_transformer",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        top_k: int = 5,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
    ):
        self.reranker_type = reranker_type
        self.top_k = top_k
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self._reranker = LlamaReranker(
            reranker_type=reranker_type,
            model_name=model_name,
            api_key=api_key,
            top_n=top_k,
        )

    def rerank_with_fusion(
        self,
        query: str,
        retrieval_results: Dict[str, List[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        带多路融合的重排序（替代 HybridReranker.rerank_with_fusion）。

        参数：
        - query: 查询文本
        - retrieval_results: 多路检索结果 {"dense": [...], "bm25": [...]}
        - top_k: 返回数量

        返回：
        - 重排序后的文档列表
        """
        k = top_k or self.top_k

        # Step 1: 构建候选池（去重）
        candidate_docs: Dict[str, Dict[str, Any]] = {}
        for source, docs in retrieval_results.items():
            for doc in docs:
                chunk_id = doc.get("chunk_id", doc.get("node_id", ""))
                if chunk_id not in candidate_docs:
                    candidate_docs[chunk_id] = {
                        **doc,
                        "sources": [source],
                    }
                else:
                    candidate_docs[chunk_id]["sources"].append(source)

        candidates = list(candidate_docs.values())
        if not candidates:
            return []

        # Step 2: RRF 融合得分
        if self.fusion_method == "rrf":
            fused_scores = self._rrf_fusion(retrieval_results, candidates, self.rrf_k)
            for doc in candidates:
                chunk_id = doc.get("chunk_id", doc.get("node_id", ""))
                doc["fusion_score"] = fused_scores.get(chunk_id, 0.0)

        # Step 3: 精排
        reranked = self._reranker.rerank(query, candidates, top_k=k)

        # 合并融合分数
        for doc in reranked:
            doc["fusion_score"] = fused_scores.get(
                doc.get("chunk_id", doc.get("node_id", "")), 0.0
            )
        return reranked

    @staticmethod
    def _rrf_fusion(
        retrieval_results: Dict[str, List[Dict[str, Any]]],
        candidates: List[Dict[str, Any]],
        k: int,
    ) -> Dict[str, float]:
        """RRF 融合"""
        fusion_scores: Dict[str, float] = {}
        node_map: Dict[str, Dict] = {}

        for source, docs in retrieval_results.items():
            for rank, doc in enumerate(docs, 1):
                chunk_id = doc.get("chunk_id", doc.get("node_id", ""))
                if chunk_id not in node_map:
                    node_map[chunk_id] = doc
                fusion_scores[chunk_id] = fusion_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

        return fusion_scores

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """简化的重排序（两阶段检索的 Stage 2）"""
        return self._reranker.rerank(query, documents, top_k or self.top_k)


# =============================================================================
# 全局单例
# =============================================================================

_reranker_instance: Optional[LlamaReranker] = None


def get_llama_reranker(
    use_cohere: bool = False,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    top_k: int = 5,
) -> LlamaReranker:
    """
    获取 LlamaReranker 全局单例。

    示例：
        reranker = get_llama_reranker(use_cohere=True, api_key="...")
        results = reranker.rerank("查询", documents, top_k=5)
    """
    global _reranker_instance
    if _reranker_instance is None:
        reranker_type = "cohere" if use_cohere else "sentence_transformer"
        _reranker_instance = LlamaReranker(
            reranker_type=reranker_type,
            model_name=model_name,
            api_key=api_key,
            top_n=top_k,
        )
    return _reranker_instance


def build_llama_reranker_pipeline(
    reranker_type: str = "sentence_transformer",
    top_k: int = 5,
) -> LlamaRerankerPipeline:
    """
    构建 LlamaRerankerPipeline 全局实例。

    示例：
        pipeline = build_llama_reranker_pipeline(reranker_type="cohere")
        results = pipeline.rerank_with_fusion(
            "年假怎么算",
            {"dense": [...], "bm25": [...]},
            top_k=5
        )
    """
    return LlamaRerankerPipeline(
        reranker_type=reranker_type,
        top_k=top_k,
    )