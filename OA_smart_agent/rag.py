"""
RAG 检索模块 (rag.py - LlamaIndex 版本)

使用 LlamaIndex 框架实现的混合检索管道：
1. Dense 向量检索 - 通过 MilvusIndexManager + ABAC 过滤
2. Sparse 向量检索 - Milvus 2.4+ SparseFloatVector + BM25
3. RRF 融合 - Reciprocal Rank Fusion 多路结果排序
4. Rerank - BGE-Reranker 语义重排序

依赖：
    llama-index-core
    llama-index-vector-stores-milvus
    llama-index-postprocessor-cohere-rerank
    llama-index-embeddings-ollama
    pymilvus>=2.4.0
"""

import logging
import time
from typing import Any, Dict, List, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerReranker

logger = logging.getLogger("oa_agent.rag")


class HybridSearchConfig:
    """混合检索配置"""

    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
        vector_top_k: int = 100,
        bm25_top_k: int = 100,
        final_top_k: int = 10,
        min_keyword_length: int = 2,
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        self.min_keyword_length = min_keyword_length
class LlamaRAGPipeline:
    """
    使用 LlamaIndex 编排的 RAG 检索链路。
    组件：Dense 向量检索 + Sparse（BM25）检索 + RRF 融合 + Rerank 精排

    Sparse 检索基于 Milvus 2.4+ 的 SparseFloatVector + BM25，无需独立 BM25Retriever。
    """

    def __init__(
        self,
        vector_index: Optional[VectorStoreIndex] = None,
        hybrid_config: Optional[HybridSearchConfig] = None,
        reranker_model: str = "BAAI/bge-reranker-large",
        abac_depts: Optional[List[str]] = None,
        abac_projects: Optional[List[str]] = None,
        vector_retriever: Optional[Any] = None,
    ):
        self.hybrid_config = hybrid_config or HybridSearchConfig()
        self.reranker_model = reranker_model
        self._vector_index = vector_index
        self._abac_depts = abac_depts or []
        self._abac_projects = abac_projects or []
        self._external_retriever = vector_retriever

    def _ensure_embed_model(self):
        """确保 Settings.embed_model 已初始化"""
        if Settings.embed_model is not None:
            return
        from config import settings as app_settings
        provider = getattr(app_settings, 'embedding_provider', 'ollama')
        model = getattr(app_settings, 'embedding_model', '')
        base_url = getattr(app_settings, 'embedding_base_url', 'http://localhost:11434')

        if provider in ('openai', 'bge'):
            try:
                from llama_index.embeddings.openai import OpenAIEmbedding
                api_key = getattr(app_settings, 'embedding_api_key', 'not-needed')
                Settings.embed_model = OpenAIEmbedding(model=model, api_key=api_key, base_url=base_url)
                logger.info("LlamaRAGPipeline: 已初始化 OpenAIEmbedding (BGE-M3)")
            except ImportError:
                logger.warning("llama_index.embeddings.openai 未安装")
        elif provider == 'ollama':
            try:
                from llama_index.embeddings.ollama import OllamaEmbedding
                Settings.embed_model = OllamaEmbedding(model=model, base_url=base_url)
                logger.info("LlamaRAGPipeline: 已初始化 OllamaEmbedding")
            except Exception as e:
                logger.warning(f"Ollama Embedding 初始化失败: {e}")

    def _get_vector_index(self) -> VectorStoreIndex:
        """获取 VectorStoreIndex（懒加载）"""
        if self._vector_index is not None:
            return self._vector_index
        from data_ingestion import get_milvus_manager
        manager = get_milvus_manager()
        self._vector_index = manager._get_index()
        return self._vector_index

    def _sparse_retrieve(self, query: str) -> List[Any]:
        """
        执行 Sparse Vector 检索（Milvus 2.4+ BM25 风格）。

        通过 MilvusVectorStore 的 sparse 模式实现关键词匹配，
        使用 jieba 分词（TokenizerEnum.JIEBA）进行中文分词。
        """
        from data_ingestion import get_milvus_manager
        manager = get_milvus_manager()
        try:
            retriever = manager.as_retriever(
                similarity_top_k=self.hybrid_config.bm25_top_k,
                abac_depts=self._abac_depts,
                abac_projects=self._abac_projects,
                use_sparse=True,
            )
            results = retriever.retrieve(query)
            logger.info(f"Sparse 检索: '{query[:30]}...' 返回 {len(results)} 条")
            return results
        except Exception as e:
            logger.error(f"Sparse 检索失败: {e}")
            return []

    def _vector_retrieve(self, query: str) -> List[Any]:
        """执行 Dense 向量检索（带 ABAC 过滤）"""
        from data_ingestion import get_milvus_manager
        manager = get_milvus_manager()
        try:
            retriever = manager.as_retriever(
                similarity_top_k=self.hybrid_config.vector_top_k,
                abac_depts=self._abac_depts,
                abac_projects=self._abac_projects,
                use_sparse=False,
            )
            results = retriever.retrieve(query)
            logger.info(f"Dense 检索: '{query[:30]}...' 返回 {len(results)} 条")
            return results
        except Exception as e:
            logger.error(f"Dense 检索失败: {e}")
            return []

    @staticmethod
    def _rrf_fusion(ranked_lists: List[List[Any]], k: int = 60) -> List[Any]:
        """RRF 融合"""
        scores: Dict[str, Any] = {}
        node_map: Dict[str, Any] = {}
        for ranked_list in ranked_lists:
            for rank, node_with_score in enumerate(ranked_list):
                node_id = node_with_score.node.node_id
                if node_id not in node_map:
                    node_map[node_id] = node_with_score
                if node_id not in scores:
                    scores[node_id] = 0.0
                scores[node_id] += 1.0 / (k + rank + 1)
        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        fused = []
        for node_id in sorted_ids:
            nws = node_map[node_id].copy()
            nws.score = scores[node_id]
            fused.append(nws)
        return fused

    def _rerank(self, nodes: List[Any], top_k: int) -> List[Any]:
        """使用 bge-reranker-large 对节点重排序"""
        if not nodes:
            return []
        postproc = SentenceTransformerReranker(model=self.reranker_model, top_n=top_k)
        try:
            reranked = postproc.postprocess_nodes(nodes, query="")
            logger.info(f"Rerank: {len(nodes)} → {len(reranked)}")
            return reranked
        except Exception as e:
            logger.error(f"Rerank 失败: {e}")
            return nodes[:top_k]

    def retrieve(
        self,
        query: str,
        user_id: str = "",
        top_k: Optional[int] = None,
        use_rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """完整的 RAG 检索链路"""
        from metrics import record_rag_retrieval

        final_k = top_k or self.hybrid_config.final_top_k
        start_time = time.perf_counter()
        success = True
        doc_count = 0

        try:
            dense_results = self._vector_retrieve(query)
            sparse_results = self._sparse_retrieve(query)

            if dense_results and sparse_results:
                all_nodes = self._rrf_fusion([dense_results, sparse_results], k=self.hybrid_config.rrf_k)
            elif dense_results:
                all_nodes = dense_results
            elif sparse_results:
                all_nodes = sparse_results
            else:
                logger.warning(f"检索无结果: '{query[:30]}...'")
                all_nodes = []

            max_candidates = max(self.hybrid_config.vector_top_k, self.hybrid_config.bm25_top_k)
            candidates = all_nodes[:max_candidates]

            if use_rerank:
                candidates = self._rerank(candidates, top_k=final_k)
            else:
                candidates = candidates[:final_k]

            doc_count = len(candidates)

            return [
                {
                    "node_id": nws.node.node_id,
                    "text": nws.node.get_content(),
                    "score": nws.score,
                    "metadata": nws.node.metadata or {},
                    "chunk_id": nws.node.metadata.get("chunk_id", nws.node.node_id),
                }
                for nws in candidates
            ]
        except Exception as e:
            success = False
            raise
        finally:
            duration = time.perf_counter() - start_time
            record_rag_retrieval(duration=duration, doc_count=doc_count, success=success)


# =============================================================================
# 工厂函数
# =============================================================================

_pipeline_cache: Dict[str, LlamaRAGPipeline] = {}


def _get_cache_key(
    abac_depts: Optional[List[str]] = None,
    abac_projects: Optional[List[str]] = None,
) -> str:
    """生成缓存 key（基于 ABAC 参数）"""
    depts = tuple(sorted(abac_depts or []))
    projects = tuple(sorted(abac_projects or []))
    return f"depts={depts}|projects={projects}"


def build_llama_rag_pipeline(
    hybrid_config: Optional[HybridSearchConfig] = None,
    abac_depts: Optional[List[str]] = None,
    abac_projects: Optional[List[str]] = None,
) -> LlamaRAGPipeline:
    """构建 LlamaIndex RAG Pipeline（基于 ABAC 参数缓存）"""
    cache_key = _get_cache_key(abac_depts, abac_projects)

    if cache_key not in _pipeline_cache:
        _pipeline_cache[cache_key] = LlamaRAGPipeline(
            hybrid_config=hybrid_config,
            abac_depts=abac_depts,
            abac_projects=abac_projects,
        )
        logger.info(f"创建 RAG Pipeline 实例 (key={cache_key})")

    return _pipeline_cache[cache_key]


def llama_retrieve(
    query: str,
    user_id: str = "",
    top_k: int = 10,
    use_rerank: bool = True,
    abac_depts: Optional[List[str]] = None,
    abac_projects: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """便捷函数：直接执行 RAG 检索"""
    pipeline = build_llama_rag_pipeline(abac_depts=abac_depts, abac_projects=abac_projects)
    return pipeline.retrieve(query, user_id=user_id, top_k=top_k, use_rerank=use_rerank)


def build_graph_filtered_retriever(
    abac_depts: Optional[List[str]] = None,
    abac_projects: Optional[List[str]] = None,
) -> LlamaRAGPipeline:
    """向后兼容：构建 RAG Pipeline（带 ABAC 过滤）"""
    from config import settings
    config = HybridSearchConfig(
        dense_weight=settings.hybrid_dense_weight,
        sparse_weight=settings.hybrid_sparse_weight,
        rrf_k=settings.hybrid_rrf_k,
        vector_top_k=settings.hybrid_vector_top_k,
        bm25_top_k=settings.hybrid_bm25_top_k,
        final_top_k=settings.hybrid_final_top_k,
    )
    return build_llama_rag_pipeline(
        hybrid_config=config,
        abac_depts=abac_depts,
        abac_projects=abac_projects,
    )


def build_abac_filtered_retriever(
    abac_depts: Optional[List[str]] = None,
    abac_projects: Optional[List[str]] = None,
) -> LlamaRAGPipeline:
    return build_graph_filtered_retriever(abac_depts, abac_projects)


def build_hybrid_retriever(
    abac_depts: Optional[List[str]] = None,
    abac_projects: Optional[List[str]] = None,
) -> LlamaRAGPipeline:
    return build_graph_filtered_retriever(abac_depts, abac_projects)