"""
知识库管理模块 (knowledge.py)

ABAC + Milvus 架构下的知识库管理：

1. 【黑话词典管理】：企业自定义术语的 CRUD
2. 【知识库管理】：文档的增删改查
3. 【检索结果管理】：置信度评估和阈值过滤

权限控制：
- 使用 ABAC 标签过滤（allowed_depts + allowed_projects）
- 文档切片和向量存储在 Milvus 向量数据库
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger("oa_agent.knowledge")

try:
    from pymilvus import MilvusClient
except ImportError:
    MilvusClient = None


# =============================================================================
# 数据模型
# =============================================================================

class RetrievalConfidence(str, Enum):
    """检索置信度等级。"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# 黑话词典管理
# =============================================================================

class SlangDictionary:
    """黑话词典管理类。"""

    def __init__(self):
        self._dict: Dict[str, str] = {}
        self._categories: Dict[str, List[str]] = {}
        self._confidence_threshold: float = 0.5
        self._last_updated: Optional[datetime] = None
        self._load_default_dict()

    def _load_default_dict(self):
        self._dict = {
            "年假": "年度带薪休假",
            "请假": "请假申请",
            "事假": "因私请假",
            "病假": "因病请假",
            "婚假": "结婚假期",
            "产假": "生育假期",
            "陪产假": "陪产假期",
            "调休": "加班调换休息日",
            "加班换班": "加班调换休息日申请",
            "换班": "班次调换",
            "门禁": "门禁权限管理",
            "工牌": "员工工牌",
            "饭卡": "员工餐卡充值",
            "密码重置": "密码重置申请",
            "开账号": "开通系统账号",
            "��权限": "权限开通申请",
            "报销": "费用报销申请",
            "差旅报销": "出差费用报销",
            "发票": "发票报销",
        }
        self._categories = {
            "hr": ["年假", "请假", "事假", "病假", "婚假", "产假", "陪产假", "调休", "加班换班", "换班"],
            "admin": ["门禁", "工牌", "饭卡"],
            "it": ["密码重置", "开账号", "开权限"],
            "finance": ["报销", "差旅报销", "发票"],
        }
        self._last_updated = datetime.now(timezone.utc)

    def get(self, slang: str) -> Optional[str]:
        return self._dict.get(slang)

    def get_all(self) -> Dict[str, str]:
        return self._dict.copy()

    def add(self, slang: str, standard: str, category: str = "general") -> bool:
        if not slang or not standard:
            return False
        self._dict[slang] = standard
        if category not in self._categories:
            self._categories[category] = []
        if slang not in self._categories[category]:
            self._categories[category].append(slang)
        self._last_updated = datetime.now(timezone.utc)
        logger.info(f"slang_added: {slang} -> {standard}")
        return True

    def remove(self, slang: str) -> bool:
        if slang in self._dict:
            del self._dict[slang]
            for cat_list in self._categories.values():
                if slang in cat_list:
                    cat_list.remove(slang)
            self._last_updated = datetime.now(timezone.utc)
            return True
        return False

    def update(self, slang: str, new_standard: str) -> bool:
        if slang in self._dict:
            self._dict[slang] = new_standard
            self._last_updated = datetime.now(timezone.utc)
            return True
        return False

    def set_confidence_threshold(self, threshold: float):
        if 0.0 <= threshold <= 1.0:
            self._confidence_threshold = threshold

    def get_confidence_level(self, score: float) -> RetrievalConfidence:
        if score >= 0.8:
            return RetrievalConfidence.HIGH
        elif score >= self._confidence_threshold:
            return RetrievalConfidence.MEDIUM
        else:
            return RetrievalConfidence.LOW

    def preprocess(self, query: str) -> str:
        import re
        sorted_keys = sorted(self._dict.keys(), key=len, reverse=True)
        result = query
        for slang in sorted_keys:
            pattern = re.compile(re.escape(slang), re.IGNORECASE)
            result = pattern.sub(self._dict[slang], result)
        return result


_slang_dict: Optional[SlangDictionary] = None


def get_slang_dict() -> SlangDictionary:
    global _slang_dict
    if _slang_dict is None:
        _slang_dict = SlangDictionary()
    return _slang_dict


# =============================================================================
# ABAC 知识库管理
# =============================================================================

class KnowledgeBase:
    """
    知识库管理类（ABAC + Milvus 架构）。

    功能：
    - 文档的增删改查（Milvus）
    - ABAC 权限标签过滤（allowed_depts + allowed_projects）
    - 检索结果置信度过滤
    """

    def __init__(self):
        self._collection_name = "oa_chunks"
        self._initialized = False
        self._milvus_client = None

    def initialize(self) -> bool:
        """初始化 Milvus 连接（嵌入式 milvus-lite）。"""
        try:
            from config import settings
            if MilvusClient is not None:
                self._milvus_client = MilvusClient(uri=settings.milvus_db_path)
            self._initialized = True
            logger.info("knowledge_base_initialized")
            return True
        except Exception as e:
            logger.warning(f"knowledge_base_init_failed: {e}")
            return False

    def add_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector: Optional[List[float]] = None,
        allowed_depts: Optional[List[str]] = None,
        allowed_projects: Optional[List[str]] = None,
    ) -> bool:
        """
        添加文档切片到知识库。

        参数：
        - chunk_id: 切片唯一标识
        - text: 切片文本内容
        - metadata: 元数据
        - vector: 向量（可选）
        - allowed_depts: 允许访问的部门列表
        - allowed_projects: 允许访问的项目列表
        """
        if not self._initialized:
            return False

        try:
            from config import settings
            from langchain_ollama import OllamaEmbeddings

            if vector is None:
                embed_model = OllamaEmbeddings(
                    model=settings.embedding_model,
                    base_url=settings.embedding_base_url,
                )
                vector = embed_model.embed_query(text)

            if self._milvus_client:
                self._milvus_client.insert(
                    collection_name=self._collection_name,
                    data=[{
                        "chunk_id": chunk_id,
                        "text": text,
                        "metadata": metadata or {},
                        "embedding": vector,
                        "allowed_depts": allowed_depts or [],
                        "allowed_projects": allowed_projects or [],
                    }],
                )

            logger.info(f"chunk_added: {chunk_id}")
            return True

        except Exception as e:
            logger.error(f"chunk_add_failed: {e}, chunk_id={chunk_id}")
            return False

    def delete_chunk(self, chunk_id: str) -> bool:
        """删除指定切片。"""
        if not self._initialized:
            return False

        try:
            if self._milvus_client:
                self._milvus_client.delete(
                    collection_name=self._collection_name,
                    filter=f'chunk_id == "{chunk_id}"',
                )
            logger.info(f"chunk_deleted: {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"chunk_delete_failed: {e}")
            return False

    def batch_add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        doc_id: str,
        allowed_depts: Optional[List[str]] = None,
        allowed_projects: Optional[List[str]] = None,
    ) -> int:
        """
        批量添加切片（带 ABAC 标签）。

        参数：
        - chunks: [{"chunk_id": "C1", "text": "...", "metadata": {...}}, ...]
        - doc_id: 所属制度 ID
        - allowed_depts: 允许访问的部门列表
        - allowed_projects: 允许访问的项目列表

        返回：成功添加的切片数量
        """
        success_count = 0

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            chunk_allowed_depts = chunk.get("allowed_depts", allowed_depts or [])
            chunk_allowed_projects = chunk.get("allowed_projects", allowed_projects or [])

            if self.add_chunk(chunk_id, text, metadata,
                            allowed_depts=chunk_allowed_depts,
                            allowed_projects=chunk_allowed_projects):
                success_count += 1

        logger.info(f"chunks_batch_added: {success_count}/{len(chunks)}")
        return success_count

    def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        检索文档（ABAC-RAG）。

        参数：
        - query: 用户问题
        - user_id: 用户 ID（用于权限过滤）
        - top_k: 返回数量
        - threshold: 置信度阈值

        返回：包含置信度等级的检索结果
        """
        slang_dict = get_slang_dict()
        processed_query = slang_dict.preprocess(query)
        
        # 统一使用 HybridRetriever 进行检索（包含向量检索、BM25 检索和 RRF 融合）
        retriever = build_abac_filtered_retriever()
        docs = retriever.retrieve(
            query=processed_query,
            user_id=user_id,
            top_k=top_k,
        )

        filtered_docs = []
        for doc in docs:
            score = doc.get("score", 0.0)
            confidence = slang_dict.get_confidence_level(score)
            doc["confidence"] = confidence.value
            doc["confidence_score"] = score
            if score >= threshold:
                filtered_docs.append(doc)

        return filtered_docs

    def close(self):
        """关闭连接。"""
        self._initialized = False


_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
        _knowledge_base.initialize()
    return _knowledge_base
