"""
向量数据库管家模块 (index_manager.py)  —  LlamaIndex 重构版

核心变化：
- 底层从直接调用 pymilvus MilvusClient → LlamaIndex MilvusVectorStore
- 索引管理从手动 schema/index 操作 → StorageContext + VectorStoreIndex
- Embedding 生成从手动调用 → LlamaIndex Settings.embed_model（全局统一）
- 保持与旧版相同的公开接口：
    MilvusIndexManager.create_collection()
    MilvusIndexManager.delete_collection()
    MilvusIndexManager.insert_chunks(chunks)
    MilvusIndexManager.delete_chunks(chunk_ids)
    MilvusIndexManager.delete_chunks_by_doc_id(doc_id)
    MilvusIndexManager.get_stats()
    get_milvus_manager()

ABAC 权限字段（allowed_depts / allowed_projects）存放在
MilvusVectorStore 的 metadata 中，通过 MetadataFilters 过滤。

依赖：
    pip install llama-index-vector-stores-milvus
    pip install llama-index-embeddings-openai  （或其他 embed 集成）
"""

import logging
from typing import Any, Dict, List, Optional

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from .models import Chunk

logger = logging.getLogger("oa_agent.index_manager")


class MilvusIndexManager:
    """
    Milvus 索引管理器（LlamaIndex MilvusVectorStore 封装）

    职责：
    ✓ 创建 / 删除 Collection
    ✓ 批量插入文档切片（Chunk → TextNode）
    ✓ 按 chunk_id / doc_id 删除切片
    ✓ 获取统计信息
    ✓ ABAC 权限标签存储（metadata 中的 allowed_depts / allowed_projects）

    实现说明：
    - MilvusVectorStore 底层自动管理 schema 和 HNSW 索引
    - Embedding 使用全局 Settings.embed_model（调用方负责初始化）
    - allowed_depts / allowed_projects 以 JSON 字符串形式存入 metadata，
      检索时通过 MetadataFilters 做精确过滤
    """

    def __init__(
        self,
        db_path: str = "./data/milvus_lite.db",
        collection_name: str = "oa_chunks",
        dimension: Optional[int] = None,
    ):
        from config import settings as app_settings
        self.db_path = db_path
        self.collection_name = collection_name
        self.dimension = dimension if dimension is not None else app_settings.milvus_dimension

        self._vector_store: Optional[Any] = None   # MilvusVectorStore
        self._index: Optional[VectorStoreIndex] = None

    # ── 初始化 embed_model ────────────────────────────────────────────────────

    def _ensure_embed_model(self) -> None:
        """
        确保 Settings.embed_model 已初始化。

        若调用方没有提前设置，则尝试从 config.settings 自动初始化 Embedding 模型。
        支持多种 provider：ollama、openai、bge（通过 OpenAI-compatible API）。
        """
        if Settings.embed_model is not None:
            return
        try:
            from config import settings as app_settings
            provider = app_settings.embedding_provider.lower()
            
            if provider in ("ollama", "bge"):
                from llama_index.embeddings.ollama import OllamaEmbedding
                Settings.embed_model = OllamaEmbedding(
                    model=app_settings.embedding_model,
                    base_url=app_settings.embedding_base_url,
                    api_key=app_settings.embedding_api_key if app_settings.embedding_api_key != "not-needed" else None,
                )
                logger.info(f"已自动初始化 OllamaEmbedding (provider={provider})")
            elif provider == "openai":
                from llama_index.embeddings.openai import OpenAIEmbedding
                Settings.embed_model = OpenAIEmbedding(
                    model=app_settings.embedding_model,
                    api_key=app_settings.embedding_api_key,
                    base_url=app_settings.embedding_base_url,
                )
                logger.info("已自动初始化 OpenAIEmbedding")
            else:
                logger.warning(f"不支持的 embedding_provider: {provider}，将使用随机向量（仅测试）")
        except Exception as e:
            logger.warning(f"Embedding 初始化失败，将使用随机向量（仅测试）: {e}")

    # ── 获取 / 创建 VectorStore ───────────────────────────────────────────────

    def _get_vector_store(self, overwrite: bool = False) -> Any:
        """
        懒加载并返回 MilvusVectorStore 实例。

        参数：
        - overwrite: 若为 True，则先删除再重建 Collection
        """
        if self._vector_store is None or overwrite:
            try:
                from llama_index.vector_stores.milvus import MilvusVectorStore
            except ImportError as e:
                logger.error(
                    "llama-index-vector-stores-milvus 未安装: "
                    "pip install llama-index-vector-stores-milvus"
                )
                raise

            self._vector_store = MilvusVectorStore(
                uri=self.db_path,
                collection_name=self.collection_name,
                dim=self.dimension,
                overwrite=overwrite,
                # 开启 metadata 过滤支持（ABAC）
                enable_sparse=False,
            )
            self._index = None  # 重建时清空缓存的 index
        return self._vector_store

    def _get_index(self) -> VectorStoreIndex:
        """获取（或懒建）VectorStoreIndex"""
        if self._index is None:
            self._ensure_embed_model()
            vector_store = self._get_vector_store()
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            self._index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                show_progress=False,
            )
        return self._index

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def create_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        创建（或复用）Milvus Collection。

        LlamaIndex MilvusVectorStore 在首次写入时自动建 Collection，
        此方法通过 _get_vector_store() 触发连接即可。
        传入 collection_name 时，创建指定 collection 的临时 VectorStore。
        """
        try:
            if collection_name and collection_name != self.collection_name:
                from llama_index.vector_stores.milvus import MilvusVectorStore
                tmp = MilvusVectorStore(
                    uri=self.db_path,
                    collection_name=collection_name,
                    dim=self.dimension,
                    overwrite=False,
                )
                logger.info(f"Collection {collection_name} 已就绪（LlamaIndex 管理）")
                return True

            self._get_vector_store()
            logger.info(f"Collection {self.collection_name} 已就绪（LlamaIndex 管理）")
            return True
        except Exception as e:
            logger.error(f"创建 Collection 失败: {e}")
            return False

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        删除 Collection。

        通过 overwrite=True 重建同名 Collection 实现"清空"语义；
        若要彻底删除，直接调用底层 pymilvus client。
        """
        name = collection_name or self.collection_name
        try:
            from pymilvus import MilvusClient
            client = MilvusClient(uri=self.db_path)
            if client.has_collection(name):
                client.drop_collection(collection_name=name)
                logger.info(f"已删除 Collection {name}")
            # 清空缓存
            if name == self.collection_name:
                self._vector_store = None
                self._index = None
            return True
        except Exception as e:
            logger.error(f"删除 Collection 失败: {e}")
            return False

    def insert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> int:
        """
        批量插入文档切片到 Milvus。

        参数 chunks 格式（与旧版一致）：
        [
            {
                "chunk_id": "D-...-C-0001",
                "text": "...",
                "allowed_depts": ["研发部"],
                "allowed_projects": [],
                "metadata": {...},
            },
            ...
        ]

        ABAC 字段（allowed_depts / allowed_projects）被写入
        TextNode.metadata，支持后续 MetadataFilters 过滤。
        """
        if not chunks:
            return 0

        self._ensure_embed_model()

        nodes: List[TextNode] = []
        for item in chunks:
            meta: Dict[str, Any] = item.get("metadata") or {}
            meta["allowed_depts"] = item.get("allowed_depts", [])
            meta["allowed_projects"] = item.get("allowed_projects", [])
            meta["chunk_id"] = item.get("chunk_id", "")
            meta["doc_id"] = item.get("doc_id", "")

            node = TextNode(
                id_=item.get("chunk_id", ""),
                text=item.get("text", ""),
                metadata=meta,
            )
            nodes.append(node)

        try:
            if collection_name and collection_name != self.collection_name:
                from llama_index.vector_stores.milvus import MilvusVectorStore
                tmp_store = MilvusVectorStore(
                    uri=self.db_path,
                    collection_name=collection_name,
                    dim=self.dimension,
                    overwrite=False,
                )
                tmp_ctx = StorageContext.from_defaults(vector_store=tmp_store)
                VectorStoreIndex(nodes=nodes, storage_context=tmp_ctx, show_progress=False)
            else:
                index = self._get_index()
                index.insert_nodes(nodes)

            logger.info(f"成功插入 {len(nodes)} 个切片到 {collection_name or self.collection_name}")
            return len(nodes)
        except Exception as e:
            logger.error(f"插入切片失败: {e}")
            return 0

    def insert_chunk_objects(
        self,
        chunks: List[Chunk],
        collection_name: Optional[str] = None,
    ) -> int:
        """
        直接插入 Chunk 业务对象（比 insert_chunks 更便捷）。

        Chunk.to_text_node() 自动携带所有元数据。
        """
        self._ensure_embed_model()
        nodes = [c.to_text_node() for c in chunks]
        try:
            if collection_name and collection_name != self.collection_name:
                from llama_index.vector_stores.milvus import MilvusVectorStore
                tmp_store = MilvusVectorStore(
                    uri=self.db_path,
                    collection_name=collection_name,
                    dim=self.dimension,
                    overwrite=False,
                )
                tmp_ctx = StorageContext.from_defaults(vector_store=tmp_store)
                VectorStoreIndex(nodes=nodes, storage_context=tmp_ctx, show_progress=False)
            else:
                self._get_index().insert_nodes(nodes)

            logger.info(f"成功插入 {len(nodes)} 个 Chunk 对象")
            return len(nodes)
        except Exception as e:
            logger.error(f"插入 Chunk 对象失败: {e}")
            return 0

    def upsert_chunk_objects(
        self,
        chunks: List[Chunk],
        collection_name: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        增量更新（Upsert）：先查后插或先删后插。

        工作流程：
        1. 根据 doc_version / file_hash 判断是否需要更新
        2. 如果版本不变，跳过（已存在则不重复插入）
        3. 如果版本变更，删除旧切片，再插入新切片
        4. 如果是新文档，直接插入

        返回：{"inserted": N, "updated": M, "skipped": K, "deleted": D}
        """
        from pymilvus import MilvusClient

        name = collection_name or self.collection_name
        stats = {"inserted": 0, "updated": 0, "skipped": 0, "deleted": 0}

        try:
            client = MilvusClient(uri=self.db_path)
            if not client.has_collection(name):
                return self._do_insert_chunks(chunks, name, stats)

            for chunk in chunks:
                doc_id = chunk.doc_id
                file_hash = chunk.metadata.file_hash
                doc_version = chunk.metadata.doc_version

                # 检查已存在的切片
                existing = client.query(
                    collection_name=name,
                    filter=f'id like "{doc_id}%"',
                    output_fields=["id", "metadata"],
                    limit=1000,
                )

                if existing:
                    # 对比 file_hash 判断是否需要更新
                    existing_hash = None
                    for record in existing:
                        meta = record.get("metadata", {})
                        if meta.get("file_hash") == file_hash and file_hash:
                            # 版本未变，跳过
                            stats["skipped"] += 1
                            break
                        existing_hash = meta.get("file_hash")
                    else:
                        # 版本变更：删除旧切片
                        for record in existing:
                            client.delete(collection_name=name, filter=f'id == "{record["id"]}"')
                            stats["deleted"] += 1
                        # 插入新切片
                        result = self._do_insert_chunks([chunk], name, stats)
                        stats["updated"] += 1
                        stats["inserted"] += result["inserted"]
                else:
                    # 新文档，直接插入
                    result = self._do_insert_chunks([chunk], name, stats)
                    stats["inserted"] += result["inserted"]

            self._index = None  # 重建索引缓存
            logger.info(
                f"upsert 完成: {stats}",
                extra={"component": "index_manager"}
            )
            return stats

        except Exception as e:
            logger.error(f"upsert_chunk_objects 失败: {e}")
            return stats

    def _do_insert_chunks(
        self,
        chunks: List[Chunk],
        collection_name: str,
        stats: Dict[str, int],
    ) -> Dict[str, int]:
        """内部方法：实际插入 chunks"""
        self._ensure_embed_model()
        nodes = [c.to_text_node() for c in chunks]
        try:
            from llama_index.vector_stores.milvus import MilvusVectorStore
            tmp_store = MilvusVectorStore(
                uri=self.db_path,
                collection_name=collection_name,
                dim=self.dimension,
                overwrite=False,
            )
            tmp_ctx = StorageContext.from_defaults(vector_store=tmp_store)
            VectorStoreIndex(nodes=nodes, storage_context=tmp_ctx, show_progress=False)
            stats["inserted"] += len(nodes)
            logger.info(f"成功插入 {len(nodes)} 个 Chunk")
            return stats
        except Exception as e:
            logger.error(f"插入 Chunk 对象失败: {e}")
            return stats

    def get_doc_versions(
        self,
        doc_id: str,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        查询某个文档的所有历史版本信息。

        返回每个切片的 file_hash 和 doc_version（用于判断是否需要更新）。
        """
        from pymilvus import MilvusClient

        name = collection_name or self.collection_name
        try:
            client = MilvusClient(uri=self.db_path)
            if not client.has_collection(name):
                return []

            results = client.query(
                collection_name=name,
                filter=f'id like "{doc_id}%"',
                output_fields=["id", "metadata"],
                limit=10000,
            )

            versions = []
            seen_hashes = set()
            for record in results:
                meta = record.get("metadata", {})
                fh = meta.get("file_hash", "")
                if fh and fh not in seen_hashes:
                    seen_hashes.add(fh)
                    versions.append({
                        "file_hash": fh,
                        "doc_version": meta.get("doc_version"),
                        "last_modified": meta.get("last_modified"),
                        "chunk_count": sum(1 for r in results if r.get("metadata", {}).get("file_hash") == fh),
                    })
            return versions
        except Exception as e:
            logger.error(f"get_doc_versions 失败: {e}")
            return []

    def delete_chunks(
        self,
        chunk_ids: List[str],
        collection_name: Optional[str] = None,
    ) -> bool:
        """按 chunk_id 列表删除切片"""
        name = collection_name or self.collection_name
        try:
            from pymilvus import MilvusClient
            client = MilvusClient(uri=self.db_path)
            if not client.has_collection(name):
                return True
            for chunk_id in chunk_ids:
                client.delete(
                    collection_name=name,
                    filter=f'id == "{chunk_id}"',
                )
            logger.info(f"成功删除 {len(chunk_ids)} 个切片")
            # 清空缓存，使下次操作重建 index
            self._index = None
            return True
        except Exception as e:
            logger.error(f"删除切片失败: {e}")
            return False

    def delete_chunks_by_doc_id(
        self,
        doc_id: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        """根据 doc_id 前缀删除所有相关切片"""
        name = collection_name or self.collection_name
        try:
            from pymilvus import MilvusClient
            client = MilvusClient(uri=self.db_path)
            if not client.has_collection(name):
                return True
            client.delete(
                collection_name=name,
                filter=f'id like "{doc_id}%"',
            )
            logger.info(f"已删除文档 {doc_id} 的所有切片")
            self._index = None
            return True
        except Exception as e:
            logger.error(f"删除文档切片失败: {e}")
            return False

    def get_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """获取 Collection 统计信息"""
        name = collection_name or self.collection_name
        try:
            from pymilvus import MilvusClient
            client = MilvusClient(uri=self.db_path)
            if client.has_collection(name):
                stats = client.get_collection_stats(collection_name=name)
                return {
                    "collection_name": name,
                    "status": "exists",
                    "row_count": stats.get("row_count", 0),
                }
            return {"collection_name": name, "status": "not_found"}
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}

    def build_abac_filter(
        self,
        user_depts: List[str],
        user_projects: List[str],
    ) -> MetadataFilters:
        """
        构建 ABAC MetadataFilters（用于 as_retriever 时传入）。

        策略（union 模式）：
        - allowed_depts 为空 OR 用户部门在其中
        - allowed_projects 为空 OR 用户项目在其中
        任一满足即可返回结果。

        由于 MilvusVectorStore 的 MetadataFilters 不支持数组 contains，
        此处使用字符串包含作为近似过滤（生产环境建议用 Milvus Expr 原生语法）。
        """
        filters = []
        for dept in user_depts:
            filters.append(
                MetadataFilter(
                    key="allowed_depts",
                    value=dept,
                    operator=FilterOperator.CONTAINS,
                )
            )
        for proj in user_projects:
            filters.append(
                MetadataFilter(
                    key="allowed_projects",
                    value=proj,
                    operator=FilterOperator.CONTAINS,
                )
            )
        return MetadataFilters(filters=filters, condition="or") if filters else MetadataFilters(filters=[])

    def as_retriever(
        self,
        similarity_top_k: int = 5,
        abac_depts: Optional[List[str]] = None,
        abac_projects: Optional[List[str]] = None,
    ):
        """
        返回带 ABAC 过滤的 retriever。

        可直接用于 RAG 流程：
            retriever = manager.as_retriever(abac_depts=["研发部"])
            results = retriever.retrieve("请假流程")
        """
        index = self._get_index()
        kwargs: Dict[str, Any] = {"similarity_top_k": similarity_top_k}

        if abac_depts or abac_projects:
            mf = self.build_abac_filter(
                abac_depts or [], abac_projects or []
            )
            kwargs["filters"] = mf

        return index.as_retriever(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 全局单例
# ─────────────────────────────────────────────────────────────────────────────

_milvus_manager: Optional[MilvusIndexManager] = None


def get_milvus_manager(
    db_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    dimension: Optional[int] = None,
) -> MilvusIndexManager:
    """获取 MilvusIndexManager 单例"""
    global _milvus_manager
    if _milvus_manager is None:
        from config import settings as app_settings
        _milvus_manager = MilvusIndexManager(
            db_path=db_path or app_settings.milvus_db_path,
            collection_name=collection_name or app_settings.milvus_collection_name,
            dimension=dimension or app_settings.milvus_dimension,
        )
    return _milvus_manager
