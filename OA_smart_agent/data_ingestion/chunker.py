"""
统一切块器 (chunker.py)  —  LlamaIndex 重构版

核心变化：
- 按文档类型自动选择 LlamaIndex NodeParser：
    Markdown / DOCX → MarkdownNodeParser（基于标题层级）
    Excel / CSV     → SentenceSplitter（行转句，固定 chunk_size）
    PDF / TXT 等    → SentenceSplitter（段落 + 重叠）
- 所有 NodeParser 的输出 TextNode 封装回业务 Chunk 对象，
  保留 ABAC 权限字段（allowed_depts / allowed_projects）以及
  版本哈希、序号等元数据
- 公开接口与旧版完全一致：
    SemanticChunker.chunk_document(document) -> List[Chunk]
    get_chunker(config)                      -> SemanticChunker
    chunk_document(document, config)         -> List[Chunk]

依赖：
    llama-index-core（MarkdownNodeParser、SentenceSplitter 均在 core 中）
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode

from .models import Chunk, ChunkMetadata, Document

logger = logging.getLogger("oa_agent.chunker")


# ─────────────────────────────────────────────────────────────────────────────
# 配置类（接口与旧版保持一致）
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkConfig:
    """切块配置"""
    max_chunk_size: int = 500
    min_chunk_size: int = 50
    overlap_chars: int = 75
    overlap_ratio: float = 0.15
    include_headers: bool = True
    header_depth: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# 主切块器
# ─────────────────────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    语义级切块器（LlamaIndex NodeParser 封装）。

    根据文档格式选择最合适的 NodeParser：
    - Markdown / DOCX → MarkdownNodeParser（标题层级切分）
    - Excel / CSV     → SentenceSplitter（行转句，无重叠）
    - 其他（PDF/TXT） → SentenceSplitter（段落 + overlap）
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        overlap = self._calc_overlap()

        # Markdown 切分器（按标题层级，内置 include_metadata）
        self._markdown_parser = MarkdownNodeParser(
            include_metadata=True,
            include_prev_next_rel=False,
        )

        # 通用段落切分器（SentenceSplitter 兼顾句子边界 + overlap）
        self._sentence_splitter = SentenceSplitter(
            chunk_size=self.config.max_chunk_size,
            chunk_overlap=overlap,
            paragraph_separator="\n\n",
        )

        # Excel 专用（行已转句，无需 overlap）
        self._excel_splitter = SentenceSplitter(
            chunk_size=self.config.max_chunk_size,
            chunk_overlap=0,
            paragraph_separator="\n",
        )

    def _calc_overlap(self) -> int:
        cfg = self.config
        if cfg.overlap_chars == 75 and cfg.overlap_ratio > 0:
            return int(cfg.max_chunk_size * cfg.overlap_ratio)
        return cfg.overlap_chars

    # ── 主入口 ───────────────────────────────────────────────────────────────

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        将 Document 切分为 Chunk 列表。

        策略：
        - md / markdown / docx → MarkdownNodeParser
        - xlsx / xls / csv     → SentenceSplitter（无 overlap）
        - 其他                 → SentenceSplitter（含 overlap）
        """
        if not document.markdown_content:
            logger.warning(f"Document {document.doc_id} 无 markdown 内容")
            return []

        fmt = document.format.value
        if fmt in ("md", "markdown", "docx", "doc"):
            return self._chunk_with_markdown_parser(document)
        elif fmt in ("xlsx", "xls", "csv"):
            return self._chunk_with_splitter(document, self._excel_splitter)
        else:
            return self._chunk_with_splitter(document, self._sentence_splitter)

    # ── Markdown 标题层级切分 ─────────────────────────────────────────────────

    def _chunk_with_markdown_parser(self, document: Document) -> List[Chunk]:
        """
        使用 MarkdownNodeParser 按标题层级切分。

        若生成的节点内容过长，再用 SentenceSplitter 二次切分。
        """
        llama_doc = document.to_llama_document()
        nodes: List[TextNode] = self._markdown_parser.get_nodes_from_documents(
            [llama_doc]
        )

        chunks: List[Chunk] = []
        seq = 0

        for node in nodes:
            text = node.get_content()
            if not text or len(text) < self.config.min_chunk_size:
                continue

            header_path = self._extract_header_path_from_node(node)

            if len(text) > self.config.max_chunk_size:
                # 二次切分
                sub_nodes = self._sentence_splitter.get_nodes_from_documents(
                    [self._node_to_llama_doc(node, document.doc_id)]
                )
                for sub in sub_nodes:
                    sub_text = sub.get_content()
                    if len(sub_text) >= self.config.min_chunk_size:
                        chunks.append(
                            self._build_chunk(
                                document, sub_text, header_path, seq
                            )
                        )
                        seq += 1
            else:
                chunks.append(
                    self._build_chunk(document, text, header_path, seq)
                )
                seq += 1

        return chunks

    # ── 通用 SentenceSplitter 切分 ────────────────────────────────────────────

    def _chunk_with_splitter(
        self, document: Document, splitter: SentenceSplitter
    ) -> List[Chunk]:
        """使用 SentenceSplitter 切分（PDF / TXT / Excel）"""
        llama_doc = document.to_llama_document()
        nodes: List[TextNode] = splitter.get_nodes_from_documents([llama_doc])

        chunks: List[Chunk] = []
        header_path = document.title or "全文"

        for seq, node in enumerate(nodes):
            text = node.get_content()
            if not text or len(text) < self.config.min_chunk_size:
                continue
            chunks.append(self._build_chunk(document, text, header_path, seq))

        return chunks

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def _build_chunk(
        self,
        document: Document,
        content: str,
        header_path: str,
        sequence: int,
    ) -> Chunk:
        """构造 Chunk 对象（附带版本哈希和元数据）"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        version_hash = f"{document.content_hash[:6]}-{content_hash}"
        
        # 使用 version_hash 作为 chunk_id 后缀，与顺序解绑
        # 彻底解决增量场景下 sequence 移位导致的全量重算问题
        chunk_id = f"{document.doc_id}-C-{version_hash}"

        metadata = ChunkMetadata(
            header_path=header_path,
            doc_title=document.title,
            source_file=document.file_path,
            extra={"content_hash": content_hash},
        )

        node = TextNode(
            id_=chunk_id,
            text=content,
            metadata={
                **metadata.to_dict(),
                "chunk_id": chunk_id,
                "doc_id": document.doc_id,
                "version_hash": version_hash,
                "sequence": sequence,
            },
        )

        return Chunk(
            chunk_id=chunk_id,
            doc_id=document.doc_id,
            version_hash=version_hash,
            content=content,
            metadata=metadata,
            sequence=sequence,
            llama_node=node,
        )

    @staticmethod
    def _extract_header_path_from_node(node: TextNode) -> str:
        """
        从 MarkdownNodeParser 生成的节点元数据中提取标题路径。

        MarkdownNodeParser 在 node.metadata 中存储：
        {
            "Header 1": "一级标题",
            "Header 2": "二级标题",
            ...
        }
        """
        meta = node.metadata or {}
        parts: List[str] = []
        for level in range(1, 7):
            key = f"Header {level}"
            if key in meta and meta[key]:
                parts.append(str(meta[key]))
        return " -> ".join(parts) if parts else ""

    @staticmethod
    def _node_to_llama_doc(node: TextNode, doc_id: str) -> "LlamaDocument":  # type: ignore[name-defined]
        """将 TextNode 转换为临时 LlamaDocument（用于二次切分）"""
        from llama_index.core.schema import Document as LlamaDocument
        return LlamaDocument(
            doc_id=doc_id,
            text=node.get_content(),
            metadata=node.metadata or {},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 全局单例 & 便捷函数
# ─────────────────────────────────────────────────────────────────────────────

_chunker: Optional[SemanticChunker] = None


def get_chunker(config: Optional[ChunkConfig] = None) -> SemanticChunker:
    """
    获取全局 SemanticChunker 单例。

    传入 config 时强制重建实例。
    """
    global _chunker
    if _chunker is None or config is not None:
        _chunker = SemanticChunker(config)
    return _chunker


def chunk_document(
    document: Document,
    config: Optional[ChunkConfig] = None,
) -> List[Chunk]:
    """便捷函数：切分 Document 为 Chunk 列表"""
    return get_chunker(config).chunk_document(document)
