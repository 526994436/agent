"""
统一数据模型 (models.py)

基于 LlamaIndex 原生数据结构进行封装，保持向后兼容。

核心映射关系：
- Document  ←→  llama_index.core.schema.Document
- Chunk     ←→  llama_index.core.schema.TextNode
- ChunkMetadata → node.metadata dict（保持 ABAC 权限字段）

这一层的职责：
1. 保留 DocumentFormat、ChunkMetadata 等业务枚举/数据类，
   供上层（parsers / chunker / index_manager）使用
2. 提供 to_llama_document() / from_llama_node() 等互转工具
3. 不再自行实现 _compute_file_hash / _generate_doc_id，
   直接透传给 LlamaIndex Document 的 doc_id/metadata
"""

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from llama_index.core.schema import Document as LlamaDocument
from llama_index.core.schema import TextNode


class DocumentFormat(str, Enum):
    """支持的文档格式"""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"
    PPTX = "pptx"
    MARKDOWN = "md"
    TEXT = "txt"
    UNKNOWN = "unknown"


@dataclass
class ChunkMetadata:
    """
    切片元数据（ABAC 权限模型 + 层级路径）。

    存储在 LlamaIndex TextNode.metadata 中，字段扁平化为：
    {
        "header_path": "...",
        "allowed_depts": [...],
        "allowed_projects": [...],
        "doc_title": "...",
        "source_file": "...",
        "page_number": 1,
        "tags": [...],
        ...extra fields
    }

    ABAC 权限字段说明：
    - allowed_depts: 空列表表示不限部门
    - allowed_projects: 空列表表示不限项目
    """
    header_path: Optional[str] = None
    allowed_depts: List[str] = field(default_factory=list)
    allowed_projects: List[str] = field(default_factory=list)
    doc_title: Optional[str] = None
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    table_data: Optional[List[List[str]]] = None
    tags: List[str] = field(default_factory=list)
    # ── 版本管理字段 ──────────────────────────────────────────────────────
    doc_version: Optional[str] = None   # 文档版本号，如 "v1.0"
    file_hash: Optional[str] = None     # 文件内容 hash（检测文件变更）
    last_modified: Optional[str] = None # 文件最后修改时间（ISO 格式）
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为扁平字典（兼容 TextNode.metadata）"""
        d = {
            "header_path": self.header_path,
            "allowed_depts": self.allowed_depts,
            "allowed_projects": self.allowed_projects,
            "doc_title": self.doc_title,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "tags": self.tags,
            "doc_version": self.doc_version,
            "file_hash": self.file_hash,
            "last_modified": self.last_modified,
        }
        if self.table_data is not None:
            d["table_data"] = self.table_data
        d.update(self.extra)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        known_keys = {
            "header_path", "allowed_depts", "allowed_projects",
            "doc_title", "source_file", "page_number", "table_data", "tags",
            "doc_version", "file_hash", "last_modified",
        }
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in data.items():
            if k in known_keys:
                kwargs[k] = v
            else:
                extra[k] = v
        kwargs["extra"] = extra
        return cls(**kwargs)

    def to_milvus_filter(self) -> Dict[str, Any]:
        return {
            "allowed_depts": self.allowed_depts,
            "allowed_projects": self.allowed_projects,
        }


@dataclass
class Chunk:
    """
    文档切片（对 LlamaIndex TextNode 的薄封装）。

    持有一个 llama_node 引用，通过属性代理访问核心字段，
    同时保留 chunk_id / doc_id / version_hash / sequence 等业务字段。
    """
    chunk_id: str
    doc_id: str
    version_hash: str
    content: str
    metadata: ChunkMetadata
    sequence: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # 对应的 LlamaIndex TextNode（可选，由 chunker 填充）
    llama_node: Optional[TextNode] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "version_hash": self.version_hash,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "sequence": self.sequence,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        metadata = ChunkMetadata.from_dict(data.get("metadata", {}))
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            version_hash=data["version_hash"],
            content=data["content"],
            metadata=metadata,
            sequence=data.get("sequence", 0),
            created_at=data.get("created_at", ""),
        )

    def to_text_node(self) -> TextNode:
        """转换为 LlamaIndex TextNode（用于向量写入）"""
        if self.llama_node is not None:
            return self.llama_node
        meta = self.metadata.to_dict()
        meta.update({
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "version_hash": self.version_hash,
            "sequence": self.sequence,
        })
        return TextNode(
            id_=self.chunk_id,
            text=self.content,
            metadata=meta,
        )

    @classmethod
    def from_text_node(cls, node: TextNode, doc_id: str = "") -> "Chunk":
        """从 LlamaIndex TextNode 创建 Chunk"""
        meta_dict = node.metadata or {}
        chunk_id = meta_dict.get("chunk_id") or node.node_id
        version_hash = meta_dict.get("version_hash", "")
        sequence = meta_dict.get("sequence", 0)
        metadata = ChunkMetadata.from_dict(meta_dict)
        return cls(
            chunk_id=chunk_id,
            doc_id=doc_id or meta_dict.get("doc_id", ""),
            version_hash=version_hash,
            content=node.get_content(),
            metadata=metadata,
            sequence=sequence,
            llama_node=node,
        )


@dataclass
class Document:
    """
    完整文档（对 LlamaIndex Document 的薄封装）。

    持有一个 llama_doc 引用，同时保留业务字段（doc_id / content_hash / chunks）。
    """
    doc_id: str
    title: str
    format: DocumentFormat
    file_path: str
    markdown_content: str = ""
    chunks: List[Chunk] = field(default_factory=list)
    content_hash: str = ""
    file_size: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # 对应的 LlamaIndex Document（可选，由 parsers 填充）
    llama_doc: Optional[LlamaDocument] = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        file_path: str,
        title: str = "",
        format: Optional[DocumentFormat] = None,
    ) -> "Document":
        """工厂方法：创建 Document 并计算哈希值"""
        if format is None:
            ext = os.path.splitext(file_path)[1].lower().lstrip(".")
            format = (
                DocumentFormat(ext)
                if ext in [e.value for e in DocumentFormat]
                else DocumentFormat.UNKNOWN
            )
        if not title:
            title = os.path.splitext(os.path.basename(file_path))[0]

        content_hash = cls._compute_file_hash(file_path)
        file_size = os.path.getsize(file_path)
        doc_id = cls._generate_doc_id(file_path, content_hash)

        return cls(
            doc_id=doc_id,
            title=title,
            format=format,
            file_path=file_path,
            content_hash=content_hash,
            file_size=file_size,
        )

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    @staticmethod
    def _generate_doc_id(file_path: str, content_hash: str) -> str:
        basename = os.path.basename(file_path)
        prefix = basename[:8].upper().replace(".", "_")
        hash_prefix = content_hash[:6]
        return f"D-{prefix}-{hash_prefix}"

    def to_llama_document(self) -> LlamaDocument:
        """转换为 LlamaIndex Document（用于 NodeParser 输入）"""
        if self.llama_doc is not None:
            return self.llama_doc
        return LlamaDocument(
            doc_id=self.doc_id,
            text=self.markdown_content,
            metadata={
                "doc_id": self.doc_id,
                "title": self.title,
                "format": self.format.value,
                "file_path": self.file_path,
                "content_hash": self.content_hash,
                "file_size": self.file_size,
            },
        )

    @classmethod
    def from_llama_document(cls, llama_doc: LlamaDocument) -> "Document":
        """从 LlamaIndex Document 创建"""
        meta = llama_doc.metadata or {}
        fmt_val = meta.get("format", "unknown")
        fmt = (
            DocumentFormat(fmt_val)
            if fmt_val in [e.value for e in DocumentFormat]
            else DocumentFormat.UNKNOWN
        )
        doc = cls(
            doc_id=meta.get("doc_id") or llama_doc.doc_id,
            title=meta.get("title", ""),
            format=fmt,
            file_path=meta.get("file_path", ""),
            markdown_content=llama_doc.get_content(),
            content_hash=meta.get("content_hash", ""),
            file_size=meta.get("file_size", 0),
            llama_doc=llama_doc,
        )
        return doc

    def add_chunk(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)

    def add_chunks(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "format": self.format.value,
            "file_path": self.file_path,
            "markdown_content": self.markdown_content,
            "chunks": [c.to_dict() for c in self.chunks],
            "content_hash": self.content_hash,
            "file_size": self.file_size,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        format_val = data.get("format", "unknown")
        format_enum = (
            DocumentFormat(format_val)
            if isinstance(format_val, str)
            else format_val
        )
        chunks = [Chunk.from_dict(c) for c in data.get("chunks", [])]
        return cls(
            doc_id=data["doc_id"],
            title=data.get("title", ""),
            format=format_enum,
            file_path=data.get("file_path", ""),
            markdown_content=data.get("markdown_content", ""),
            chunks=chunks,
            content_hash=data.get("content_hash", ""),
            file_size=data.get("file_size", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )
