"""
数据接入流水线包 (data_ingestion)  —  LlamaIndex 重构版

基于 LlamaIndex + Milvus 的文档 ETL 流水线。

模块说明：
- models:        统一数据模型（Document, Chunk, ChunkMetadata）
                 Document / Chunk 均持有对应的 LlamaIndex 原生对象引用
- parsers:       六大格式解析器（PDF, Word, Excel, PPTX, Markdown, TXT）
                 底层使用 llama-index-readers-file 官方 Reader
- chunker:       语义切块器
                 底层使用 MarkdownNodeParser / SentenceSplitter
- index_manager: Milvus 索引管理器（带 ABAC 标签）
                 底层使用 LlamaIndex MilvusVectorStore + VectorStoreIndex
- excel_storage: Excel 关系存储（PostgreSQL + Text-to-SQL）


快速使用示例：

    from data_ingestion import parse_document, chunk_document, get_milvus_manager

    # 1. 解析文档
    doc = parse_document("员工手册.pdf")

    # 2. 切块
    chunks = chunk_document(doc)

    # 3. 写入 Milvus
    manager = get_milvus_manager()
    manager.insert_chunk_objects(chunks)

    # 4. 检索（带 ABAC 过滤）
    retriever = manager.as_retriever(abac_depts=["研发部"])
    results = retriever.retrieve("年假怎么申请")
"""

from .models import Document, Chunk, ChunkMetadata, DocumentFormat
from .parsers import (
    BaseParser,
    PDFParser,
    WordParser,
    ExcelParser,
    PPTXParser,
    MarkdownParser,
    TextParser,
    get_parser_for_file,
    parse_document,
)
from .chunker import SemanticChunker, ChunkConfig, get_chunker, chunk_document
from .excel_storage import (
    ExcelRelationalStorage,
    get_excel_storage,
)
from .index_manager import (
    MilvusIndexManager,
    get_milvus_manager,
)
from .incremental_updater import (
    IncrementalUpdater,
    DiffResult,
)


__all__ = [
    # 数据模型
    "Document", "Chunk", "ChunkMetadata", "DocumentFormat",
    # 解析器
    "BaseParser", "PDFParser", "WordParser", "ExcelParser", "PPTXParser",
    "MarkdownParser", "TextParser", "get_parser_for_file", "parse_document",
    # 切块器
    "SemanticChunker", "ChunkConfig", "get_chunker", "chunk_document",
    # Excel 关系存储
    "ExcelRelationalStorage", "get_excel_storage",
    # 索引管理器
    "MilvusIndexManager", "get_milvus_manager",
    # 增量更新引擎
    "IncrementalUpdater", "DiffResult",
]
