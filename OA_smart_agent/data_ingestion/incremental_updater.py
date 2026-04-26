"""
增量更新引擎 (incremental_updater.py)

核心职责：计算新旧 Chunk 集合的差异，产出精准的操作计划。

设计原则：
- chunk_id 与内容强绑定（version_hash），与顺序无关
- 集合求差算法：新增 = 新_hash - 旧_hash，删除 = 旧_hash - 新_hash
- 幂等性：相同内容永远生成相同的 chunk_id，避免重复计算

使用场景：
- CSV/Excel 第一行插入数据，后续行的 chunk_id 不受影响
- Markdown 文档局部修改，只重新处理变更的章节
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .models import Chunk

logger = logging.getLogger("oa_agent.incremental")


@dataclass
class DiffResult:
    """增量更新的计算结果"""
    chunks_to_add: List[Chunk]      # 需要 Embedding 并写入向量库的新 Chunk
    chunk_ids_to_delete: List[str] # 需要从向量库删除的旧 Chunk ID
    keep_count: int                # 保持不变的 Chunk 数量
    add_count: int = 0
    delete_count: int = 0
    
    def __post_init__(self):
        self.add_count = len(self.chunks_to_add)
        self.delete_count = len(self.chunk_ids_to_delete)
    
    @property
    def has_changes(self) -> bool:
        """是否有任何变更"""
        return self.add_count > 0 or self.delete_count > 0
    
    def log_summary(self) -> str:
        return (
            f"增量计算完成: 新增/更新 {self.add_count} 块, "
            f"删除 {self.delete_count} 块, "
            f"保持不变 {self.keep_count} 块"
        )


class IncrementalUpdater:
    """
    文档增量更新计算引擎
    
    针对 Markdown（标题树块）和 CSV（行独立块）效果极佳。
    由于 chunk_id 与内容强绑定，局部修改不会引发全局雪崩。
    """
    
    def compute_diff(
        self,
        old_chunks_info: Dict[str, str],
        new_chunks: List[Chunk],
    ) -> DiffResult:
        """
        计算增量更新计划
        
        Args:
            old_chunks_info: 从关系型数据库拉取的旧 Chunk 记录
                             格式：{chunk_id: version_hash}
            new_chunks: 切块后的新 Chunk 列表
        
        Returns:
            DiffResult: 包含新增、删除、不变块信息的结构体
        """
        if not new_chunks:
            logger.warning("新 Chunk 列表为空，将删除所有旧 Chunk")
            return DiffResult(
                chunks_to_add=[],
                chunk_ids_to_delete=list(old_chunks_info.keys()),
                keep_count=0,
            )
        
        # 构建新数据的 Hash → Chunk 映射
        new_hash_map: Dict[str, Chunk] = {}
        for chunk in new_chunks:
            new_hash_map[chunk.version_hash] = chunk
        
        new_hashes: Set[str] = set(new_hash_map.keys())
        
        # 构建旧数据的 Hash 集合
        old_hashes: Set[str] = set(old_chunks_info.values())
        
        # 核心逻辑：集合求差
        hashes_to_add = new_hashes - old_hashes       # 新增或修改的
        hashes_to_keep = new_hashes & old_hashes      # 不变的
        hashes_to_delete = old_hashes - new_hashes    # 已删除的
        
        # 1. 提取需要新增/更新的 Chunk 对象
        chunks_to_add = [new_hash_map[h] for h in hashes_to_add]
        
        # 2. 找出需要删除的 chunk_id
        # 反向映射：根据 hash 找到旧的 chunk_id
        hash_to_old_id: Dict[str, str] = {v: k for k, v in old_chunks_info.items()}
        chunk_ids_to_delete = [hash_to_old_id[h] for h in hashes_to_delete]
        
        result = DiffResult(
            chunks_to_add=chunks_to_add,
            chunk_ids_to_delete=chunk_ids_to_delete,
            keep_count=len(hashes_to_keep),
        )
        
        logger.info(result.log_summary())
        return result
    
    def is_full_reindex_needed(
        self,
        old_chunks_info: Dict[str, str],
        new_chunks: List[Chunk],
        full_reindex_threshold: float = 0.5,
    ) -> bool:
        """
        判断是否需要全量重索引
        
        当变更比例超过阈值时，全量重索引可能比增量更新更高效。
        
        Args:
            old_chunks_info: 旧 Chunk 信息
            new_chunks: 新 Chunk 列表
            full_reindex_threshold: 触发全量重索引的变更比例阈值 (0.0-1.0)
        
        Returns:
            True 如果变更比例超过阈值
        """
        if not old_chunks_info:
            return True
        
        diff_result = self.compute_diff(old_chunks_info, new_chunks)
        total_chunks = len(old_chunks_info)
        
        if total_chunks == 0:
            return True
        
        change_ratio = (diff_result.add_count + diff_result.delete_count) / total_chunks
        
        if change_ratio > full_reindex_threshold:
            logger.info(
                f"变更比例 {change_ratio:.1%} 超过阈值 {full_reindex_threshold:.1%}，"
                f"建议全量重索引"
            )
            return True
        
        return False
