"""
查询路由层 (query_router.py)  —  LlamaIndex 重构版（兼容层）

将 Text-to-SQL 和向量检索整合到 Agent 查询流程中：

1. 意图判断：统计查询 vs 语义查询
2. 路由执行：
   - 统计查询 → TextToSQLGenerator → PostgreSQL
   - 语义查询 → MilvusIndexManager.as_retriever()（LlamaIndex 检索）
3. 结果合并

注意：语义查询路径已改为通过 LlamaIndex Retriever 接口，
      需要在调用处确保 Settings.embed_model 已初始化。
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .excel_storage import ExcelRelationalStorage, TextToSQLGenerator, get_excel_storage, get_text_to_sql_generator

logger = logging.getLogger("oa_agent.query_router")


class QueryType(str, Enum):
    """查询类型"""
    SEMANTIC = "semantic"
    STATISTICAL = "statistical"


@dataclass
class QueryIntent:
    """查询意图"""
    query_type: QueryType
    target_tables: List[str]
    keywords: List[str]
    confidence: float


@dataclass
class QueryResult:
    """查询结果"""
    query_type: QueryType
    answer: str
    data: Optional[List[Dict]]
    source: str


class QueryRouter:
    """
    查询路由器。

    根据用户查询类型，选择合适的检索/查询路径：
    1. 统计类查询 → Text-to-SQL → PostgreSQL
    2. 语义类查询 → 向量检索 → Milvus
    """

    STAT_KEYWORDS = [
        "多少", "几个", "总计", "合计", "总数", "总额", "总金额", "总数量",
        "sum", "count", "total", "最高", "最低", "最大", "最小",
        "平均", "均", "avg", "排名", "排序", "对比", "各部门",
    ]

    TABLE_KEYWORDS = [
        "报销", "工资", "考勤", "请假", "费用", "支出", "收入",
        "员工", "部门", "项目", "客户",
    ]

    def __init__(
        self,
        pg_storage: Optional[ExcelRelationalStorage] = None,
        sql_generator: Optional[TextToSQLGenerator] = None,
    ):
        self.pg_storage = pg_storage or get_excel_storage()
        self.sql_generator = sql_generator or get_text_to_sql_generator(self.pg_storage)

    def analyze_intent(self, query: str) -> QueryIntent:
        """分析查询意图"""
        query_lower = query.lower()

        stat_score = 0
        matched_keywords = []

        for kw in self.STAT_KEYWORDS:
            if kw.lower() in query_lower:
                stat_score += 1
                matched_keywords.append(kw)

        target_tables = self._find_target_tables(query_lower)

        if stat_score >= 1:
            query_type = QueryType.STATISTICAL
            confidence = min(0.5 + stat_score * 0.1, 0.95)
        else:
            query_type = QueryType.SEMANTIC
            confidence = 0.7

        return QueryIntent(
            query_type=query_type,
            target_tables=target_tables,
            keywords=matched_keywords,
            confidence=confidence,
        )

    def _find_target_tables(self, query_lower: str) -> List[str]:
        """查找与查询相关的表"""
        tables = []
        try:
            conn = self.pg_storage._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name LIKE 'excel_%'
            """)
            for row in cursor.fetchall():
                table_name = row[0]
                for kw in self.TABLE_KEYWORDS:
                    if kw in table_name.lower():
                        tables.append(table_name)
                        break
            cursor.close()
        except Exception as e:
            logger.warning(f"Failed to get tables: {e}")
        return tables

    def route(self, query: str, intent: Optional[QueryIntent] = None) -> QueryResult:
        """根据查询类型执行路由"""
        if intent is None:
            intent = self.analyze_intent(query)

        if intent.query_type == QueryType.STATISTICAL:
            return self._execute_statistical_query(query, intent)
        else:
            return self._execute_semantic_query(query, intent)

    def _execute_statistical_query(self, query: str, intent: QueryIntent) -> QueryResult:
        """执行统计查询（Text-to-SQL）"""
        if not intent.target_tables:
            return QueryResult(
                query_type=QueryType.STATISTICAL,
                answer="未找到相关数据表，请先上传 Excel 文件。",
                data=None,
                source="none",
            )

        results = []
        for table_name in intent.target_tables:
            table_info = self.pg_storage.get_table_info(table_name)
            if not table_info:
                continue

            columns = [c["name"] for c in table_info["columns"]]
            sql = self.sql_generator.generate_sql(query, table_name, columns)
            if not sql:
                continue

            try:
                data = self.pg_storage.execute_query(sql)
                results.extend(data)
            except Exception as e:
                logger.error(f"SQL execution failed: {e}")

        if results:
            answer = self._format_stat_result(query, results)
            return QueryResult(
                query_type=QueryType.STATISTICAL,
                answer=answer,
                data=results,
                source="postgresql",
            )
        else:
            return QueryResult(
                query_type=QueryType.STATISTICAL,
                answer="未找到匹配的数据。",
                data=None,
                source="postgresql",
            )

    def _execute_semantic_query(self, query: str, intent: QueryIntent) -> QueryResult:
        """
        执行语义查询（LlamaIndex Retriever → Milvus 向量检索）

        通过 MilvusIndexManager.as_retriever() 使用 LlamaIndex 标准检索接口，
        返回 NodeWithScore 列表，汇总为自然语言答案。
        """
        try:
            from .index_manager import get_milvus_manager
            manager = get_milvus_manager()
            retriever = manager.as_retriever(similarity_top_k=5)
            results = retriever.retrieve(query)
            if results:
                snippets = [r.node.get_content()[:300] for r in results]
                answer = "检索到以下相关内容：\n\n" + "\n\n---\n\n".join(snippets)
                data = [
                    {
                        "chunk_id": r.node.metadata.get("chunk_id", r.node.node_id),
                        "score": r.score,
                        "text": r.node.get_content()[:500],
                        "source": r.node.metadata.get("source_file", ""),
                    }
                    for r in results
                ]
                return QueryResult(
                    query_type=QueryType.SEMANTIC,
                    answer=answer,
                    data=data,
                    source="milvus_vector_db",
                )
        except Exception as e:
            logger.warning(f"语义检索失败: {e}")

        return QueryResult(
            query_type=QueryType.SEMANTIC,
            answer="向量检索暂时不可用，请稍后重试。",
            data=None,
            source="vector_db",
        )

    def _format_stat_result(self, query: str, results: List[Dict]) -> str:
        """格式化��计结果为自然语言"""
        if not results:
            return "未找到匹配的数据。"

        first_result = results[0]

        if len(results) == 1 and len(first_result) == 1:
            key = list(first_result.keys())[0]
            value = first_result[key]
            return f"查询结果为：{value}"

        lines = []
        for i, row in enumerate(results[:10], 1):
            row_str = "，".join([f"{k}为{v}" for k, v in row.items()])
            lines.append(f"{i}. {row_str}")

        if len(results) > 10:
            lines.append(f"... 还有 {len(results) - 10} 条结果")

        return "查询结果如下：\n" + "\n".join(lines)


_router: Optional[QueryRouter] = None


def get_query_router() -> QueryRouter:
    """获取查询路由器单例"""
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router