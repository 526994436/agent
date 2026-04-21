"""
Excel 原始表格存储 (excel_storage.py)  —  LlamaIndex 重构版（兼容层）

将 Excel 原始表格数据存储到关系型数据库（PostgreSQL）：

1. 原始表格 → PostgreSQL（关系型精确检索）
2. 支持 Text-to-SQL 查询

架构：
- 简单/语义查询 → 向量检索（行转句 → Milvus，由 parsers.py 处理）
- 复杂统计查询 → Text-to-SQL（PostgreSQL）

注意：此模块无需引入 LlamaIndex 依赖，
      向量写入已统一由 index_manager.MilvusIndexManager 负责。
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("oa_agent.excel_storage")

# pandas 为可选依赖
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


@dataclass
class TableSchema:
    """表结构定义"""
    table_name: str          # 数据库表名
    columns: List[str]       # 列名
    column_types: List[str]  # 列类型
    primary_key: str         # 主键列
    doc_id: str              # 关联的文档ID


@dataclass
class TableMetadata:
    """表格元数据（存储在 ChunkMetadata 中）"""
    # 数据库表名
    table_name: str

    # 表的 SQL 查询语句（用于 Text-to-SQL）
    select_query: str

    # 统计列（用于聚合查询）
    stat_columns: List[str]

    # 原始表格数据（JSON 格式，用于快速返回）
    raw_data: List[Dict[str, Any]]

    # 表结构
    schema: TableSchema


class ExcelRelationalStorage:
    """
    Excel 表格关系型存储。

    功能：
    1. 将 Excel 原始数据存入 PostgreSQL
    2. 创建图谱路由节点指向该表
    3. 提供 Text-to-SQL 生成接口

    设计理念：
    - 原始表格数据 → PostgreSQL（精确检索、统计分析）
    - 行转句描述 → Milvus（语义检索 + ABAC 标签过滤）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化存储。

        参数：
        - config: 配置（如 PostgreSQL 连接信息）
        """
        self.config = config or {}

        # PostgreSQL 配置
        self.pg_host = self.config.get("pg_host", "localhost")
        self.pg_port = self.config.get("pg_port", 5432)
        self.pg_database = self.config.get("pg_database", "oa_knowledge")
        self.pg_user = self.config.get("pg_user", "postgres")
        self.pg_password = self.config.get("pg_password", "password")

        self._connection = None

    def _get_connection(self):
        """获取数据库连接"""
        if self._connection is None:
            try:
                import psycopg2
                self._connection = psycopg2.connect(
                    host=self.pg_host,
                    port=self.pg_port,
                    database=self.pg_database,
                    user=self.pg_user,
                    password=self.pg_password,
                )
                logger.info(f"Connected to PostgreSQL at {self.pg_host}:{self.pg_port}")
            except ImportError:
                logger.error("psycopg2 not installed, run: pip install psycopg2-binary")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
                raise
        return self._connection

    def save_table(
        self,
        doc_id: str,
        df: 'pd.DataFrame',
        sheet_name: str,
    ) -> TableMetadata:
        """
        保存 Excel 表格到 PostgreSQL。

        参数：
        - doc_id: 文档ID
        - df: pandas DataFrame
        - sheet_name: Sheet 名称

        返回：
        - TableMetadata: 表格元数据（用于图谱路由）
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 生成表名（格式：excel_{doc_id}_{sheet_name}）
            safe_sheet = sheet_name.replace(" ", "_").replace("-", "_")
            table_name = f"excel_{doc_id}_{safe_sheet}".lower()

            # 构建 CREATE TABLE 语句
            columns_def = []
            for col in df.columns:
                safe_col = col.replace(" ", "_").replace("-", "_")
                columns_def.append(f'"{safe_col}" TEXT')

            # 添加 id 列
            columns_def.insert(0, 'id SERIAL PRIMARY KEY')

            create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(columns_def)})'

            # 执行创建表
            cursor.execute(create_sql)

            # 插入数据
            for idx, row in df.iterrows():
                values = [f"'{str(v).replace('', '')}'" if pd.notna(v) else "NULL"
                         for v in row]
                values_str = ", ".join(values)
                insert_sql = f'INSERT INTO "{table_name}" VALUES (DEFAULT, {values_str})'
                cursor.execute(insert_sql)

            conn.commit()

            # 构建元数据
            schema = TableSchema(
                table_name=table_name,
                columns=[str(c).replace(" ", "_").replace("-", "_") for c in df.columns],
                column_types=["TEXT"] * len(df.columns),
                primary_key="id",
                doc_id=doc_id,
            )

            # 构建 SELECT 查询语句
            select_query = f'SELECT * FROM "{table_name}"'

            # 统计列
            stat_columns = [str(c).replace(" ", "_").replace("-", "_")
                          for c in df.columns if any(kw in str(c).lower()
                          for kw in ["金额", "数量", "费用", "工资", "销量", "金额", "total", "sum", "count", "num"])]

            # 原始数据（JSON）
            raw_data = df.fillna("").to_dict(orient="records")

            metadata = TableMetadata(
                table_name=table_name,
                select_query=select_query,
                stat_columns=stat_columns,
                raw_data=raw_data,
                schema=schema,
            )

            logger.info(f"Saved table {table_name} with {len(df)} rows")

            cursor.close()
            return metadata

        except Exception as e:
            logger.error(f"Failed to save table: {e}")
            conn.rollback()
            raise

    def delete_table(self, table_name: str) -> bool:
        """
        删除表格。

        参数：
        - table_name: 表名

        返回：是否成功
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            drop_sql = f'DROP TABLE IF EXISTS "{table_name}"'
            cursor.execute(drop_sql)
            conn.commit()

            cursor.close()
            logger.info(f"Deleted table: {table_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete table: {e}")
            conn.rollback()
            return False

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """
        执行 SQL 查询。

        参数：
        - sql: SQL 语句

        返回：查询结果列表
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))

            cursor.close()
            return results

        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return []

    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        获取表格信息。

        返回：表结构信息
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 获取列信息
            cursor.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """)

            columns = []
            for row in cursor.fetchall():
                columns.append({
                    "name": row[0],
                    "type": row[1],
                })

            # 获取行数
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            row_count = cursor.fetchone()[0]

            cursor.close()

            return {
                "table_name": table_name,
                "columns": columns,
                "row_count": row_count,
            }

        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            return None


class TextToSQLGenerator:
    """
    Text-to-SQL 生成器。

    将自然语言查询转换为 SQL 语句。
    """

    def __init__(self, storage: Optional[ExcelRelationalStorage] = None):
        """
        初始化。

        参数：
        - storage: Excel 关系型存储实例
        """
        self.storage = storage or ExcelRelationalStorage()

        # 统计关键词映射
        self.stat_keywords = {
            "金额": "SUM({column})",
            "数量": "SUM({column})",
            "销量": "SUM({column})",
            "工资": "AVG({column})",
            "平均": "AVG({column})",
            "总计": "SUM({column})",
            "最高": "MAX({column})",
            "最低": "MIN({column})",
            "个数": "COUNT({column})",
            "人数": "COUNT({column})",
        }

    def generate_sql(
        self,
        query: str,
        table_name: str,
        column_hints: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        根据自然语言查询生成 SQL。

        参数：
        - query: 自然语言查询
        - table_name: 表名
        - column_hints: 列名提示

        返回：SQL 语句
        """
        query_lower = query.lower()

        # 判断查询类型
        if any(kw in query_lower for kw in ["多少", "几个", "总计", "合计", "sum", "count", "total"]):
            return self._generate_aggregate_sql(query, table_name, column_hints)
        elif any(kw in query_lower for kw in ["最高", "最大", "最低", "最小", "max", "min"]):
            return self._generate_extreme_sql(query, table_name, column_hints)
        elif any(kw in query_lower for kw in ["平均", "均", "avg"]):
            return self._generate_average_sql(query, table_name, column_hints)
        else:
            return self._generate_select_sql(query, table_name, column_hints)

    def _find_column(self, keywords: List[str], columns: List[str]) -> Optional[str]:
        """根据关键词找到匹配的列"""
        for kw in keywords:
            for col in columns:
                if kw in col.lower():
                    return f'"{col}"'
        return None

    def _generate_aggregate_sql(
        self,
        query: str,
        table_name: str,
        column_hints: Optional[List[str]] = None,
    ) -> str:
        """生成聚合查询 SQL"""
        columns = column_hints or []
        query_lower = query.lower()

        # 找金额/数量列
        stat_col = None
        for kw, func in self.stat_keywords.items():
            stat_col = self._find_column([kw], columns)
            if stat_col:
                break

        if not stat_col:
            # 默认统计行数
            return f'SELECT COUNT(*) as count FROM "{table_name}"'

        # 构建聚合查询
        col_name = stat_col.strip('"')

        if "金额" in columns[columns.index(col_name)] if col_name in columns else False:
            func = "SUM"
        else:
            func = "COUNT"

        return f'SELECT {func}({stat_col}) as result FROM "{table_name}"'

    def _generate_extreme_sql(
        self,
        query: str,
        table_name: str,
        column_hints: Optional[List[str]] = None,
    ) -> str:
        """生成极值查询 SQL"""
        columns = column_hints or []
        query_lower = query.lower()

        # 判断是最大值还是最小值
        if any(kw in query_lower for kw in ["最高", "最大", "max"]):
            func = "MAX"
        else:
            func = "MIN"

        # 找数值列
        stat_col = None
        for kw in ["金额", "数量", "销量", "工资", "价格"]:
            stat_col = self._find_column([kw], columns)
            if stat_col:
                break

        if not stat_col:
            # 默认取第一列
            stat_col = f'"{columns[0]}"' if columns else "*"

        return f'SELECT {func}({stat_col}) as extreme_value FROM "{table_name}"'

    def _generate_average_sql(
        self,
        query: str,
        table_name: str,
        column_hints: Optional[List[str]] = None,
    ) -> str:
        """生成平均值查询 SQL"""
        columns = column_hints or []

        # 找数值列
        stat_col = None
        for kw in ["金额", "数量", "销量", "工资", "价格"]:
            stat_col = self._find_column([kw], columns)
            if stat_col:
                break

        if not stat_col:
            stat_col = f'"{columns[0]}"' if columns else "*"

        return f'SELECT AVG({stat_col}) as average FROM "{table_name}"'

    def _generate_select_sql(
        self,
        query: str,
        table_name: str,
        column_hints: Optional[List[str]] = None,
    ) -> str:
        """生成普通 SELECT SQL"""
        return f'SELECT * FROM "{table_name}" LIMIT 100'


# 全局实例
_excel_storage: Optional[ExcelRelationalStorage] = None
_text_to_sql: Optional[TextToSQLGenerator] = None


def get_excel_storage(config: Optional[Dict[str, Any]] = None) -> ExcelRelationalStorage:
    """获取 Excel 关系型存储单例"""
    global _excel_storage
    if _excel_storage is None:
        _excel_storage = ExcelRelationalStorage(config)
    return _excel_storage


def get_text_to_sql_generator(
    storage: Optional[ExcelRelationalStorage] = None,
) -> TextToSQLGenerator:
    """获取 Text-to-SQL 生成器单例"""
    global _text_to_sql
    if _text_to_sql is None:
        _text_to_sql = TextToSQLGenerator(storage)
    return _text_to_sql
