"""
Excel 原始表格存储 (excel_storage.py)

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
from typing import Optional, Any

logger = logging.getLogger("oa_agent.excel_storage")

# pandas 为可选依赖
try:
    import pandas as pd
except ImportError:
    pd = None


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

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

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
        df: "pd.DataFrame",
        sheet_name: str,
    ) -> dict:
        """
        保存 Excel 表格到 PostgreSQL。

        返回：
            dict: 包含 table_name, stat_columns 等信息
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            safe_sheet = sheet_name.replace(" ", "_").replace("-", "_")
            table_name = f"excel_{doc_id}_{safe_sheet}".lower()

            columns_def = []
            for col in df.columns:
                safe_col = col.replace(" ", "_").replace("-", "_")
                columns_def.append(f'"{safe_col}" TEXT')

            columns_def.insert(0, 'id SERIAL PRIMARY KEY')
            create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(columns_def)})'
            cursor.execute(create_sql)

            for idx, row in df.iterrows():
                values = [
                    f"'{str(v).replace(\"'\", \"''\")}'" if pd.notna(v) else "NULL"
                    for v in row
                ]
                insert_sql = f'INSERT INTO "{table_name}" VALUES (DEFAULT, {", ".join(values)})'
                cursor.execute(insert_sql)

            conn.commit()

            stat_columns = [
                str(c).replace(" ", "_").replace("-", "_")
                for c in df.columns
                if any(kw in str(c).lower() for kw in ["金额", "数量", "费用", "工资", "销量", "total", "sum", "count", "num"])
            ]

            logger.info(f"Saved table {table_name} with {len(df)} rows")
            cursor.close()

            return {
                "table_name": table_name,
                "stat_columns": stat_columns,
                "row_count": len(df),
                "column_count": len(df.columns),
            }

        except Exception as e:
            logger.error(f"Failed to save table: {e}")
            conn.rollback()
            raise


_excel_storage: Optional[ExcelRelationalStorage] = None


def get_excel_storage(config: Optional[dict] = None) -> ExcelRelationalStorage:
    """获取 Excel 关系型存储单例"""
    global _excel_storage
    if _excel_storage is None:
        _excel_storage = ExcelRelationalStorage(config)
    return _excel_storage
