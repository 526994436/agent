"""
配置管理模块 (config.py)

作用：
    这个文件就像是一个"总控面板"，集中管理 OA 智能 Agent 的所有配置。
    所有敏感的连接信息（数据库密码、API密钥等）都通过环境变量注入，
    而不是硬编码在代码里，这样更安全。

配置项说明：
- OPENAI 配置：设置使用哪个 AI 模型、API密钥、请求超时等
- ABAC 权限过滤：控制哪些用户可以查看哪些文档
- 混合检索：结合向量搜索和关键词搜索，提供更准确的搜索结果
- Java 后端：配置如何连接到后端的 Java 服务
- 数据库：配置 PostgreSQL 数据库连接，用于保存对话状态
- 监控：配置如何暴露监控指标给 Prometheus

环境隔离：
    支持 dev（开发）、staging（预发布）、prod（生产）三套环境配置。
    不同环境加载不同的配置值，互不影响。
"""

# 导入 Pydantic Settings，用于从环境变量读取配置
from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    """
    全局配置类，统一管理所有环境变量。

    这个类的每个属性都对应一个配置项，可以直接在代码中使用 settings.xxx 来访问。
    配置值会从 .env 文件或系统环境变量中读取。

    示例：
        from config import settings
        print(settings.vllm_model)  # 输出: Qwen/Qwen2.5-14B-Instruct
    """

    # ─────────────────────────────────────────────
    # vLLM（Qwen）模型配置
    # ─────────────────────────────────────────────
    vllm_base_url: str = "http://localhost:8000/v1"  # vLLM API 地址
    vllm_api_key: str = "not-needed"  # vLLM 不需要 key
    vllm_model: str = "Qwen3.5-14B-Instruct"  # 模型名称
    vllm_request_timeout: int = 60  # 超时时间（秒）
    vllm_max_tokens: int = 2000  # 最大输出 token

    # ─────────────────────────────────────────────
    # Embedding 模型配置（用于文本向量化）
    # ─────────────────────────────────────────────
    embedding_provider: str = "bge"  # ollama / openai / bge（内网部署）
    embedding_model: str = "BGE-M3"  # 内网部署的 BGE-M3 多语言向量化模型
    embedding_base_url: str = "http://localhost:8001/v1"  # BGE embedding 服务地址（内网）
    embedding_api_key: str = "not-needed"  # BGE 内网部署通常不需要 key

    # ─────────────────────────────────────────────
    # ABAC 权限过滤配置
    # 作用：控制哪些用户可以查看哪些文档，实现"千人千面"的权限管理
    # ─────────────────────────────────────────────
    # 访问控制模式：abac（基于属性的访问控制）、rbac（基于角色的）、disabled（不启用）
    access_control_mode: str = "abac"
    
    # 默认权限策略：
    # "union" = 满足部门或项目任一条件即可访问
    # "intersection" = 必须同时满足部门和项目条件才能访问
    default_access_policy: str = "union"
    
    # 是否启用 ABAC 过滤（关闭后所有用户都能看到所有文档）
    abac_filter_enabled: bool = True
    
    # Milvus 向量数据库配置（用于存储文档的数学表示）
    milvus_db_path: str = "./data/milvus_lite.db"  # 数据库文件存放路径
    milvus_collection_name: str = "oa_chunks"  # 集合名称，类似数据库的表名
    milvus_dimension: int = 1024  # 向量维度，BGE-M3 dense 向量为 1024 维

    # ─────────────────────────────────────────────
    # 混合检索配置
    # 作用：结合两种搜索方式，提供更准确的搜索结果
    # - 向量检索：理解语义，比如"电脑坏了"能找到"设备故障"
    # - BM25检索：精准匹配关键词
    # ─────────────────────────────────────────────
    hybrid_search_enabled: bool = True  # 是否启用混合检索
    hybrid_dense_weight: float = 0.7  # 向量检索的权重（0-1之间），越大越依赖语义理解
    hybrid_sparse_weight: float = 0.3  # 关键词检索的权重，越大越依赖精确匹配
    hybrid_rrf_k: int = 60  # RRF融合参数，用于合并两种搜索结果
    hybrid_vector_top_k: int = 100  # 向量检索返回的最多个数
    hybrid_bm25_top_k: int = 100  # 关键词检索返回的最多个数
    hybrid_final_top_k: int = 10  # 最终返回给用户的结果个数

    # Chunk ID 缓存配置（加速重复查询，避免重复查询数据库）
    chunk_id_cache_ttl: int = 300  # 缓存有效期（秒），5分钟
    chunk_id_cache_enabled: bool = True  # 是否启用缓存

    # ─────────────────────────────────────────────
    # Java 后端地址配置
    # 作用：Python Agent 通过这些地址调用 Java 后端服务
    # 注意：实际生产环境中，Java 后端负责真正的业务操作
    # ─────────────────────────────────────────────
    java_backend_base_url: str = "http://localhost:8000"  # Java 后端的基础地址
    java_backend_timeout: int = 30  # 请求超时时间（秒）
    java_backend_max_retries: int = 3  # 请求失败重试次数
    java_backend_circuit_breaker_timeout: int = 60  # 断路器冷却时间（秒），防止频繁调用不可用服务

    # ─────────────────────────────────────────────
    # MCP 配置
    # 作用：MCP（Model Context Protocol）是一种标准化协议，用于 Agent 调用外部工具
    # ─────────────────────────────────────────────
    mcp_enabled: bool = True  # 是否启用 MCP
    mcp_server_name: str = "JavaBackendTools"  # MCP 服务器名称

    # ─────────────────────────────────────────────
    # JWT 鉴权配置
    # 作用：验证用户身份，确保只有合法用户才能使用 Agent
    # 注意：只验证 Token 有效性，不解析具体业务内容
    # ─────────────────────────────────────────────
    jwt_secret: str = "change-me-in-production"  # JWT 密钥，生产环境必须修改！
    jwt_algorithm: str = "HS256"  # 加密算法
    jwt_expire_minutes: int = 480  # Token 过期时间（分钟），480分钟=8小时

    # ─────────────────────────────────────────────
    # CORS 配置
    # 作用：控制哪些网站可以访问这个 API，防止被恶意调用
    # ─────────────────────────────────────────────
    cors_allowed_origins: List[str] = ["http://localhost:3000"]  # 允许访问的网站列表

    # ─────────────────────────────────────────────
    # PostgreSQL Checkpointer 配置
    # 作用：PostgreSQL 数据库用于保存 Agent 的对话状态
    # 这样即使用户关闭浏览器，下次打开还能继续上次的对话
    # ─────────────────────────────────────────────
    postgres_checkpointer_url: Optional[str] = None  # 数据库连接地址
    postgres_pool_size: int = 10  # 连接池大小，同时保持多少个连接
    postgres_max_overflow: int = 20  # 最大溢出连接数，紧急情况可以多用几个

    # ─────────────────────────────────────────────
    # 对话历史持久化配置
    # 作用：保存用户和 Agent 的完整对话记录，用于审计和分析
    # 支持 PostgreSQL 或 MySQL
    # ─────────────────────────────────────────────
    history_db_type: str = "postgresql"  # 数据库类型
    history_db_url: Optional[str] = None  # 数据库连接地址
    history_db_pool_size: int = 10  # 连接池大小
    history_db_max_overflow: int = 20  # 最大溢出连接数
    history_save_async: bool = True  # 是否异步保存（开启后不阻塞主流程）

    # ─────────────────────────────────────────────
    # 日志配置
    # 作用：配置日志输出格式和级别，便于排查问题
    # ─────────────────────────────────────────────
    log_level: str = "INFO"  # 日志级别：DEBUG/INFO/WARNING/ERROR
    log_format: str = "json"  # 日志格式：json（机器解析）或 text（人类阅读）
    log_output: str = "stdout"  # 日志输出位置：stdout（控制台）或 file（文件）

    # ─────────────────────────────────────────────
    # Re-Ranking 配置
    # 作用：对搜索结果进行二次排序，把最相关的结果排在前面
    # 使用 BGE-Rerank 模型进行精排
    # ─────────────────────────────────────────────
    reranking_enabled: bool = True  # 是否启用重排序
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 重排序模型名称
    reranking_initial_top_k: int = 50  # 初次检索返回的数量
    reranking_final_top_k: int = 5  # 重排后返回的数量
    reranking_use_mock: bool = False  # 是否使用模拟重排序（没有GPU时用）

    # ─────────────────────────────────────────────
    # 可控 Self-RAG 配置
    # 作用：RAG（检索增强生成）是一种让 AI 回答更准确的技术
    # "可控"意味着我们可以精确控制什么时候检索、检索什么内容
    # 这个配置包含三个核心功能：
    # 1. 检索路由：判断用户问题是否需要从知识库检索
    # 2. 有用判断：用模型评估检索结果是否有用
    # 3. 事实校验：检查回答中的数字、日期等是否和知识库一致
    # ─────────────────────────────────────────────
    controlled_self_rag_enabled: bool = True  # 是否启用可控 Self-RAG
    
    # 必须检索的关键词（命中任一即检索）
    # 比如用户问"请假怎么申请"，肯定需要检索知识库
    retrieval_must_keywords: List[str] = [
        "故障码", "错误码", "Error", "exception",  # 技术问题
        "发票", "报销", "金额", "标准",  # 财务问题
        "接口", "API", "service",  # 技术术语
        "请假", "年假", "调休", "加班",  # HR问题
        "密码", "权限", "账号", "开通",  # IT问题
        "制度", "规定", "政策", "流程",  # 政策问题
        "怎么", "如何", "怎么办", "怎么操作",  # 操作类问题
        "步骤", "教程", "指南",  # 教程类
    ]
    
    # 明确不检索的关键词（命中任一则跳过检索）
    # 比如用户只是打个招呼，不需要检索
    retrieval_skip_keywords: List[str] = [
        "你好", "您好", "嗨", "hi", "hello",  # 问候语
        "谢谢", "感谢", "辛苦了",  # 礼貌用语
        "在吗", "在不在", "有人吗",  # 确认类
        "随便问问", "没事", "没什么",  # 闲聊类
    ]
    
    # LLM 判断阈值（当问题既不在必须检索也不在不检索列表时）
    retrieval_llm_threshold: float = 0.5  # AI 判断的模糊度阈值（0-1）
    
    # BGE-Rerank 评分阈值
    rerank_score_threshold: float = 0.3  # 低于这个分数的结果会被过滤掉
    rerank_min_results: int = 1  # 最少保留几个结果

    # ─────────────────────────────────────────────
    # Celery 异步任务队列配置
    # 作用：Celery 是一个任务队列，用于处理耗时的后台任务
    # 比如用户提交一个申请，不需要等待处理完成
    # ─────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/0"  # 消息队列地址
    celery_result_expires: int = 86400  # 任务结果过期时间（秒），24小时
    celery_task_acks_late: bool = True  # 任务完成后才确认，防止中途失败丢任务
    celery_worker_prefetch_multiplier: int = 1  # 预取任务数量
    celery_async_enabled: bool = False  # 是否启用异步模式
    celery_sse_streaming_enabled: bool = True  # 是否支持实时推送
    celery_sse_streaming_chunk_size: int = 15  # 每次推送多少字符

    # ─────────────────────────────────────────────
    # Redis 内存存储配置
    # 作用：存储短期记忆、会话状态等
    # ─────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"  # Redis 连接地址
    redis_socket_timeout: float = 5.0  # Socket 超时时间（秒）
    redis_session_ttl: int = 1800  # 会话 TTL（秒），默认 30 分钟

    # ─────────────────────────────────────────────
    # 多模态 AI 配置
    # 作用：支持处理图片、语音等多种类型的数据
    # ─────────────────────────────────────────────
    multimodal_enabled: bool = True  # 是否启用多模态
    multimodal_provider: str = "qwen"  # 模型提供商：openai/qwen/claude/mock
    multimodal_model: str = "Qwen/Qwen2.5-VL-32B-A03"  # 具体模型名称（vLLM 部署的 Qwen3.5-VL）
    multimodal_base_url: str = "http://localhost:8000/v1"  # vLLM 多模态服务地址
    multimodal_api_key: str = "not-needed"  # vLLM 内网部署通常不需要 key
    multimodal_request_timeout: int = 120  # 请求超时时间（秒）
    multimodal_max_tokens: int = 2048  # 最大输出 token

    # ─────────────────────────────────────────────
    # 应用全局配置
    # ─────────────────────────────────────────────
    app_name: str = "OA Smart Agent"  # 应用名称
    debug: bool = False  # 是否调试模式
    environment: str = "production"  # 环境：dev（开发）/staging（预发布）/production（生产）
    app_version: str = "1.0.0"  # 版本号

    class Config:
        env_file = ".env"  # 从 .env 文件读取环境变量
        env_file_encoding = "utf-8"  # 文件编码
        extra = "ignore"  # 忽略额外配置项


# 全局配置单例
# 使用单例模式确保整个应用使用同一份配置
settings = Settings()
