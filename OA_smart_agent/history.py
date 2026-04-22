# -*- coding: utf-8 -*-
"""
对话历史持久化模块 (history.py)
================================================================================

【模块功能说明】
这个模块负责将用户与AI的对话历史保存到数据库中。就像学校的"考勤记录本"一样，
系统会记录每一个用户的每一次提问和AI的每一次回答，以便日后查看、审计和分析。

【为什么需要这个模块？】
1. 合规审计：公司和法规要求保留对话记录，以备检查
2. 问题追踪：当出现问题时，可以通过历史记录追溯发生了什么
3. 服务优化：通过分析历史对话，改进AI的回答质量
4. 用户体验：用户可以查看自己之前的提问和回答

【工作原理】
1. 当用户发起一个新对话时，系统会分配一个唯一的"会话ID"
2. 用户说的每一句话、AI的每一个回答都会被记录下来
3. 这些记录会异步（不等待）保存到数据库，不影响AI响应速度
4. 如果数据库暂时不可用，会降级为保存到本地JSON文件

【数据库支持】
支持 PostgreSQL 和 MySQL 两种数据库，通过配置文件选择使用哪种。
"""

# =============================================================================
# 导入必要的库
# =============================================================================

import logging  # 日志模块，用于记录程序运行信息
import json  # JSON格式处理，用于数据转换
from typing import Optional, List  # 类型提示，提高代码可读性
from datetime import datetime, timezone  # 日期时间处理
from enum import Enum  # 枚举类型，定义固定常量
from dataclasses import dataclass, asdict  # 数据类，简化数据结构定义

# 从其他模块导入数据模型
from schemas import AgentState

# 创建日志记录器，用于记录本模块的运行日志
logger = logging.getLogger("oa_agent.history")


# =============================================================================
# 数据模型定义
# =============================================================================

class SessionStatus(str, Enum):
    """
    会话状态枚举
    
    【什么是枚举？】
    枚举就像是一个"可选值列表"，每个值都有特定含义。
    比如一个灯的状态只能是：开、关、闪烁。
    这里定义了对话会话可能处于的几种状态。
    """
    
    # 进行中：用户正在使用，系统还没有决定要做什么
    ACTIVE = "active"
    
    # 已完成：AI已经回答了用户的问题，对话正常结束
    COMPLETED = "completed"
    
    # 等待审批：这个操作需要人工审批（比如大额报销、请假多天等）
    APPROVAL_PENDING = "approval_pending"
    
    # 已过期：会话超过了有效期（比如30分钟没操作）
    EXPIRED = "expired"
    
    # 已取消：用户主动取消了操作，或者审批被拒绝
    CANCELLED = "cancelled"


@dataclass
class ConversationHistory:
    """
    对话历史记录的数据结构
    
    【什么是数据类？】
    数据类就像是一个"数据表格"，定义了需要保存哪些信息。
    每一行记录包含以下字段。
    
    【对应数据库表】conversation_history
    
    【字段说明】
    """
    # 记录的ID号，数据库自动生成，类似Excel的行号
    id: Optional[int] = None
    
    # 会话唯一标识，就像学生的学号，用于区分不同对话
    session_id: str = ""
    
    # 用户ID，记录是谁在提问
    user_id: str = ""
    
    # 用户的问题原始文本
    user_query: str = ""
    
    # AI理解的意图，比如"请假"、"查工资"等
    intent: str = ""
    
    # AI的最终回答
    final_response: str = ""
    
    # 执行的动作类型，比如"create_leave_request"（创建请假单）
    action_type: Optional[str] = None
    
    # 执行动作时使用的参数，比如请假需要：日期、原因、天数
    action_params: Optional[dict] = None
    
    # 审批状态：pending（待审批）、approved（已批准）、rejected（已拒绝）
    approval_status: Optional[str] = None
    
    # 当前会话状态，使用上面定义的枚举
    status: str = SessionStatus.ACTIVE.value
    
    # 记录创建时间
    created_at: Optional[datetime] = None
    
    # 最后更新时间
    updated_at: Optional[datetime] = None
    
    # 其他附加信息，用字典保存，方便扩展
    metadata: Optional[dict] = None


# =============================================================================
# 数据库连接管理
# =============================================================================

class HistoryDB:
    """
    对话历史数据库操作类
    
    【什么是类？】
    类就像是一个"工厂"，可以生产产品（实例化对象）。
    这个类专门负责与数据库打交道：连接、查询、保存数据。
    
    【主要功能】
    - 连接数据库
    - 保存对话记录
    - 查询历史记录
    - 更新记录状态
    - 清理过期记录
    
    【降级机制】
    如果配置文件没有设置数据库，系统不会报错，
    而是降级为保存到本地文件（JSON格式），确保数据不丢失。
    """

    def __init__(self):
        """
        初始化数据库连接对象
        
        【为什么用延迟初始化？】
        程序启动时可能还没有配置好数据库，所以先创建"空壳子"，
        等到真正需要时才连接数据库。
        """
        self._engine = None  # 数据库引擎，用于执行SQL命令
        self._session_factory = None  # 会话工厂，用于创建数据库会话
        self._initialized = False  # 标记是否已完成初始化

    def initialize(self):
        """
        初始化数据库连接
        
        【初始化流程】
        1. 检查是否配置了数据库
        2. 根据数据库类型选择合适的驱动
        3. 创建连接池
        4. 测试连接是否成功
        """
        # 检查是否启用了历史保存功能
        if not settings.history_save_async or not settings.history_db_url:
            logger.info(
                "history_db_disabled",
                extra={"reason": "history_db_url not configured or async disabled"}
            )
            return

        try:
            # 根据配置的数据库类型选择驱动
            # PostgreSQL 和 MySQL 的连接方式略有不同
            
            if settings.history_db_type == "postgresql":
                # PostgreSQL 数据库
                try:
                    # 导入异步PostgreSQL驱动
                    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
                    
                    # 将普通URL转换为异步URL（需要添加asyncpg驱动）
                    url = settings.history_db_url.replace(
                        "postgresql://", "postgresql+asyncpg://"
                    )
                except ImportError:
                    # 如果没有安装asyncpg，提示用户
                    logger.warning("asyncpg_not_installed", extra={"component": "history"})
                    return

            elif settings.history_db_type == "mysql":
                # MySQL 数据库
                try:
                    # 导入异步MySQL驱动
                    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
                    
                    # 将普通URL转换为异步URL（需要添加aiomysql驱动）
                    url = settings.history_db_url.replace(
                        "mysql://", "mysql+aiomysql://"
                    )
                except ImportError:
                    # 如果没有安装aiomysql，提示用户
                    logger.warning("aiomysql_not_installed", extra={"component": "history"})
                    return
            else:
                # 不支持的数据库类型
                logger.warning(
                    "history_db_unsupported_type",
                    extra={"db_type": settings.history_db_type}
                )
                return

            # 创建数据库引擎（连接池）
            self._engine = create_async_engine(
                url,
                pool_size=settings.history_db_pool_size,  # 连接池大小
                max_overflow=settings.history_db_max_overflow,  # 最大溢出连接数
                pool_pre_ping=True,  # 使用前先测试连接是否有效
                echo=False,  # 不打印SQL语句
            )
            
            # 设置会话工厂
            self._session_factory = AsyncSession
            
            # 标记初始化成功
            self._initialized = True
            
            logger.info(
                "history_db_initialized",
                extra={
                    "db_type": settings.history_db_type,
                    "component": "history",
                }
            )

        except Exception as e:
            # 如果初始化失败，记录错误日志
            logger.error(
                "history_db_init_failed",
                extra={
                    "error": str(e),
                    "component": "history",
                }
            )
            self._initialized = False

    @property
    def is_available(self) -> bool:
        """
        检查数据库是否可用
        
        【什么是属性？】
        通过 @property 装饰器，可以像访问变量一样调用方法。
        例如：db.is_available 而不是 db.is_available()
        """
        return self._initialized and self._engine is not None


# =============================================================================
# 全局数据库实例管理
# =============================================================================

# 全局变量，用于存储数据库实例（单例模式）
_history_db: Optional[HistoryDB] = None


def get_history_db() -> HistoryDB:
    """
    获取 HistoryDB 单例
    
    【什么是单例模式？】
    单例就是"只有一个实例"。不管调用多少次，获取到的都是同一个数据库对象。
    这样可以避免创建多个数据库连接，节省资源。
    """
    global _history_db
    if _history_db is None:
        _history_db = HistoryDB()
        _history_db.initialize()
    return _history_db


# =============================================================================
# 历史记录保存和更新函数
# =============================================================================

async def save_conversation_history(
    session_id: str,
    user_id: str,
    state: AgentState,
    status: str = SessionStatus.ACTIVE.value,
) -> bool:
    """
    异步保存对话历史记录
    
    【什么是异步？】
    异步就像"委托"。你把任务交给别人去做，然后自己去忙别的事。
    等别人做完了再通知你。这样不会阻塞主流程，用户感觉更快。
    
    【为什么要保存历史？】
    - 合规要求：某些行业必须保留通信记录
    - 问题排查：当出现问题时，可以追溯历史
    - 机器学习：利用历史数据训练更好的AI模型
    
    【参数说明】
    - session_id: 会话ID，区分不同对话
    - user_id: 用户ID，知道是谁在提问
    - state: LangGraph的状态对象，包含完整的对话信息
    - status: 会话状态
    
    【返回值】
    - True: 保存成功
    - False: 保存失败
    """
    # 获取数据库连接
    db = get_history_db()

    # 如果数据库不可用，使用降级方案
    if not db.is_available:
        _log_to_json_file(session_id, user_id, state, status)
        return False

    try:
        # 从state中提取对话内容
        messages = state.get("messages", [])
        
        # 找到用户最后说的话
        user_query = ""
        for msg in messages:
            # 如果是用户消息（human类型），保存其内容
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content
                break

        # 构建历史记录对象
        record = ConversationHistory(
            session_id=session_id,
            user_id=user_id,
            user_query=user_query,
            intent=state.get("intent", ""),
            final_response=state.get("final_response", ""),
            action_type=state.get("action_payload", {}).get("action_type"),
            action_params=state.get("action_payload", {}).get("params"),
            approval_status="pending" if state.get("requires_approval") else None,
            status=status,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metadata={
                "retrieved_docs_count": len(state.get("retrieved_docs") or []),
                "requires_approval": state.get("requires_approval", False),
            }
        )

        # 打开数据库连接，执行保存操作
        async with db._session_factory(bind=db._engine) as session:
            session.add(record)  # 添加记录到会话
            await session.commit()  # 提交事务，真正写入数据库

        logger.info(
            "history_saved",
            extra={
                "session_id": session_id,
                "user_id": user_id,
                "intent": record.intent,
                "component": "history",
            }
        )
        return True

    except Exception as e:
        # 保存失败，记录错误日志
        logger.error(
            "history_save_failed",
            extra={
                "session_id": session_id,
                "error": str(e),
                "component": "history",
            }
        )
        return False


async def update_session_status(
    session_id: str,
    status: str,
    final_response: Optional[str] = None,
) -> bool:
    """
    更新会话状态
    
    【使用场景】
    1. 用户提交了一个请假申请 → 状态变为"等待审批"
    2. 经理批准了请假 → 状态变为"已完成"
    3. 经理拒绝了请假 → 状态变为"已取消"
    4. 会话超时无操作 → 状态变为"已过期"
    
    【参数说明】
    - session_id: 要更新的会话ID
    - status: 新的状态值
    - final_response: 可选的最终回答（用于审批通过时更新）
    """
    db = get_history_db()

    # 如果数据库不可用，记录警告
    if not db.is_available:
        logger.warning(
            "history_update_skipped",
            extra={"session_id": session_id, "reason": "db unavailable"}
        )
        return False

    try:
        # 导入SQLAlchemy的更新语句
        from sqlalchemy import update, text
        
        # 执行数据库更新操作
        async with db._session_factory(bind=db._engine) as session:
            # 构建更新语句
            stmt = (
                update(ConversationHistory)
                .where(ConversationHistory.session_id == session_id)
                .values(
                    status=status,
                    final_response=final_response,
                    updated_at=datetime.now(timezone.utc)
                )
            )
            await session.execute(stmt)
            await session.commit()

        logger.info(
            "history_status_updated",
            extra={
                "session_id": session_id,
                "status": status,
                "component": "history",
            }
        )
        return True

    except Exception as e:
        logger.error(
            "history_update_failed",
            extra={
                "session_id": session_id,
                "error": str(e),
                "component": "history",
            }
        )
        return False


# =============================================================================
# 降级方案：JSON文件保存
# =============================================================================

def _log_to_json_file(session_id: str, user_id: str, state: AgentState, status: str):
    """
    降级方案：将历史记录写入JSON文件
    
    【什么时候使用？】
    当数据库不可用时（比如数据库正在维护、网络断开了），
    系统不会丢失数据，而是降级为保存到本地文件。
    
    【文件保存位置】
    ./data/history/{user_id}/{session_id}.json
    
    【例如】
    ./data/history/zhangsan/abc123.json
    这表示用户zhangsan的会话abc123的记录
    """
    import os, json
    
    # 构建保存目录路径
    history_dir = os.path.join(
        os.path.dirname(__file__),  # 当前文件所在目录
        "data",  # data子目录
        "history",  # history子目录
        user_id  # 以用户名命名的子目录
    )
    
    # 创建目录（如果不存在）
    os.makedirs(history_dir, exist_ok=True)

    # 构建文件路径
    filepath = os.path.join(history_dir, f"{session_id}.json")
    
    try:
        # 打开文件并写入JSON数据
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "status": status,
                    "state": {
                        "intent": state.get("intent"),
                        "final_response": state.get("final_response"),
                        "action_type": state.get("action_payload", {}).get("action_type"),
                        "requires_approval": state.get("requires_approval"),
                    },
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                ensure_ascii=False,  # 支持中文显示
                indent=2,  # 格式化缩进
            )
    except Exception:
        # 如果写入失败，静默处理（避免产生更多错误）
        pass
