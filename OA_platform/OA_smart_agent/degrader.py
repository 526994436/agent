# -*- coding: utf-8 -*-
"""
降级策略模块 (degrader.py)
================================================================================

【模块功能说明】
这个模块是系统的"安全网"，当AI服务出现问题时，保证用户仍然能得到回应。



# =============================================================================
# 导入必要的库
# =============================================================================

import logging  # 日志模块，记录降级状态变化
from typing import Dict, Any, Optional, Callable  # 类型提示
from enum import Enum  # 枚举类型
from dataclasses import dataclass  # 数据类

# 创建降级模块专用的日志记录器
logger = logging.getLogger("oa_agent.degrader")


# =============================================================================
# 降级级别定义
# =============================================================================

class DegradeLevel(str, Enum):
    """
    降级级别枚举
    """

    # 正常模式：所有组件都正常工作
    NORMAL = "normal"

    # 降级模式：服务不可用，返回固定话语
    DEGRADED = "degraded"


class DegradeStatus:
    """
    降级状态数据类
    """

    # 当前降级级别
    level: DegradeLevel

    # 知识库是否可用
    rag_available: bool

    # AI模型是否可用
    llm_available: bool

    # Java后端是否可用
    java_backend_available: bool

    # 当前激活的层级（1=正常, 2=降级）
    active_layer: int

    # 连续失败次数
    failure_count: int

    # 最后一次失败时间
    last_failure_time: Optional[float]

    # 最后一次恢复时间
    last_recovery_time: Optional[float]

    # 给用户的提示信息
    message: str


# =============================================================================
# 健康状态检查器
# =============================================================================

class HealthChecker:
    """
    健康状态检查器
    """

    def __init__(self):
        """初始化健康检查器"""
        self._rag_healthy = True
        self._llm_healthy = True
        self._java_healthy = True
        self._current_level = DegradeLevel.NORMAL
        self._failure_count = 0
        self._recovery_threshold = 3

    def check_rag_health(self) -> bool:
        """检查知识库(RAG)组件健康状态"""
        try:
            from rag import build_hybrid_retriever
            retriever = build_hybrid_retriever()
            docs = retriever.retrieve("测试", top_k=1)
            return True
        except Exception as e:
            logger.warning("rag_health_check_failed", extra={"error": str(e), "component": "degrader"})
            return False

    def check_llm_health(self) -> bool:
        """检查AI模型(LLM)组件健康状态"""
        try:
            from graph import llm_with_timeout
            from langchain_core.messages import HumanMessage
            response = llm_with_timeout([HumanMessage(content="test")])
            return True
        except Exception as e:
            logger.warning("llm_health_check_failed", extra={"error": str(e), "component": "degrader"})
            return False

    def report_rag_failure(self):
        """报告RAG失败"""
        self._failure_count += 1
        if self._failure_count >= 1:
            self._current_level = DegradeLevel.DEGRADED
            logger.warning("rag_degraded", extra={"failure_count": self._failure_count, "component": "degrader"})

    def report_rag_success(self):
        """报告RAG恢复"""
        self._failure_count = 0
        self._current_level = DegradeLevel.NORMAL
        logger.info("rag_recovered", extra={"component": "degrader"})

    def report_llm_failure(self):
        """报告LLM失败"""
        self._failure_count += 1
        if self._failure_count >= 1:
            self._current_level = DegradeLevel.DEGRADED
            logger.error("llm_degraded", extra={"failure_count": self._failure_count, "component": "degrader"})

    def report_llm_success(self):
        """报告LLM恢复"""
        self._failure_count = 0
        self._current_level = DegradeLevel.NORMAL
        logger.info("llm_recovered", extra={"component": "degrader"})

    def get_status(self) -> DegradeStatus:
        """获取当前降级状态"""
        active_layer = 1 if self._current_level == DegradeLevel.NORMAL else 2
        message = "服务正常" if self._current_level == DegradeLevel.NORMAL else "⚠️ 当前服务不可用，请稍后再试"

        return DegradeStatus(
            level=self._current_level,
            rag_available=self._rag_healthy,
            llm_available=self._llm_healthy,
            java_backend_available=self._java_healthy,
            active_layer=active_layer,
            failure_count=self._failure_count,
            last_failure_time=None,
            last_recovery_time=None,
            message=message,
        )


# =============================================================================
# 全局健康检查器
# =============================================================================

# 全局变量，存储健康检查器实例
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """获取健康检查器单例"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


# =============================================================================
# 降级策略执行器
# =============================================================================

class DegradeStrategy:
    """降级策略执行器"""

    def __init__(self, health_checker: Optional[HealthChecker] = None):
        self.health_checker = health_checker or get_health_checker()

    def execute_policy_query(self, query: str) -> Dict[str, Any]:
        """执行政策查询（带降级策略）"""
        status = self.health_checker.get_status()

        if status.level == DegradeLevel.NORMAL:
            return self._layer_normal(query)
        else:
            return self._layer_degraded()

    def _layer_normal(self, query: str) -> Dict[str, Any]:
        """正常模式：使用 Self-RAG"""
        try:
            from controlled_self_rag import get_controlled_self_rag

            self_rag = get_controlled_self_rag()
            result = self_rag.process(query=query, user_id="system")

            if result.needs_escalation:
                self.health_checker.report_rag_failure()
            else:
                self.health_checker.report_rag_success()

            return {
                "layer": 1,
                "level": "normal",
                "answer": result.answer or "您好，有什么可以帮您？",
                "references": [doc.get("text", "") for doc in (result.docs or [])[:3]],
                "confidence": "high",
                "degraded_note": "",
            }
        except Exception as e:
            logger.error("layer_normal_failed", extra={"error": str(e), "query": query[:30], "component": "degrader"})
            self.health_checker.report_rag_failure()
            return self._layer_degraded()

    def _layer_degraded(self) -> Dict[str, Any]:
        """降级模式：返回固定话语"""
        return {
            "layer": 2,
            "level": "degraded",
            "answer": "抱歉，当前服务暂时不可用，请稍后再试或联系管理员处理。",
            "references": [],
            "confidence": "low",
            "degraded_note": "⚠️ 服务降级中",
        }


# =============================================================================
# 全局降级策略执行器
# =============================================================================

# 全局变量，存储策略执行器实例
_degrade_strategy: Optional[DegradeStrategy] = None


def get_degrade_strategy() -> DegradeStrategy:
    """获取降级策略执行器单例"""
    global _degrade_strategy
    if _degrade_strategy is None:
        _degrade_strategy = DegradeStrategy()
    return _degrade_strategy
