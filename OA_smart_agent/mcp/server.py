"""
MCP Server - Java 后端 API 调用封装

使用 FastMCP 实现标准化工具调用，替代原来的 HTTP 直连方式。

提供的工具（Tools）：
- leave_request: 请假申请
- expense_reimburse: 费用报销
- password_reset: 密码重置
- permission_open: 权限开通
"""

import time
import uuid
from typing import Any, Callable, Optional

from fastmcp import FastMCP

from ..config import settings
from ..utils.logging import logger

# 导入指标记录
try:
    from ..metrics import record_java_api_call
except ImportError:
    record_java_api_call = None


# ─────────────────────────────────────────────
# MCP Server 实例
# ─────────────────────────────────────────────
mcp = FastMCP(
    name="JavaBackendTools",
    description="企业 OA 系统 Java 后端 API 调用工具",
    dependencies=["httpx"],
)


# ─────────────────────────────────────────────
# 断路器实现（保留原有容错机制）
# ─────────────────────────────────────────────
class CircuitBreaker:
    """
    断路器模式实现（带指标记录）。

    状态流转：
    CLOSED（正常）→ 失败次数超过阈值 → OPEN（熔断）→ 冷却期后 → HALF_OPEN（试探）→ 成功 → CLOSED
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, backend_name: str = "java_backend"):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._backend_name = backend_name

    def _get_state_value(self) -> int:
        """将状态字符串转换为数字（用于 Prometheus Gauge）"""
        state_map = {"CLOSED": 0, "OPEN": 1, "HALF_OPEN": 2}
        return state_map.get(self.state, 0)

    def _update_circuit_breaker_metric(self):
        """更新断路器状态指标"""
        try:
            from ..metrics import set_circuit_breaker_state
            set_circuit_breaker_state(self._backend_name, self._get_state_value())
        except Exception:
            pass

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """通过断路器执行函数。"""
        if self.state == "OPEN":
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time > self.recovery_timeout
            ):
                self.state = "HALF_OPEN"
                self._update_circuit_breaker_metric()
                logger.info("circuit_breaker_half_open", extra={"component": "mcp_java_backend"})
            else:
                self._update_circuit_breaker_metric()
                raise RuntimeError(
                    f"Java 后端断路器已打开（OPEN），"
                    f"请等待 {self.recovery_timeout} 秒后重试。"
                )

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                self._update_circuit_breaker_metric()
                logger.info("circuit_breaker_closed", extra={"component": "mcp_java_backend"})
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self._update_circuit_breaker_metric()
                logger.error(
                    "circuit_breaker_opened",
                    extra={
                        "failure_count": self.failure_count,
                        "component": "mcp_java_backend",
                    },
                )
            raise


# 全局断路器实例
_java_circuit_breaker = CircuitBreaker(
    failure_threshold=settings.java_backend_max_retries,
    recovery_timeout=settings.java_backend_circuit_breaker_timeout,
)


# ─────────────────────────────────────────────
# HTTP 调用（带重试）
# ─────────────────────────────────────────────
async def _http_call_with_retry(url: str, payload: dict, headers: dict) -> dict:
    """
    带重试的 HTTP 调用（指数退避）。

    P0 保障：
    - 最多重试 settings.java_backend_max_retries 次
    - 超时使用 settings.java_backend_timeout
    - 5xx 错误和超时重试，4xx 错误直接失败
    """
    import asyncio
    import httpx

    async def _do_request():
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,
                read=float(settings.java_backend_timeout),
                write=5.0,
                pool=10.0,
            ),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        ) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code >= 500:
                response.raise_for_status()
            return response.json()

    attempt = 0
    max_attempts = settings.java_backend_max_retries + 1

    while attempt < max_attempts:
        try:
            return await _do_request()
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                raise
            attempt += 1
            if attempt >= max_attempts:
                raise
            wait_seconds = min(2 ** attempt, 16)
            logger.warning(
                "mcp_java_backend_retry",
                extra={
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "wait_seconds": wait_seconds,
                    "error": str(e),
                },
            )
            await asyncio.sleep(wait_seconds)

    raise RuntimeError("重试次数耗尽")


def _call_java_api(action_type: str, func: Callable) -> Any:
    """
    调用 Java API 并记录指标的统一包装函数

    Args:
        action_type: API 类型（如 leave_request, expense_reimburse）
        func: 实际执行 HTTP 调用的函数

    Returns:
        API 调用结果
    """
    start_time = time.perf_counter()
    success = True

    try:
        result = _java_circuit_breaker.call(func)
        return result
    except RuntimeError as e:
        success = False
        raise
    finally:
        duration = time.perf_counter() - start_time
        if record_java_api_call:
            record_java_api_call(
                action_type=action_type,
                duration=duration,
                success=success,
            )


# ─────────────────────────────────────────────
# MCP Tools 定义
# ─────────────────────────────────────────────
@mcp.tool()
async def leave_request(
    start_date: str,
    end_date: str,
    leave_type: str,
    reason: str = "",
    user_token: str = "",
) -> dict:
    """
    提交请假申请。

    参数:
        start_date: 开始日期（YYYY-MM-DD 格式）
        end_date: 结束日期（YYYY-MM-DD 格式）
        leave_type: 请假类型（年假/病假/事假/婚假/产假/丧假）
        reason: 请假原因（可选）
        user_token: 用户认证令牌
    """
    import httpx

    request_id = str(uuid.uuid4())
    payload = {
        "action_type": "leave_request",
        "params": {
            "start_date": start_date,
            "end_date": end_date,
            "leave_type": leave_type,
            "reason": reason,
        },
        "request_id": request_id,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    headers = {
        "Authorization": f"Bearer {user_token}",
        "Content-Type": "application/json",
        "X-Request-ID": request_id,
        "X-Source": "oa-smart-agent-mcp",
    }

    url = f"{settings.java_backend_base_url}/api/leave/submit"

    def _do_call():
        import asyncio
        return asyncio.run(_http_call_with_retry(url, payload, headers))

    try:
        result = _call_java_api("leave_request", _do_call)
        return {
            "success": True,
            "request_id": request_id,
            "message": result.get("message", "请假申请已提交"),
            "data": result,
        }
    except RuntimeError as e:
        logger.error("mcp_leave_request_failed", extra={"error": str(e)})
        return {
            "success": False,
            "request_id": request_id,
            "message": str(e),
        }


@mcp.tool()
async def expense_reimburse(
    expense_type: str,
    amount: float,
    description: str,
    invoice_no: str = "",
    user_token: str = "",
) -> dict:
    """
    提交费用报销申请。

    参数:
        expense_type: 费用类型（差旅费/交通费/餐饮费/办公费/通讯费/其他）
        amount: 报销金额（元）
        description: 费用说明
        invoice_no: 发票号（可选）
        user_token: 用户认证令牌
    """
    request_id = str(uuid.uuid4())
    payload = {
        "action_type": "expense_reimburse",
        "params": {
            "expense_type": expense_type,
            "amount": amount,
            "description": description,
            "invoice_no": invoice_no,
        },
        "request_id": request_id,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    headers = {
        "Authorization": f"Bearer {user_token}",
        "Content-Type": "application/json",
        "X-Request-ID": request_id,
        "X-Source": "oa-smart-agent-mcp",
    }

    url = f"{settings.java_backend_base_url}/api/expense/submit"

    def _do_call():
        import asyncio
        return asyncio.run(_http_call_with_retry(url, payload, headers))

    try:
        result = _call_java_api("expense_reimburse", _do_call)
        return {
            "success": True,
            "request_id": request_id,
            "message": result.get("message", "报销申请已提交"),
            "data": result,
        }
    except RuntimeError as e:
        logger.error("mcp_expense_reimburse_failed", extra={"error": str(e)})
        return {
            "success": False,
            "request_id": request_id,
            "message": str(e),
        }


@mcp.tool()
async def password_reset(
    system_name: str,
    user_token: str = "",
) -> dict:
    """
    申请密码重置。

    参数:
        system_name: 系统名称（OA/邮箱/ERP/CRM/其他）
        user_token: 用户认证令牌
    """
    request_id = str(uuid.uuid4())
    payload = {
        "action_type": "password_reset",
        "params": {
            "system_name": system_name,
        },
        "request_id": request_id,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    headers = {
        "Authorization": f"Bearer {user_token}",
        "Content-Type": "application/json",
        "X-Request-ID": request_id,
        "X-Source": "oa-smart-agent-mcp",
    }

    url = f"{settings.java_backend_base_url}/api/password/reset"

    def _do_call():
        import asyncio
        return asyncio.run(_http_call_with_retry(url, payload, headers))

    try:
        result = _call_java_api("password_reset", _do_call)
        return {
            "success": True,
            "request_id": request_id,
            "message": result.get("message", "密码重置申请已提交"),
            "data": result,
        }
    except RuntimeError as e:
        logger.error("mcp_password_reset_failed", extra={"error": str(e)})
        return {
            "success": False,
            "request_id": request_id,
            "message": str(e),
        }


@mcp.tool()
async def permission_open(
    system_name: str,
    permission_level: str = "普通",
    reason: str = "",
    user_token: str = "",
) -> dict:
    """
    申请开通系统权限。

    参数:
        system_name: 系统名称（OA/邮箱/ERP/CRM/财务系统/其他）
        permission_level: 权限级别（普通/管理员/超级管理员）
        reason: 申请原因（可选）
        user_token: 用户认证令牌
    """
    request_id = str(uuid.uuid4())
    payload = {
        "action_type": "permission_open",
        "params": {
            "system_name": system_name,
            "permission_level": permission_level,
            "reason": reason,
        },
        "request_id": request_id,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    headers = {
        "Authorization": f"Bearer {user_token}",
        "Content-Type": "application/json",
        "X-Request-ID": request_id,
        "X-Source": "oa-smart-agent-mcp",
    }

    url = f"{settings.java_backend_base_url}/api/permission/open"

    def _do_call():
        import asyncio
        return asyncio.run(_http_call_with_retry(url, payload, headers))

    try:
        result = _call_java_api("permission_open", _do_call)
        return {
            "success": True,
            "request_id": request_id,
            "message": result.get("message", "权限开通申请已提交"),
            "data": result,
        }
    except RuntimeError as e:
        logger.error("mcp_permission_open_failed", extra={"error": str(e)})
        return {
            "success": False,
            "request_id": request_id,
            "message": str(e),
        }


# ─────────────────────────────────────────────
# 通用执行接口（保留兼容性）
# ─────────────────────────────────────────────
@mcp.tool()
async def execute_action(
    action_type: str,
    params: dict,
    user_token: str = "",
) -> dict:
    """
    通用动作执行接口。

    参数:
        action_type: 动作类型（leave_request/expense_reimburse/password_reset/permission_open/其他）
        params: 动作参数字典
        user_token: 用户认证令牌
    """
    request_id = str(uuid.uuid4())
    payload = {
        "action_type": action_type,
        "params": params,
        "request_id": request_id,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    headers = {
        "Authorization": f"Bearer {user_token}",
        "Content-Type": "application/json",
        "X-Request-ID": request_id,
        "X-Source": "oa-smart-agent-mcp",
    }

    url = f"{settings.java_backend_base_url}/mock/java/api/execute"

    def _do_call():
        import asyncio
        return asyncio.run(_http_call_with_retry(url, payload, headers))

    try:
        result = _call_java_api(action_type, _do_call)
        return {
            "success": True,
            "request_id": request_id,
            "message": result.get("message", "操作已提交"),
            "data": result,
        }
    except RuntimeError as e:
        logger.error("mcp_execute_action_failed", extra={"error": str(e)})
        return {
            "success": False,
            "request_id": request_id,
            "message": str(e),
        }


# ─────────────────────────────────────────────
# RAG 检索工具
# ─────────────────────────────────────────────
@mcp.tool()
async def rag_retrieve(
    query: str,
    top_k: int = 5,
) -> dict:
    """
    检索企业知识库，获取与问题最相关的文档内容。

    使用场景：
    - 用户询问政策、制度、流程等问题时
    - 需要查阅公司文档才能回答的问题
    - 如"年假怎么休"、"报销标准是什么"等

    参数:
        query: 用户的查询问题
        top_k: 返回的最相关文档数量，默认5篇

    返回:
        包含检索到的文档列表和生成的回答
    """
    try:
        from controlled_self_rag import get_controlled_self_rag
        
        self_rag = get_controlled_self_rag()
        
        result = self_rag.process(
            query=query,
            user_id="default_user",
            user_dept=None,
            user_projects=None,
        )
        
        if not result.is_useful or not result.docs:
            return {
                "success": True,
                "is_useful": False,
                "answer": "在知识库中未找到相关信息，建议联系 HR 部门（内线 8001）获取帮助。",
                "docs": [],
                "message": "检索结果无用",
            }
        
        # 返回成功结果
        return {
            "success": True,
            "is_useful": True,
            "answer": result.answer,
            "docs": [
                {
                    "content": doc.get("text", "")[:500],  # 限制长度
                    "score": doc.get("score", 0),
                    "metadata": doc.get("metadata", {}),
                }
                for doc in (result.docs or [])[:top_k]
            ],
            "quality_score": result.usefulness_score,
            "needs_escalation": result.needs_escalation,
            "message": f"成功检索到 {len(result.docs)} 篇相关文档",
        }
        
    except Exception as e:
        logger.error(f"RAG 检索失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "answer": "系统暂时无法处理您的请求，请稍后重试或联系 HR（内线 8001）获取帮助。",
            "docs": [],
        }


# ─────────────────────────────────────────────
# 工具列表导出（方便调试和查看）
# ─────────────────────────────────────────────
java_backend_tools = [
    {
        "name": "rag_retrieve",
        "description": "检索企业知识库，获取与问题最相关的文档内容。用于政策咨询、制度查询等场景。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户的查询问题"},
                "top_k": {"type": "integer", "description": "返回的最相关文档数量，默认5篇", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "leave_request",
        "description": "提交请假申请",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "开始日期（YYYY-MM-DD）"},
                "end_date": {"type": "string", "description": "结束日期（YYYY-MM-DD）"},
                "leave_type": {"type": "string", "description": "请假类型"},
                "reason": {"type": "string", "description": "请假原因"},
            },
            "required": ["start_date", "end_date", "leave_type"],
        },
    },
    {
        "name": "expense_reimburse",
        "description": "提交费用报销申请",
        "input_schema": {
            "type": "object",
            "properties": {
                "expense_type": {"type": "string", "description": "费用类型"},
                "amount": {"type": "number", "description": "报销金额"},
                "description": {"type": "string", "description": "费用说明"},
            },
            "required": ["expense_type", "amount", "description"],
        },
    },
    {
        "name": "password_reset",
        "description": "申请密码重置",
        "input_schema": {
            "type": "object",
            "properties": {
                "system_name": {"type": "string", "description": "系统名称"},
            },
            "required": ["system_name"],
        },
    },
    {
        "name": "permission_open",
        "description": "申请开通系统权限",
        "input_schema": {
            "type": "object",
            "properties": {
                "system_name": {"type": "string", "description": "系统名称"},
                "permission_level": {"type": "string", "description": "权限级别"},
                "reason": {"type": "string", "description": "申请原因"},
            },
            "required": ["system_name"],
        },
    },
    {
        "name": "execute_action",
        "description": "通用动作执行接口",
        "input_schema": {
            "type": "object",
            "properties": {
                "action_type": {"type": "string", "description": "动作类型"},
                "params": {"type": "object", "description": "动作参数"},
            },
            "required": ["action_type", "params"],
        },
    },
]
