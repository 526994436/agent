"""
MCP Server - 入职全流程自动化

使用 FastMCP 实现标准化工具调用，覆盖新员工入职全生命周期：
- 入职通知书生成
- OA 账号开通
- IT 设备申请
- 培训课程添加
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

def _build_headers(user_token: str, request_id: str) -> dict:
    return {
        "Authorization": f"Bearer {user_token}",
        "Content-Type": "application/json",
        "X-Request-ID": request_id,
        "X-Source": "oa-smart-agent-mcp",
    }


def _sync_http_call(url: str, payload: dict, headers: dict) -> dict:
    """
    在线程池中执行的同步 HTTP 调用（供 asyncio.to_thread 使用）。
    """
    import asyncio
    return asyncio.run(_http_call_with_retry(url, payload, headers))


async def _http_tool_call(
    url: str,
    payload: dict,
    headers: dict,
    action_type: str,
) -> dict:
    """通用异步 HTTP 工具调用，包装断路器 + 线程池 + 指标记录。"""
    import asyncio

    start_time = time.perf_counter()
    success = True
    try:
        result = await asyncio.to_thread(
            _java_circuit_breaker.call,
            lambda: _sync_http_call(url, payload, headers),
        )
        return {"success": True, "data": result}
    except RuntimeError as e:
        success = False
        logger.error(f"mcp_{action_type}_failed", extra={"error": str(e)})
        return {"success": False, "error": str(e)}
    finally:
        if record_java_api_call:
            record_java_api_call(
                action_type=action_type,
                duration=time.perf_counter() - start_time,
                success=success,
            )


# ─────────────────────────────────────────────
# 入职通知书生成
# ─────────────────────────────────────────────
@mcp.tool()
async def generate_offer_letter(
    candidate_name: str,
    department: str,
    position: str,
    entry_date: str,
    salary: str,
    contract_type: str = "全职",
    probation_period: str = "3个月",
    reporting_manager: str = "",
    reporting_location: str = "",
    user_token: str = "",
) -> dict:
    """
    生成新员工入职通知书。

    参数:
        candidate_name: 候选人姓名
        department: 部门名称
        position: 岗位名称
        entry_date: 入职日期（YYYY-MM-DD）
        salary: 薪资待遇
        contract_type: 合同类型（默认全职）
        probation_period: 试用期（默认3个月）
        reporting_manager: 直属上级（可选）
        reporting_location: 报到地点（可选）
        user_token: 操作人认证令牌
    """
    request_id = str(uuid.uuid4())
    payload = {
        "action_type": "generate_offer_letter",
        "params": {
            "candidate_name": candidate_name,
            "department": department,
            "position": position,
            "entry_date": entry_date,
            "salary": salary,
            "contract_type": contract_type,
            "probation_period": probation_period,
            "reporting_manager": reporting_manager,
            "reporting_location": reporting_location,
        },
        "request_id": request_id,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    headers = _build_headers(user_token, request_id)
    url = f"{settings.java_backend_base_url}/api/onboarding/offer-letter"

    result = await _http_tool_call(url, payload, headers, "generate_offer_letter")
    if result["success"]:
        return {
            "success": True,
            "request_id": request_id,
            "message": "入职通知书已生成",
            "data": result["data"],
        }
    return {
        "success": False,
        "request_id": request_id,
        "message": f"生成失败: {result.get('error', '未知错误')}",
    }


# ─────────────────────────────────────────────
# OA 账号开通
# ─────────────────────────────────────────────
@mcp.tool()
async def create_oa_account(
    employee_name: str,
    employee_id: str,
    department: str,
    position: str,
    email: str,
    mobile: str = "",
    manager_id: str = "",
    user_token: str = "",
) -> dict:
    """
    为新员工开通 OA 系统账号。

    参数:
        employee_name: 员工姓名
        employee_id: 工号
        department: 部门
        position: 岗位
        email: 邮箱地址
        mobile: 手机号（可选）
        manager_id: 直属上级工号（可选）
        user_token: 操作人认证令牌
    """
    request_id = str(uuid.uuid4())
    payload = {
        "action_type": "create_oa_account",
        "params": {
            "employee_name": employee_name,
            "employee_id": employee_id,
            "department": department,
            "position": position,
            "email": email,
            "mobile": mobile,
            "manager_id": manager_id,
        },
        "request_id": request_id,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    headers = _build_headers(user_token, request_id)
    url = f"{settings.java_backend_base_url}/api/onboarding/oa-account"

    result = await _http_tool_call(url, payload, headers, "create_oa_account")
    if result["success"]:
        return {
            "success": True,
            "request_id": request_id,
            "message": "OA 账号开通成功",
            "data": result["data"],
        }
    return {
        "success": False,
        "request_id": request_id,
        "message": f"开通失败: {result.get('error', '未知错误')}",
    }


# ─────────────────────────────────────────────
# IT 设备申请
# ─────────────────────────────────────────────
@mcp.tool()
async def apply_it_equipment(
    employee_id: str,
    employee_name: str,
    department: str,
    device_type: str,
    device_spec: str = "",
    quantity: int = 1,
    reason: str = "",
    user_token: str = "",
) -> dict:
    """
    申请 IT 设备（笔记本电脑、显示器、键鼠等）。

    参数:
        employee_id: 员工工号
        employee_name: 员工姓名
        department: 部门
        device_type: 设备类型（笔记本电脑/显示器/键盘鼠标套装/耳机/其他）
        device_spec: 设备规格描述（可选，如"MacBook Pro 14寸 M3/16GB/512GB"）
        quantity: 数量（默认1）
        reason: 申请理由（可选）
        user_token: 操作人认证令牌
    """
    request_id = str(uuid.uuid4())
    payload = {
        "action_type": "apply_it_equipment",
        "params": {
            "employee_id": employee_id,
            "employee_name": employee_name,
            "department": department,
            "device_type": device_type,
            "device_spec": device_spec,
            "quantity": quantity,
            "reason": reason,
        },
        "request_id": request_id,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    headers = _build_headers(user_token, request_id)
    url = f"{settings.java_backend_base_url}/api/onboarding/it-equipment"

    result = await _http_tool_call(url, payload, headers, "apply_it_equipment")
    if result["success"]:
        return {
            "success": True,
            "request_id": request_id,
            "message": "IT 设备申请已提交",
            "data": result["data"],
        }
    return {
        "success": False,
        "request_id": request_id,
        "message": f"申请失败: {result.get('error', '未知错误')}",
    }


# ─────────────────────────────────────────────
# 培训课程添加
# ─────────────────────────────────────────────
@mcp.tool()
async def enroll_training_courses(
    employee_id: str,
    employee_name: str,
    department: str,
    course_list: list,
    enrollment_type: str = "新员工入职培训",
    deadline_days: int = 30,
    user_token: str = "",
) -> dict:
    """
    为新员工批量添加培训课程。

    参数:
        employee_id: 员工工号
        employee_name: 员工姓名
        department: 部门
        course_list: 课程列表（课程 ID 或名称列表）
        enrollment_type: 培训类型（默认新员工入职培训）
        deadline_days: 完成期限（默认30天）
        user_token: 操作人认证令牌
    """
    request_id = str(uuid.uuid4())
    payload = {
        "action_type": "enroll_training_courses",
        "params": {
            "employee_id": employee_id,
            "employee_name": employee_name,
            "department": department,
            "course_list": course_list,
            "enrollment_type": enrollment_type,
            "deadline_days": deadline_days,
        },
        "request_id": request_id,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    headers = _build_headers(user_token, request_id)
    url = f"{settings.java_backend_base_url}/api/onboarding/training"

    result = await _http_tool_call(url, payload, headers, "enroll_training_courses")
    if result["success"]:
        return {
            "success": True,
            "request_id": request_id,
            "message": f"已为 {employee_name} 添加 {len(course_list)} 门培训课程",
            "data": result["data"],
        }
    return {
        "success": False,
        "request_id": request_id,
        "message": f"添加失败: {result.get('error', '未知错误')}",
    }


# ─────────────────────────────────────────────
# 入职全流程编排
# ─────────────────────────────────────────────
@mcp.tool()
async def onboarding_full_flow(
    candidate_name: str,
    department: str,
    position: str,
    entry_date: str,
    salary: str,
    employee_id: str,
    email: str,
    device_type: str,
    course_list: list,
    contract_type: str = "全职",
    probation_period: str = "3个月",
    reporting_manager: str = "",
    reporting_location: str = "",
    mobile: str = "",
    manager_id: str = "",
    user_token: str = "",
) -> dict:
    """
    入职全流程自动化：依次执行入职通知书生成、OA 账号开通、IT 设备申请、培训课程添加。
    返回每一步的执行结果。

    参数:
        candidate_name: 候选人姓名
        department: 部门名称
        position: 岗位名称
        entry_date: 入职日期（YYYY-MM-DD）
        salary: 薪资待遇
        employee_id: 工号
        email: 邮箱地址
        device_type: IT 设备类型（笔记本电脑/显示器/键盘鼠标套装/耳机/其他）
        course_list: 培训课程列表（课程 ID 或名称列表）
        contract_type: 合同类型（默认全职）
        probation_period: 试用期（默认3个月）
        reporting_manager: 直属上级（可选）
        reporting_location: 报到地点（可选）
        mobile: 手机号（可选）
        manager_id: 直属上级工号（可选）
        user_token: 操作人认证令牌
    """
    import asyncio

    request_id = str(uuid.uuid4())
    steps = []

    # 步骤 1：入职通知书
    offer_payload = {
        "action_type": "generate_offer_letter",
        "params": {
            "candidate_name": candidate_name,
            "department": department,
            "position": position,
            "entry_date": entry_date,
            "salary": salary,
            "contract_type": contract_type,
            "probation_period": probation_period,
            "reporting_manager": reporting_manager,
            "reporting_location": reporting_location,
        },
        "request_id": request_id,
        "metadata": {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }
    offer_headers = _build_headers(user_token, request_id)
    offer_url = f"{settings.java_backend_base_url}/api/onboarding/offer-letter"
    offer_result = await _http_tool_call(offer_url, offer_payload, offer_headers, "generate_offer_letter")
    steps.append({
        "step": "入职通知书生成",
        "success": offer_result["success"],
        "message": "入职通知书已生成" if offer_result["success"] else f"失败: {offer_result.get('error', '')}",
        "data": offer_result.get("data"),
    })

    # 步骤 2：OA 账号
    account_payload = {
        "action_type": "create_oa_account",
        "params": {
            "employee_name": candidate_name,
            "employee_id": employee_id,
            "department": department,
            "position": position,
            "email": email,
            "mobile": mobile,
            "manager_id": manager_id,
        },
        "request_id": request_id,
        "metadata": {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }
    account_headers = _build_headers(user_token, request_id)
    account_url = f"{settings.java_backend_base_url}/api/onboarding/oa-account"
    account_result = await _http_tool_call(account_url, account_payload, account_headers, "create_oa_account")
    steps.append({
        "step": "OA 账号开通",
        "success": account_result["success"],
        "message": "OA 账号开通成功" if account_result["success"] else f"失败: {account_result.get('error', '')}",
        "data": account_result.get("data"),
    })

    # 步骤 3：IT 设备
    it_payload = {
        "action_type": "apply_it_equipment",
        "params": {
            "employee_id": employee_id,
            "employee_name": candidate_name,
            "department": department,
            "device_type": device_type,
            "quantity": 1,
            "reason": f"新员工入职: {position}",
        },
        "request_id": request_id,
        "metadata": {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }
    it_headers = _build_headers(user_token, request_id)
    it_url = f"{settings.java_backend_base_url}/api/onboarding/it-equipment"
    it_result = await _http_tool_call(it_url, it_payload, it_headers, "apply_it_equipment")
    steps.append({
        "step": "IT 设备申请",
        "success": it_result["success"],
        "message": "IT 设备申请已提交" if it_result["success"] else f"失败: {it_result.get('error', '')}",
        "data": it_result.get("data"),
    })

    # 步骤 4：培训课程
    training_payload = {
        "action_type": "enroll_training_courses",
        "params": {
            "employee_id": employee_id,
            "employee_name": candidate_name,
            "department": department,
            "course_list": course_list,
            "enrollment_type": "新员工入职培训",
            "deadline_days": 30,
        },
        "request_id": request_id,
        "metadata": {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }
    training_headers = _build_headers(user_token, request_id)
    training_url = f"{settings.java_backend_base_url}/api/onboarding/training"
    training_result = await _http_tool_call(training_url, training_payload, training_headers, "enroll_training_courses")
    steps.append({
        "step": "培训课程添加",
        "success": training_result["success"],
        "message": f"已添加 {len(course_list)} 门课程" if training_result["success"] else f"失败: {training_result.get('error', '')}",
        "data": training_result.get("data"),
    })

    all_success = all(s["success"] for s in steps)
    summary = f"入职流程完成，成功 {sum(s['success'] for s in steps)}/4 步"
    return {
        "success": all_success,
        "request_id": request_id,
        "summary": summary,
        "steps": steps,
        "message": summary,
    }


# ─────────────────────────────────────────────
# RAG 检索工具
# ─────────────────────────────────────────────
@mcp.tool()
async def rag_retrieve(
    query: str,
    top_k: int = 5,
    user_token: str = "",
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
        user_token: 用户认证令牌（用于 ABAC 权限过滤）

    返回:
        包含检索到的文档列表和生成的回答
    """
    import asyncio

    try:
        from ..controlled_self_rag import get_controlled_self_rag
        from ..auth import decode_jwt_token

        self_rag = get_controlled_self_rag()

        user_dept = None
        user_projects: list = []
        if user_token:
            try:
                token_info = decode_jwt_token(user_token)
                user_dept = token_info.departments[0] if token_info.departments else None
                user_projects = token_info.projects or []
            except Exception:
                pass

        result = await asyncio.to_thread(
            self_rag.process,
            query=query,
            user_dept=user_dept,
            user_projects=user_projects,
        )

        if not result.is_useful or not result.docs:
            return {
                "success": True,
                "is_useful": False,
                "answer": "在知识库中未找到相关信息，建议联系 HR 部门（内线 8001）获取帮助。",
                "docs": [],
                "message": "检索结果无用",
            }

        return {
            "success": True,
            "is_useful": True,
            "answer": result.answer,
            "docs": [
                {
                    "content": doc.get("text", "")[:500],
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
                "user_token": {"type": "string", "description": "用户认证令牌（用于 ABAC 权限过滤）"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "generate_offer_letter",
        "description": "生成新员工入职通知书，包含薪资、合同类型、试用期等关键信息。",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_name": {"type": "string", "description": "候选人姓名"},
                "department": {"type": "string", "description": "部门名称"},
                "position": {"type": "string", "description": "岗位名称"},
                "entry_date": {"type": "string", "description": "入职日期（YYYY-MM-DD）"},
                "salary": {"type": "string", "description": "薪资待遇"},
                "contract_type": {"type": "string", "description": "合同类型，默认全职"},
                "probation_period": {"type": "string", "description": "试用期，默认3个月"},
                "reporting_manager": {"type": "string", "description": "直属上级"},
                "reporting_location": {"type": "string", "description": "报到地点"},
            },
            "required": ["candidate_name", "department", "position", "entry_date", "salary"],
        },
    },
    {
        "name": "create_oa_account",
        "description": "为新员工开通 OA 系统账号，自动配置部门、岗位、汇报关系。",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_name": {"type": "string", "description": "员工姓名"},
                "employee_id": {"type": "string", "description": "工号"},
                "department": {"type": "string", "description": "部门"},
                "position": {"type": "string", "description": "岗位"},
                "email": {"type": "string", "description": "邮箱地址"},
                "mobile": {"type": "string", "description": "手机号"},
                "manager_id": {"type": "string", "description": "直属上级工号"},
            },
            "required": ["employee_name", "employee_id", "department", "position", "email"],
        },
    },
    {
        "name": "apply_it_equipment",
        "description": "为新员工申请 IT 设备（笔记本电脑、显示器、键鼠套装、耳机等）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string", "description": "员工工号"},
                "employee_name": {"type": "string", "description": "员工姓名"},
                "department": {"type": "string", "description": "部门"},
                "device_type": {"type": "string", "description": "设备类型（笔记本电脑/显示器/键盘鼠标套装/耳机/其他）"},
                "device_spec": {"type": "string", "description": "设备规格描述"},
                "quantity": {"type": "integer", "description": "数量，默认1"},
                "reason": {"type": "string", "description": "申请理由"},
            },
            "required": ["employee_id", "employee_name", "department", "device_type"],
        },
    },
    {
        "name": "enroll_training_courses",
        "description": "为新员工批量添加入职培训课程，支持设置完成期限。",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string", "description": "员工工号"},
                "employee_name": {"type": "string", "description": "员工姓名"},
                "department": {"type": "string", "description": "部门"},
                "course_list": {"type": "array", "description": "课程列表（课程 ID 或名称）", "items": {"type": "string"}},
                "enrollment_type": {"type": "string", "description": "培训类型，默认新员工入职培训"},
                "deadline_days": {"type": "integer", "description": "完成期限（天），默认30"},
            },
            "required": ["employee_id", "employee_name", "department", "course_list"],
        },
    },
    {
        "name": "onboarding_full_flow",
        "description": "入职全流程自动化：依次执行入职通知书生成、OA 账号开通、IT 设备申请、培训课程添加，返回每步结果。",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_name": {"type": "string", "description": "候选人姓名"},
                "department": {"type": "string", "description": "部门名称"},
                "position": {"type": "string", "description": "岗位名称"},
                "entry_date": {"type": "string", "description": "入职日期（YYYY-MM-DD）"},
                "salary": {"type": "string", "description": "薪资待遇"},
                "employee_id": {"type": "string", "description": "工号"},
                "email": {"type": "string", "description": "邮箱地址"},
                "device_type": {"type": "string", "description": "IT 设备类型"},
                "course_list": {"type": "array", "description": "培训课程列表", "items": {"type": "string"}},
                "contract_type": {"type": "string", "description": "合同类型，默认全职"},
                "probation_period": {"type": "string", "description": "试用期，默认3个月"},
                "reporting_manager": {"type": "string", "description": "直属上级"},
                "reporting_location": {"type": "string", "description": "报到地点"},
                "mobile": {"type": "string", "description": "手机号"},
                "manager_id": {"type": "string", "description": "直属上级工号"},
            },
            "required": [
                "candidate_name", "department", "position", "entry_date",
                "salary", "employee_id", "email", "device_type", "course_list",
            ],
        },
    },
]
