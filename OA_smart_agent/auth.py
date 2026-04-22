"""
用户认证与权限控制模块 (auth.py)

这个模块是系统的"门卫"，负责：
1. 验证用户身份（JWT Token）
2. 检查用户权限（RBAC）

就像公司门禁系统：
- Token 验证 = 检查员工卡是否有效
- 权限检查 = 检查员工是否有权进入某个区域
"""

# ========== 引入工具库 ==========
import logging  # 日志记录
from typing import Optional, List  # 类型注解
from functools import wraps  # 函数装饰器
from fastapi import HTTPException, Request  # FastAPI 异常处理
import jwt  # JWT Token 处理库

from config import settings  # 全局配置

logger = logging.getLogger("oa_agent.auth")  # 创建日志记录器


# =============================================================================
# Token 数据结构
# =============================================================================

class TokenPayload:
    """
    Token 载荷 - 解码后的 JWT Token 内容

    就像员工的工牌信息卡，上面写着：
    - user_id: 工号
    - departments: 所属部门列表
    - roles: 角色列表（普通员工 / 管理员 / HR 等）
    """
    def __init__(
        self,
        user_id: str,
        departments: List[str] = None,
        roles: List[str] = None,
        projects: List[str] = None,
    ):
        self.user_id = user_id  # 用户ID
        self.departments = departments or []  # 所属部门
        self.roles = roles or []  # 角色列表
        self.projects = projects or []  # 参与的项目列表

    def has_role(self, role: str) -> bool:
        """检查用户是否有指定角色"""
        return role in self.roles


# =============================================================================
# JWT Token 验证
# =============================================================================

def decode_jwt_token(token: str) -> TokenPayload:
    """
    解码 JWT Token，返回用户信息

    就像扫描工牌，读取里面的信息：
    1. 验证签名是否正确（卡是否是伪造的）
    2. 验证是否过期（工牌是否挂失）
    3. 提取用户信息
    """
    try:
        # 如果是 Bearer Token 格式，先去掉前缀
        if token.startswith("Bearer "):
            token = token[7:]

        # 解码 Token
        payload = jwt.decode(
            token,
            settings.jwt_secret,  # 解密密钥
            algorithms=["HS256"],  # 算法
            options={"verify_exp": True}  # 验证过期时间
        )

        # 提取用户信息
        return TokenPayload(
            user_id=payload.get("user_id", ""),
            departments=payload.get("departments", []),
            roles=payload.get("roles", []),
            projects=payload.get("projects", []),
        )

    except jwt.ExpiredSignatureError:
        # Token 过期了
        logger.warning("token_expired", extra={"component": "auth"})
        raise HTTPException(status_code=401, detail="Token 已过期，请重新登录")

    except jwt.InvalidTokenError as e:
        # Token 无效（伪造或损坏）
        logger.warning(f"token_invalid: {e}", extra={"component": "auth"})
        raise HTTPException(status_code=401, detail="无效的 Token")


def verify_token(request: Request) -> TokenPayload:
    """
    FastAPI 依赖注入：验证 Token

    这是 FastAPI 的"关卡守卫"，每个需要认证的接口都会调用它。
    自动从请求头或请求体中提取 Token 并验证。

    就像每个入口都有人检查工牌：
    - 有有效工牌 → 放行进去
    - 没有/无效工牌 → 挡在外面
    """
    # 尝试从请求头获取 Token
    auth_header = request.headers.get("Authorization", "")

    # 如果请求头没有，尝试从请求体获取
    if not auth_header:
        try:
            body = request._body  # 获取请求体
            if body:
                import json
                body_json = json.loads(body)
                auth_header = body_json.get("user_token", "")
        except:
            pass

    if not auth_header:
        # 没有 Token，直接拒绝
        raise HTTPException(
            status_code=401,
            detail="未提供认证 Token，请先登录获取"
        )

    # 验证 Token
    return decode_jwt_token(auth_header)


# =============================================================================
# 权限控制（RBAC）
# =============================================================================

# 权限映射表：定义每种 action_type 需要什么角色
ACTION_PERMISSION_MAP = {
    # 格式：action_type: [允许的角色列表]
    # 如果列表为空，表示所有用户都可以执行

    # HR 相关操作
    "leave_request": [],  # 请假申请 - 所有人
    "leave_approve": ["hr", "manager", "admin"],  # 请假审批 - HR/经理/管理员
    "leave_cancel": [],  # 取消请假 - 所有人

    # IT 相关操作
    "reset_password": [],  # 密码重置 - 所有人
    "create_account": ["it", "admin"],  # 创建账号 - IT/管理员
    "grant_permission": ["it", "admin"],  # 授予权限 - IT/管理员

    # 财务相关操作
    "reimbursement": [],  # 报销申请 - 所有人
    "reimbursement_approve": ["finance", "admin"],  # 报销审批 - 财务/管理员

    # 管理员操作
    "delete_knowledge": ["admin"],  # 删除知识 - 仅管理员
    "system_config": ["admin"],  # 系统配置 - 仅管理员
}


def check_action_permission(token_info: TokenPayload, action_type: str) -> bool:
    """
    检查用户是否有权执行指定操作（RBAC）

    就像检查员工是否有权进入某个区域：
    1. 查权限表，看这个操作需要什么角色
    2. 看员工卡上有没有这些角色
    3. 有权限 → 放行；没权限 → 拒绝
    """
    # 获取该操作需要的角色列表
    required_roles = ACTION_PERMISSION_MAP.get(action_type, [])

    # 如果为空，表示不需要特殊权限，直接放行
    if not required_roles:
        return True

    # 检查用户是否有所需角色
    for role in required_roles:
        if token_info.has_role(role):
            logger.info(
                "permission_granted",
                extra={
                    "user_id": token_info.user_id,
                    "action_type": action_type,
                    "matched_role": role,
                    "component": "auth",
                }
            )
            return True

    # 没有权限
    logger.warning(
        "permission_denied",
        extra={
            "user_id": token_info.user_id,
            "action_type": action_type,
            "required_roles": required_roles,
            "user_roles": token_info.roles,
            "component": "auth",
        }
    )
    raise HTTPException(
        status_code=403,
        detail=f"您没有权限执行 '{action_type}' 操作，请联系管理员"
    )


# =============================================================================
# 会话清理
# =============================================================================

def cleanup_session(session_id: str):
    """
    清理会话相关资源

    当用户结束会话或会话超时时调用，清理 LangGraph Checkpointer 中的状态。

    注意：不再需要清理 Redis，因为状态现在统一存储在 Checkpointer (PostgreSQL) 中。
    """
    try:
        # Checkpointer 状态会随着新会话自动覆盖，无需手动清理
        logger.info(
            "session_cleaned",
            extra={"session_id": session_id, "component": "auth"}
        )
    except Exception as e:
        logger.error(
            "session_cleanup_failed",
            extra={"session_id": session_id, "error": str(e), "component": "auth"}
        )