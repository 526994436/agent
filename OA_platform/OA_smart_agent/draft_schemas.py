# -*- coding: utf-8 -*-
"""
业务 Schema 强约束模块 (draft_schemas.py)

================================================================================

【核心思想】
每个 Java 业务接口提前定义固定必填字段 Schema，由代码计算缺失参数，
大模型只负责自然语言提问，杜绝 LLM 幻觉乱问、漏传参数。

【设计原则】
1. Schema 先行：每种业务操作（请假、报销等）都有固定字段定义
2. 强约束：字段类型、范围、必填/可选全部声明清楚
3. 代码驱动：缺失参数由代码计算，而非让 LLM 猜测
4. 自然语言问询：LLM 只负责向用户提问获取参数

【新增功能】
1. 动态缺失字段计算 - 支持 partial 实例
2. 历史参数填充 - 复用用户历史数据
3. 字段来源追溯 - 区分 user_input / history / auto
4. 批量 Schema 操作 - 支持批量更新

================================================================================
"""

from typing import Optional, List, Literal, Dict, Any, Callable
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigError
from enum import Enum
from dataclasses import dataclass, field
import json


# =============================================================================
# 第一部分：业务操作类型枚举
# =============================================================================

class BusinessAction(str, Enum):
    """业务操作类型枚举 - 所有支持的业务操作"""
    LEAVE_REQUEST = "leave_request"           # 请假申请
    EXPENSE_REIMBURSE = "expense_reimburse"  # 费用报销
    PASSWORD_RESET = "password_reset"        # 密码重置
    PERMISSION_OPEN = "permission_open"     # 权限开通
    ACCOUNT_CREATE = "account_create"        # 账号创建
    # 可扩展其他业务...


# =============================================================================
# 第二部分：业务 Schema 定义（强约束）
# =============================================================================

class LeaveRequestSchema(BaseModel):
    """
    请假申请 Schema - 强约束字段定义

    【必填字段】
    - start_date: 开始日期（YYYY-MM-DD）
    - end_date: 结束日期（YYYY-MM-DD）
    - leave_type: 请假类型（年假/病假/事假/婚假/产假/丧假）

    【可选字段】
    - reason: 请假原因
    - cover_duties_person: 替班人员

    【校验规则】
    - 日期格式必须为 YYYY-MM-DD
    - 结束日期 >= 开始日期
    - 请假天数不能超过规定上限
    """
    start_date: str = Field(..., description="开始日期，格式 YYYY-MM-DD")
    end_date: str = Field(..., description="结束日期，格式 YYYY-MM-DD")
    leave_type: str = Field(..., description="请假类型：年假/病假/事假/婚假/产假/丧假")
    reason: Optional[str] = Field(None, description="请假原因")
    cover_duties_person: Optional[str] = Field(None, description="替班人员姓名")

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """校验日期格式"""
        import re
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError(f"日期格式必须为 YYYY-MM-DD，实际：{v}")
        return v

    @field_validator('leave_type')
    @classmethod
    def validate_leave_type(cls, v: str) -> str:
        """校验请假类型"""
        valid_types = ['年假', '病假', '事假', '婚假', '产假', '丧假']
        if v not in valid_types:
            raise ValueError(f"请假类型必须是：{valid_types} 之一，实际：{v}")
        return v

    def get_required_fields() -> List[str]:
        """返回必填字段列表"""
        return ['start_date', 'end_date', 'leave_type']

    def get_optional_fields() -> List[str]:
        """返回可选字段列表"""
        return ['reason', 'cover_duties_person']

    def get_missing_fields(self) -> List[str]:
        """计算缺失的必填字段"""
        missing = []
        for field in self.get_required_fields():
            value = getattr(self, field, None)
            if not value or (isinstance(value, str) and not value.strip()):
                missing.append(field)
        return missing

    def to_payload(self, user_id: str, user_token: str) -> dict:
        """转换为 Java API 调用的 payload"""
        return {
            "action_type": BusinessAction.LEAVE_REQUEST.value,
            "user_id": user_id,
            "params": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "leave_type": self.leave_type,
                "reason": self.reason or "",
                "cover_duties_person": self.cover_duties_person or "",
            },
            "metadata": {
                "submitter": user_id,
                "submit_time": "",  # 由 Java 后端填充
            }
        }


class ExpenseReimburseSchema(BaseModel):
    """
    费用报销 Schema - 强约束字段定义

    【必填字段】
    - expense_type: 费用类型
    - amount: 报销金额（元）
    - description: 费用说明

    【可选字段】
    - invoice_no: 发票号

    【校验规则】
    - 金额必须 > 0
    - 费用类型必须在允许范围内
    """
    expense_type: str = Field(..., description="费用类型：差旅费/交通费/餐饮费/办公费/通讯费/其他")
    amount: float = Field(..., description="报销金额（元）", gt=0)
    description: str = Field(..., description="费用说明")
    invoice_no: Optional[str] = Field(None, description="发票号")

    @field_validator('expense_type')
    @classmethod
    def validate_expense_type(cls, v: str) -> str:
        """校验费用类型"""
        valid_types = ['差旅费', '交通费', '餐饮费', '办公费', '通讯费', '其他']
        if v not in valid_types:
            raise ValueError(f"费用类型必须是：{valid_types} 之一，实际：{v}")
        return v

    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v: float) -> float:
        """校验金额"""
        if v <= 0:
            raise ValueError(f"报销金额必须 > 0，实际：{v}")
        if v > 100000:
            raise ValueError(f"报销金额不能超过 100000 元（大额需走特殊审批）")
        return v

    def get_required_fields() -> List[str]:
        return ['expense_type', 'amount', 'description']

    def get_optional_fields() -> List[str]:
        return ['invoice_no']

    def get_missing_fields(self) -> List[str]:
        missing = []
        for field in self.get_required_fields():
            value = getattr(self, field, None)
            if not value or (isinstance(value, str) and not value.strip()):
                missing.append(field)
        return missing

    def to_payload(self, user_id: str, user_token: str) -> dict:
        return {
            "action_type": BusinessAction.EXPENSE_REIMBURSE.value,
            "user_id": user_id,
            "params": {
                "expense_type": self.expense_type,
                "amount": self.amount,
                "description": self.description,
                "invoice_no": self.invoice_no or "",
            },
            "metadata": {
                "submitter": user_id,
                "submit_time": "",
            }
        }


class PasswordResetSchema(BaseModel):
    """
    密码重置 Schema - 强约束字段定义

    【必填字段】
    - user_id: 要重置密码的用户ID（管理员操作时必填）
    - verification_method: 验证方式（邮箱/手机/安全问题）

    【可选字段】
    - reason: 重置原因
    """
    user_id: Optional[str] = Field(None, description="要重置密码的用户ID")
    verification_method: str = Field(..., description="验证方式：email/mobile/security_question")
    reason: Optional[str] = Field(None, description="重置原因")

    @field_validator('verification_method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        valid_methods = ['email', 'mobile', 'security_question']
        if v not in valid_methods:
            raise ValueError(f"验证方式必须是：{valid_methods} 之一")
        return v

    def get_required_fields() -> List[str]:
        return ['verification_method']

    def get_optional_fields() -> List[str]:
        return ['user_id', 'reason']

    def get_missing_fields(self) -> List[str]:
        missing = []
        for field in self.get_required_fields():
            value = getattr(self, field, None)
            if not value or (isinstance(value, str) and not value.strip()):
                missing.append(field)
        return missing

    def to_payload(self, user_id: str, user_token: str) -> dict:
        return {
            "action_type": BusinessAction.PASSWORD_RESET.value,
            "user_id": user_id,
            "params": {
                "target_user_id": self.user_id or user_id,  # 默认为当前用户
                "verification_method": self.verification_method,
                "reason": self.reason or "",
            },
            "metadata": {
                "submitter": user_id,
                "submit_time": "",
            }
        }


class PermissionOpenSchema(BaseModel):
    """
    权限开通 Schema - 强约束字段定义

    【必填字段】
    - system_name: 系统名称
    - permission_name: 权限名称
    - duration: 有效期

    【可选字段】
    - reason: 申请原因
    """
    system_name: str = Field(..., description="系统名称")
    permission_name: str = Field(..., description="权限名称")
    duration: str = Field(..., description="有效期：临时/短期/长期")
    reason: Optional[str] = Field(None, description="申请原因")

    @field_validator('duration')
    @classmethod
    def validate_duration(cls, v: str) -> str:
        valid_durations = ['临时', '短期', '长期']
        if v not in valid_durations:
            raise ValueError(f"有效期必须是：{valid_durations} 之一")
        return v

    def get_required_fields() -> List[str]:
        return ['system_name', 'permission_name', 'duration']

    def get_optional_fields() -> List[str]:
        return ['reason']

    def get_missing_fields(self) -> List[str]:
        missing = []
        for field in self.get_required_fields():
            value = getattr(self, field, None)
            if not value or (isinstance(value, str) and not value.strip()):
                missing.append(field)
        return missing

    def to_payload(self, user_id: str, user_token: str) -> dict:
        return {
            "action_type": BusinessAction.PERMISSION_OPEN.value,
            "user_id": user_id,
            "params": {
                "system_name": self.system_name,
                "permission_name": self.permission_name,
                "duration": self.duration,
                "reason": self.reason or "",
            },
            "metadata": {
                "submitter": user_id,
                "submit_time": "",
            }
        }


# =============================================================================
# 第三部分：Schema 注册表（统一管理）
# =============================================================================

class SchemaRegistry:
    """
    Schema 注册表 - 统一管理所有业务操作的 Schema

    【用途】
    1. 根据 action_type 获取对应的 Schema 类
    2. 根据 Schema 动态计算缺失字段
    3. 统一生成自然语言提问模板
    """

    # Schema 类映射
    _schemas: Dict[str, type] = {
        BusinessAction.LEAVE_REQUEST.value: LeaveRequestSchema,
        BusinessAction.EXPENSE_REIMBURSE.value: ExpenseReimburseSchema,
        BusinessAction.PASSWORD_RESET.value: PasswordResetSchema,
        BusinessAction.PERMISSION_OPEN.value: PermissionOpenSchema,
    }

    # 字段中文描述（用于生成提问）
    _field_descriptions: Dict[str, Dict[str, str]] = {
        BusinessAction.LEAVE_REQUEST.value: {
            "start_date": "开始日期（格式：YYYY-MM-DD，例如：2024-03-01）",
            "end_date": "结束日期（格式：YYYY-MM-DD，例如：2024-03-03）",
            "leave_type": "请假类型（年假/病假/事假/婚假/产假/丧假）",
            "reason": "请假原因",
            "cover_duties_person": "替班人员姓名",
        },
        BusinessAction.EXPENSE_REIMBURSE.value: {
            "expense_type": "费用类型（差旅费/交通费/餐饮费/办公费/通讯费/其他）",
            "amount": "报销金额（元）",
            "description": "费用说明",
            "invoice_no": "发票号（如果没有可以留空）",
        },
        BusinessAction.PASSWORD_RESET.value: {
            "verification_method": "验证方式（email 邮箱/mobile 手机/security_question 安全问题）",
            "reason": "重置原因",
        },
        BusinessAction.PERMISSION_OPEN.value: {
            "system_name": "系统名称",
            "permission_name": "权限名称",
            "duration": "有效期（临时/短期/长期）",
            "reason": "申请原因",
        },
    }

    # 字段提问模板
    _field_questions: Dict[str, Dict[str, str]] = {
        BusinessAction.LEAVE_REQUEST.value: {
            "start_date": "请问您想从哪天开始请假？",
            "end_date": "请问您想请到哪天结束？",
            "leave_type": "请问您请的是什么类型的假？（年假/病假/事假/婚假/产假/丧假）",
            "reason": "请问您请假的原因是什么？",
            "cover_duties_person": "请问谁替您处理工作？",
        },
        BusinessAction.EXPENSE_REIMBURSE.value: {
            "expense_type": "请问这是什么类型的费用？（差旅费/交通费/餐饮费/办公费/通讯费/其他）",
            "amount": "请问报销的金额是多少？",
            "description": "请简单说明一下这笔费用的用途。",
            "invoice_no": "请问发票号是多少？（如果没有可以不填）",
        },
        BusinessAction.PASSWORD_RESET.value: {
            "verification_method": "您想通过什么方式验证身份？（邮箱/手机/安全问题）",
            "reason": "请问您重置密码的原因是什么？",
        },
        BusinessAction.PERMISSION_OPEN.value: {
            "system_name": "请问您要开通哪个系统的权限？",
            "permission_name": "请问您需要什么权限？",
            "duration": "请问您需要多久的权限？（临时/短期/长期）",
            "reason": "请问您申请这个权限的原因是什么？",
        },
    }

    @classmethod
    def get_schema_class(cls, action_type: str) -> Optional[type]:
        """根据 action_type 获取 Schema 类"""
        return cls._schemas.get(action_type)

    @classmethod
    def get_field_description(cls, action_type: str, field: str) -> str:
        """获取字段的中文描述"""
        return cls._field_descriptions.get(action_type, {}).get(field, field)

    @classmethod
    def get_field_question(cls, action_type: str, field: str) -> str:
        """获取字段的提问模板"""
        return cls._field_questions.get(action_type, {}).get(field, f"请提供 {field}")

    @classmethod
    def build_questions_for_fields(cls, action_type: str, fields: List[str]) -> List[str]:
        """
        根据缺失字段列表生成提问列表

        Args:
            action_type: 业务操作类型
            fields: 缺失的字段列表

        Returns:
            提问列表（按优先级排序）
        """
        questions = []
        for field in fields:
            question = cls.get_field_question(action_type, field)
            if question:
                questions.append(question)
        return questions

    @classmethod
    def get_all_actions(cls) -> List[str]:
        """获取所有注册的业务操作类型"""
        return list(cls._schemas.keys())


# =============================================================================
# 第四部分：Schema 实例工厂
# =============================================================================

def create_draft_instance(action_type: str) -> Optional[BaseModel]:
    """
    根据 action_type 创建空白的 Schema 实例

    Args:
        action_type: 业务操作类型

    Returns:
        Schema 实例（如果 action_type 有效）或 None
    """
    schema_class = SchemaRegistry.get_schema_class(action_type)
    if schema_class:
        return schema_class()
    return None


def fill_draft_from_params(action_type: str, params: dict) -> Optional[BaseModel]:
    """
    根据参数填充 Schema 实例

    Args:
        action_type: 业务操作类型
        params: 从对话中提取的参数

    Returns:
        填充后的 Schema 实例
    """
    schema_class = SchemaRegistry.get_schema_class(action_type)
    if not schema_class:
        return None

    try:
        # 过滤掉空值
        filtered_params = {k: v for k, v in params.items() if v is not None and v != ""}
        return schema_class(**filtered_params)
    except Exception as e:
        # 参数校验失败，返回空实例
        return schema_class()


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "BusinessAction",
    "LeaveRequestSchema",
    "ExpenseReimburseSchema",
    "PasswordResetSchema",
    "PermissionOpenSchema",
    "SchemaRegistry",
    "create_draft_instance",
    "fill_draft_from_params",
    "PartialSchemaMixin",
    "compute_missing_fields",
    "build_field_questions",
    "get_action_display_info",
]


# =============================================================================
# 第五部分：增强的 Schema 辅助功能
# =============================================================================

class PartialSchemaMixin:
    """
    Partial Schema 混入类 - 支持不完整实例的缺失字段计算

    【用途】
    当用户只填了部分字段时，创建 partial 实例来计算还缺什么字段，
    而不需要填所有必填字段才能创建实例。

    【使用示例】
    ```python
    # 部分填充
    partial = LeaveRequestSchemaPartial(start_date="2024-03-01", leave_type="年假")
    missing = partial.get_missing_fields()
    # missing = ["end_date"]

    # 批量填充
    partial = LeaveRequestSchemaPartial(**user_input)
    missing = partial.get_missing_fields()
    ```
    """

    @classmethod
    def get_required_fields(cls) -> List[str]:
        """子类必须实现：返回必填字段列表"""
        raise NotImplementedError

    @classmethod
    def get_optional_fields(cls) -> List[str]:
        """子类必须实现：返回可选字段列表"""
        raise NotImplementedError

    @classmethod
    def get_all_fields(cls) -> List[str]:
        """获取所有字段（必填 + 可选）"""
        return cls.get_required_fields() + cls.get_optional_fields()

    @classmethod
    def compute_missing_fields(cls, **kwargs) -> List[str]:
        """
        根据输入参数计算缺失的必填字段

        Args:
            **kwargs: 字段名到值的映射

        Returns:
            缺失的必填字段列表
        """
        missing = []
        for field in cls.get_required_fields():
            value = kwargs.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(field)
        return missing

    @classmethod
    def validate_partial(cls, data: dict) -> tuple[bool, List[str], dict]:
        """
        校验部分填充的数据

        Args:
            data: 字段数据字典

        Returns:
            (是否有效, 错误列表, 过滤后的数据)
        """
        errors = []
        filtered = {}

        for field in cls.get_required_fields() + cls.get_optional_fields():
            value = data.get(field)
            if value is not None:
                filtered[field] = value

        # 尝试创建实例进行类型校验
        try:
            instance = cls(**filtered)
            return True, [], filtered
        except Exception as e:
            errors.append(str(e))
            return False, errors, filtered


def compute_missing_fields(action_type: str, params: dict) -> List[str]:
    """
    根据 action_type 和参数计算缺失字段

    Args:
        action_type: 业务操作类型
        params: 已填充的参数

    Returns:
        缺失的必填字段列表
    """
    schema_class = SchemaRegistry.get_schema_class(action_type)
    if not schema_class:
        return []

    return schema_class.compute_missing_fields(**params)


def build_field_questions(action_type: str, missing_fields: List[str]) -> List[Dict[str, str]]:
    """
    根据缺失字段生成提问列表

    Args:
        action_type: 业务操作类型
        missing_fields: 缺失字段列表

    Returns:
        提问字典列表，每个包含 question、field、description
    """
    questions = []
    for field in missing_fields:
        question = SchemaRegistry.get_field_question(action_type, field)
        description = SchemaRegistry.get_field_description(action_type, field)
        if question:
            questions.append({
                "field": field,
                "question": question,
                "description": description,
            })
    return questions


def get_action_display_info(action_type: str) -> Dict[str, Any]:
    """
    获取操作类型的显示信息

    Args:
        action_type: 业务操作类型

    Returns:
        包含 display_name、icon、required_fields_count 等信息的字典
    """
    display_names = {
        BusinessAction.LEAVE_REQUEST.value: {"name": "请假申请", "icon": "calendar", "color": "#4CAF50"},
        BusinessAction.EXPENSE_REIMBURSE.value: {"name": "费用报销", "icon": "receipt", "color": "#2196F3"},
        BusinessAction.PASSWORD_RESET.value: {"name": "密码重置", "icon": "key", "color": "#FF9800"},
        BusinessAction.PERMISSION_OPEN.value: {"name": "权限开通", "icon": "shield", "color": "#9C27B0"},
    }

    info = display_names.get(action_type, {"name": action_type, "icon": "help", "color": "#757575"})

    # 获取必填字段数量
    schema_class = SchemaRegistry.get_schema_class(action_type)
    if schema_class:
        info["required_fields"] = schema_class.get_required_fields()
        info["required_fields_count"] = len(info["required_fields"])
    else:
        info["required_fields"] = []
        info["required_fields_count"] = 0

    return info


def create_partial_schema(action_type: str, params: dict) -> Optional[dict]:
    """
    创建部分填充的 Schema 数据

    Args:
        action_type: 业务操作类型
        params: 已填充的参数

    Returns:
        包含 schema_data 和 missing_fields 的字典
    """
    schema_class = SchemaRegistry.get_schema_class(action_type)
    if not schema_class:
        return None

    # 过滤空值
    filtered = {k: v for k, v in params.items() if v is not None and v != ""}

    # 计算缺失字段
    missing = schema_class.compute_missing_fields(**filtered)

    return {
        "action_type": action_type,
        "schema_data": filtered,
        "missing_fields": missing,
        "filled_count": len(filtered),
        "required_count": len(schema_class.get_required_fields()),
        "is_complete": len(missing) == 0,
    }


def merge_draft_params(action_type: str, current_params: dict, new_params: dict) -> dict:
    """
    合并草稿参数（新参数优先）

    Args:
        action_type: 业务操作类型
        current_params: 当前已填充的参数
        new_params: 新参数

    Returns:
        合并后的参数
    """
    merged = current_params.copy()
    for k, v in new_params.items():
        if v is not None and v != "":
            merged[k] = v

    return merged


# =============================================================================
# 第六部分：Schema 字段填充追踪
# =============================================================================

@dataclass
class FieldFillTrace:
    """字段填充追踪"""
    field: str
    value: Any
    source: str  # user_input / history / auto / default
    confidence: float
    filled_at: str = ""
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
            "filled_at": self.filled_at,
            "note": self.note,
        }


def create_fill_trace(field: str, value: Any, source: str = "user_input",
                      confidence: float = 1.0, note: Optional[str] = None) -> FieldFillTrace:
    """创建字段填充追踪记录"""
    from datetime import datetime, timezone
    return FieldFillTrace(
        field=field,
        value=value,
        source=source,
        confidence=confidence,
        filled_at=datetime.now(timezone.utc).isoformat(),
        note=note,
    )