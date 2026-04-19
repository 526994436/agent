# -*- coding: utf-8 -*-
"""
灰度发布辅助模块 (canary.py)
================================================================================

【模块功能说明】
这个模块负责"灰度发布"功能，就像建筑工地先用小推车测试，再换成大卡车。
在正式发布新功能前，先让一小部分用户试用，观察效果后再全面推广。

【什么是灰度发布？】
灰度发布（Canary Release）是一种软件发布策略：
- 不一次性让所有用户都用新功能
- 先让5%、10%、20%的用户"尝鲜"
- 观察这一小部分用户的使用情况
- 确认没问题后，再逐步扩大范围
- 出了问题可以快速回滚，影响范围小

【为什么需要灰度发布？】
1. 降低风险：新功能可能有bug，灰度发布可以把影响范围控制在小范围
2. 收集反馈：先让"种子用户"试用，收集真实反馈改进产品
3. 性能验证：验证新功能在高并发下的表现
4. 渐进式升级：让用户慢慢适应新变化，不至于突然改变使用习惯

【核心概念】
1. 功能开关（Feature Flag）
   - 控制某个功能是否开启
   - 可以随时开启/关闭，不需要重新部署

2. 灰度路由（Gray Route）
   - 根据用户ID决定使用哪个版本
   - 保证同一用户每次访问都一致（稳定性）

3. 就绪探针（Readiness Probe）
   - 检查系统是否准备好接收请求
   - 用于Kubernetes滚动更新时的流量控制

【应用场景】
场景1：新RAG算法上线
  - 第1天：10%用户使用新算法
  - 第3天：50%用户使用新算法
  - 第7天：100%用户使用新算法
  - 如果有问题，立即关闭开关

场景2：滚动更新
  - K8s正在更新Pod
  - 老Pod继续处理已有会话
  - 新Pod只接收新会话
  - 全部更新完成后才完全切换

场景3：A/B测试
  - 50%用户看到红色按钮
  - 50%用户看到蓝色按钮
  - 统计哪个按钮点击率高
  - 根据数据决定默认颜色
"""

# =============================================================================
# 导入必要的库
# =============================================================================

import logging  # 日志模块，记录开关状态变化
import hashlib  # 哈希算法，用于灰度分配
from typing import Optional, Dict, Callable  # 类型提示
from dataclasses import dataclass, field  # 数据类
import redis  # Redis 客户端

# 创建日志记录器
logger = logging.getLogger("oa_agent.canary")

# Redis 配置
_redis_client: Optional[redis.Redis] = None

def _get_redis_client() -> redis.Redis:
    """获取 Redis 客户端单例"""
    global _redis_client
    if _redis_client is None:
        try:
            from config import settings
            redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379')
            _redis_client = redis.from_url(redis_url, decode_responses=True)
        except Exception as e:
            logger.warning(f"Redis连接失败，将使用内存存储: {e}")
            return None
    return _redis_client


# =============================================================================
# 功能开关管理
# =============================================================================

@dataclass
class FeatureFlag:
    """
    功能开关数据模型
    
    【什么是功能开关？】
    功能开关就像电灯的"开关"。
    - 打开：功能可用
    - 关闭：功能不可用
    
    【额外参数】
    - rollout_percentage：灰度比例，0-100
      * 0% = 功能关闭，没人能用
      * 50% = 一半用户能用
      * 100% = 功能对所有人开放
    
    【元数据】
    可以存储额外信息，如负责人、预计全量时间等
    """
    
    # 开关名称，唯一标识
    name: str
    
    # 是否启用（总开关）
    enabled: bool = False
    
    # 灰度发布比例，0.0-100.0
    rollout_percentage: float = 0.0
    
    # 功能描述
    description: str = ""
    
    # 附加元数据
    metadata: Dict[str, str] = field(default_factory=dict)


class FeatureFlagManager:
    """
    功能开关管理器

    【核心职责】
    管理所有功能开关的状态，提供开关检查和灰度路由能力。

    【存储方式】
    - enabled 状态：存储在内存中（开关是否存在）
    - rollout_percentage：存储在 Redis 中，支持分布式环境下的实时调整

    【Redis 存储结构】
    - Hash: feature_flags:config -> { flag_name: enabled (0/1) }
    - Hash: feature_flags:rollout -> { flag_name: rollout_percentage (0-100) }

    【灰度分配算法】
    使用哈希算法确保：
    1. 同一用户对同一开关，每次判断结果一致（稳定性）
    2. 用户分布均匀，符合配置的百分比（公平性）

    【算法原理】
    hash(user_id + flag_name) -> [0, 1) -> 百分比

    【示例】
    flag_name = "new_rag_algorithm"
    rollout_percentage = 10.0

    用户A: hash("zhangsan:new_rag_algorithm") = 0.03 -> 启用 ✓
    用户B: hash("lisi:new_rag_algorithm") = 0.85 -> 不启用 ✗
    用户C: hash("wangwu:new_rag_algorithm") = 0.12 -> 不启用 ✗

    多次判断：
    用户A第2次判断 -> hash值不变 -> 结果不变 ✓

    【使用示例】
    ```python
    # 获取管理器
    fm = get_feature_flag_manager()

    # 检查用户是否在灰度组
    if fm.is_enabled("new_rag_algorithm", user_id="zhangsan"):
        print("使用新功能")
    else:
        print("使用旧功能")

    # 动态调整灰度比例（存储在 Redis）
    fm.update_rollout("new_rag_algorithm", 50.0)  # 提升到50%
    ```
    """

    # Redis Key 常量
    REDIS_KEY_CONFIG = "feature_flags:config"
    REDIS_KEY_ROLLOUT = "feature_flags:rollout"

    def __init__(self):
        """
        初始化功能开关管理器

        【默认开关】
        系统启动时预设一些默认开关，
        这些开关默认开启，表示基础功能。
        """
        self._flags: Dict[str, FeatureFlag] = {}  # 所有注册的开关

        # 默认功能开关配置
        self._default_flags = {
            # 混合检索开关：同时使用多种检索方式，提高准确性
            "hybrid_search": FeatureFlag(
                name="hybrid_search",
                enabled=True,
                rollout_percentage=100.0,  # 100%用户使用
                description="混合检索（Dense + RRF）"
            ),

            # LLM降级开关：允许LLM在某些情况下降级使用
            "llm_fallback": FeatureFlag(
                name="llm_fallback",
                enabled=True,
                rollout_percentage=100.0,
                description="LLM模型降级"
            ),

            # 断路器开关：防止Java后端故障时请求堆积
            "circuit_breaker": FeatureFlag(
                name="circuit_breaker",
                enabled=True,
                rollout_percentage=100.0,
                description="Java后端断路器"
            ),

            # 结构化日志开关：使用JSON格式日志
            "structured_logging": FeatureFlag(
                name="structured_logging",
                enabled=True,
                rollout_percentage=100.0,
                description="结构化JSON日志"
            ),
        }

        # 将默认开关注册到管理器
        self._flags.update(self._default_flags)

        # 初始化 Redis 中的默认配置
        self._init_redis_defaults()

    def _init_redis_defaults(self):
        """初始化 Redis 中的默认配置"""
        redis_client = _get_redis_client()
        if redis_client is None:
            return

        try:
            pipe = redis_client.pipeline()
            for name, flag in self._default_flags.items():
                # 初始化 enabled 状态
                if not redis_client.hexists(self.REDIS_KEY_CONFIG, name):
                    pipe.hset(self.REDIS_KEY_CONFIG, name, "1" if flag.enabled else "0")
                # 初始化 rollout_percentage
                if not redis_client.hexists(self.REDIS_KEY_ROLLOUT, name):
                    pipe.hset(self.REDIS_KEY_ROLLOUT, name, str(flag.rollout_percentage))
            pipe.execute()
            logger.info("Redis 功能开关配置初始化完成")
        except Exception as e:
            logger.warning(f"Redis 配置初始化失败: {e}")

    def _get_rollout_from_redis(self, flag_name: str) -> Optional[float]:
        """从 Redis 读取灰度比例"""
        redis_client = _get_redis_client()
        if redis_client is None:
            return None

        try:
            value = redis_client.hget(self.REDIS_KEY_ROLLOUT, flag_name)
            if value is not None:
                return float(value)
        except Exception as e:
            logger.warning(f"从 Redis 读取 rollout 失败: {e}")
        return None

    def _get_enabled_from_redis(self, flag_name: str) -> Optional[bool]:
        """从 Redis 读取 enabled 状态"""
        redis_client = _get_redis_client()
        if redis_client is None:
            return None

        try:
            value = redis_client.hget(self.REDIS_KEY_CONFIG, flag_name)
            if value is not None:
                return value == "1"
        except Exception as e:
            logger.warning(f"从 Redis 读取 enabled 失败: {e}")
        return None

    def register_flag(self, flag: FeatureFlag):
        """
        注册新的功能开关
        
        【调用时机】
        在系统启动时或动态创建新功能开关时调用。
        
        【参数】
        - flag: 功能开关对象
        """
        self._flags[flag.name] = flag
        logger.info(
            "feature_flag_registered",
            extra={"name": flag.name, "rollout": flag.rollout_percentage, "component": "canary"}
        )

    def is_enabled(self, flag_name: str, user_id: str = "") -> bool:
        """
        检查功能开关是否对指定用户启用

        【核心逻辑】
        1. 开关是否存在？不存在返回False
        2. 开关是否启用？（优先从 Redis 读取，否则用内存值）
        3. 灰度比例是否为100%？是则返回True
        4. 是否有user_id？没有返回False（需要用户ID才能灰度）
        5. 计算hash，决定是否在灰度组中

        【灰度分配算法详解】
        ```
        1. 拼接输入: "zhangsan:new_rag_algorithm"
        2. 计算SHA256哈希
        3. 取前8位十六进制转为整数
        4. 除以最大值得到[0,1)的比例
        5. 乘以100得到[0,100)的百分比
        6. 与rollout_percentage比较
        ```

        【返回值】
        - True: 用户在灰度组，可以使用新功能
        - False: 用户不在灰度组，使用旧功能
        """
        # 1. 检查开关是否存在
        flag = self._flags.get(flag_name)
        if not flag:
            return False

        # 2. 检查开关是否启用（优先从 Redis 读取）
        redis_enabled = self._get_enabled_from_redis(flag_name)
        if redis_enabled is not None:
            enabled = redis_enabled
        else:
            enabled = flag.enabled

        if not enabled:
            return False

        # 3. 获取灰度比例（优先从 Redis 读取）
        redis_rollout = self._get_rollout_from_redis(flag_name)
        if redis_rollout is not None:
            rollout_percentage = redis_rollout
        else:
            rollout_percentage = flag.rollout_percentage

        # 如果灰度比例是100%，所有人都能用
        if rollout_percentage >= 100.0:
            return True

        # 4. 灰度分配需要user_id
        if not user_id:
            return False

        # 5. 灰度分配hash算法
        # 拼接user_id和flag_name
        hash_input = f"{user_id}:{flag_name}"

        # 计算SHA256哈希，取前8位十六进制
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest()[:8], 16)

        # 转换为[0, 100)的百分比
        bucket = (hash_value / 0xFFFFFFFF) * 100.0

        # 比较是否在灰度范围内
        result = bucket < rollout_percentage

        logger.debug(
            "feature_flag_check",
            extra={
                "flag": flag_name,
                "user_id": user_id,
                "rollout": rollout_percentage,
                "bucket": round(bucket, 2),
                "enabled": result,
                "component": "canary",
            }
        )

        return result

    def update_rollout(self, flag_name: str, percentage: float):
        """
        动态更新灰度比例（存储到 Redis）

        【特点】
        - 无需重启服务
        - 实时生效
        - 可以逐步提升（5% -> 10% -> 50% -> 100%）
        - 存储在 Redis 中，支持分布式环境

        【参数】
        - flag_name: 开关名称
        - percentage: 新的灰度比例，会被限制在0-100之间
        """
        flag = self._flags.get(flag_name)
        if flag:
            # 确保比例在合法范围内
            rollout_percentage = max(0.0, min(100.0, percentage))

            # 存储到 Redis
            redis_client = _get_redis_client()
            if redis_client:
                try:
                    redis_client.hset(self.REDIS_KEY_ROLLOUT, flag_name, str(rollout_percentage))
                    logger.info(
                        "feature_flag_rollout_updated",
                        extra={"name": flag_name, "rollout": rollout_percentage, "source": "redis", "component": "canary"}
                    )
                except Exception as e:
                    logger.warning(f"更新 Redis rollout 失败: {e}")
                    # 回退到内存更新
                    flag.rollout_percentage = rollout_percentage
            else:
                # Redis 不可用时回退到内存
                flag.rollout_percentage = rollout_percentage

    def get_all_flags(self) -> Dict[str, Dict]:
        """
        获取所有功能开关状态

        【用途】
        - 管理后台展示所有开关状态
        - 监控系统检查开关配置
        - 生成配置报告

        【说明】
        - 优先从 Redis 读取最新值
        - 如果 Redis 不可用，回退到内存值
        """
        result = {}
        for name, flag in self._flags.items():
            # 从 Redis 读取最新值
            redis_enabled = self._get_enabled_from_redis(name)
            redis_rollout = self._get_rollout_from_redis(name)

            result[name] = {
                "enabled": redis_enabled if redis_enabled is not None else flag.enabled,
                "rollout_percentage": redis_rollout if redis_rollout is not None else flag.rollout_percentage,
                "description": flag.description,
            }
        return result


# =============================================================================
# 全局功能开关管理器
# =============================================================================

# 全局变量，存储功能开关管理器实例
_feature_flag_manager: Optional[FeatureFlagManager] = None


def get_feature_flag_manager() -> FeatureFlagManager:
    """
    获取功能开关管理器单例
    
    【单例模式】
    整个应用只需要一个开关管理器，
    所有地方都用同一个实例，保证状态一致。
    """
    global _feature_flag_manager
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager()
    return _feature_flag_manager


# =============================================================================
# 灰度路由
# =============================================================================

def gray_route(
    user_id: str,
    flag_name: str,
    primary_handler: Callable,
    canary_handler: Callable,
    default_handler: Optional[Callable] = None,
) -> any:
    """
    灰度路由辅助函数
    
    【核心思想】
    根据功能开关状态，选择执行不同的处理逻辑（处理器）。
    
    【参数说明】
    - user_id: 用户ID，用于决定走哪条路
    - flag_name: 功能开关名称
    - primary_handler: 主处理器（旧功能/稳定版本）
    - canary_handler: 金丝雀处理器（新功能/实验版本）
    - default_handler: 默认处理器（当用户ID为空时使用）
    
    【执行逻辑】
    ```
    if 开关对用户启用:
        执行 canary_handler（新功能）
    else:
        if 有 default_handler:
            执行 default_handler
        else:
            执行 primary_handler（旧功能）
    ```
    
    【使用示例】
    ```python
    # 定义两个版本的处理逻辑
    old_rag_search = lambda: old_rag.search(query)
    new_rag_search = lambda: new_rag.search(query)
    
    # 灰度路由
    result = gray_route(
        user_id="zhangsan",
        flag_name="new_rag_algorithm",
        primary_handler=old_rag_search,
        canary_handler=new_rag_search,
    )
    ```
    
    【返回值】
    返回被选中的处理器执行结果
    """
    fm = get_feature_flag_manager()
    
    # 检查开关是否对用户启用
    if fm.is_enabled(flag_name, user_id):
        logger.debug(
            "gray_route_canary",
            extra={"flag": flag_name, "user_id": user_id, "component": "canary"}
        )
        return canary_handler()
    else:
        logger.debug(
            "gray_route_primary",
            extra={"flag": flag_name, "user_id": user_id, "component": "canary"}
        )
        # 如果有默认处理器，执行默认的；否则执行主处理器
        return primary_handler() if default_handler is None else default_handler()


# =============================================================================
# 就绪探针辅助
# =============================================================================

class ReadinessProbe:
    """
    Kubernetes就绪探针辅助类
    
    【什么是就绪探针？】
    Kubernetes（K8s）用它来判断Pod是否准备好接收流量。
    
    【探针类型】
    1. 存活探针（Liveness Probe）
       - 检查应用是否存活
       - 失败会重启Pod
       - 不影响流量
    
    2. 就绪探针（Readiness Probe）
       - 检查应用是否就绪
       - 失败会摘除流量
       - 已建立的连接不受影响
    
    3. 启动探针（Startup Probe）
       - 应用启动时的检查
       - 启动成功前其他探针禁用
    
    【我们的就绪探针检查】
    - Redis：会话状态存储，必须可用
    - LLM：AI模型，必须可用
    - Java后端：业务操作，可选（可以降级）
    
    【返回值】
    - (True, "healthy"): 所有核心依赖正常
    - (True, "degraded"): 部分依赖不可用，但可降级运行
    - (False, "unhealthy"): 核心依赖不可用，无法处理请求
    
    【滚动更新中的作用】
    当K8s滚动更新Pod时：
    1. 启动新Pod
    2. 新Pod就绪探针检查通过
    3. K8s将流量切换到新Pod
    4. 老Pod就绪探针失败
    5. 老Pod从Service摘除
    6. 老Pod处理完已有会话后退出
    
    【好处】
    - 用户无感知：已有会话继续处理
    - 零停机部署
    - 优雅下线
    """

    def __init__(self):
        """初始化就绪探针"""
        # 各组件的健康状态
        self._checks: Dict[str, bool] = {
            "redis": False,  # Redis连接
            "llm": False,  # AI模型
            "java_backend": False,  # Java后端
        }

    def update_check(self, component: str, healthy: bool):
        """
        更新指定组件的健康状态
        
        【调用时机】
        定时任务检查各组件后调用。
        或者组件状态变化时调用。
        
        【参数】
        - component: 组件名称（redis/llm/java_backend）
        - healthy: 是否健康
        """
        self._checks[component] = healthy

    def is_ready(self) -> tuple[bool, str]:
        """
        判断系统是否就绪
        
        【判断逻辑】
        1. Redis必须可用（会话状态存储）
        2. LLM必须可用（核心AI能力）
        3. Java后端可选（可以降级）
        
        【返回值】
        - (True, "healthy"): 全功能正常
        - (True, "degraded"): 部分降级，但可用
        - (False, "unhealthy"): 不可用
        """
        # 核心依赖检查
        redis_ok = self._checks.get("redis", False)
        llm_ok = self._checks.get("llm", False)
        
        # 可选依赖检查
        java_ok = self._checks.get("java_backend", False)

        if redis_ok and llm_ok:
            # 核心依赖都正常
            if java_ok:
                return True, "healthy"
            else:
                # Java后端不可用，但可以降级
                return True, "degraded"
        else:
            # 核心依赖不可用
            return False, "unhealthy"

    def get_status(self) -> Dict[str, bool]:
        """
        获取各组件健康状态详情
        
        【用途】
        - 监控系统展示组件状态
        - 排查问题时查看具体哪个组件异常
        """
        return self._checks.copy()


# =============================================================================
# 全局就绪探针实例
# =============================================================================

# 全局变量，存储就绪探针实例
_readiness_probe: Optional[ReadinessProbe] = None


def get_readiness_probe() -> ReadinessProbe:
    """
    获取就绪探针单例
    """
    global _readiness_probe
    if _readiness_probe is None:
        _readiness_probe = ReadinessProbe()
    return _readiness_probe
