"""
多模态处理模块 (multimodal.py)

实现图片 + 文本的混合处理能力，调用多模态模型分析图像内容。

技术方案：
- GPT-4o Vision / Qwen-VL / Claude 3 作为多模态模型
- 支持 Base64 图片输入
- 通用图像分析，返回描述文本和结构化数据

流程图：
  ┌─────────────────────────────────────────────────────────────┐
  │                    用户上传图片                              │
  │                    (截图 / 照片)                             │
  └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │              多模态模型分析（GPT-4o/Qwen-VL/Claude）          │
  │  → 提取关键文字、识别场景、生成内容描述                       │
  └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                    返回分析结果                              │
  │  { description, extracted_data, raw_text, confidence }       │
  └─────────────────────────────────────────────────────────────┘

使用方式：
```python
processor = get_multimodal_processor()
result = processor.analyze_image(image_base64, query="请描述这张图片的内容")
```
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("oa_agent.multimodal")

# =============================================================================
# 多模态处理器
# =============================================================================

class MultimodalProcessor:
    """
    多模态处理器。

    支持多种多模态模型：
    1. GPT-4o Vision（OpenAI）
    2. Qwen-VL（阿里云）
    3. Claude 3（Anthropic）
    4. 模拟模式（无 API Key 时）

    使用方式：
    ```python
    processor = get_multimodal_processor()
    result = processor.analyze_image(image_base64, query="这是什么？")
    ```
    """

    def __init__(
        self,
        model_provider: str = "openai",  # "openai" | "qwen" | "claude" | "mock"
        model_name: Optional[str] = None,
    ):
        """
        初始化多模态处理器。

        参数：
        - model_provider: 模型提供商
        - model_name: 模型名称（默认根据 provider 自动选择）
        """
        self.model_provider = model_provider
        self.model_name = model_name or self._get_default_model(model_provider)
        self._client = None
        self._initialized = False

    def _get_default_model(self, provider: str) -> str:
        """获取默认模型名称。"""
        defaults = {
            "openai": "gpt-4o",
            "qwen": "qwen-vl-max",
            "claude": "claude-3-opus-20240229",
            "mock": "mock",
            "vllm": "Qwen/Qwen2.5-VL-32B-A03",  # vLLM 部署的 Qwen3.5-VL
        }
        return defaults.get(provider, "gpt-4o")

    def _ensure_initialized(self):
        """延迟初始化客户端。"""
        if self._initialized:
            return

        try:
            if self.model_provider == "openai":
                from openai import OpenAI
                self._client = OpenAI()
            elif self.model_provider == "qwen":
                # vLLM OpenAI 兼容接口（本地部署的 Qwen3.5-VL）
                from openai import OpenAI
                self._client = OpenAI(
                    base_url="http://localhost:8000/v1",
                    api_key="not-needed",
                    timeout=120,
                )
            elif self.model_provider == "claude":
                from anthropic import Anthropic
                self._client = Anthropic()
            else:
                self._client = None

            self._initialized = True
            logger.info(
                "multimodal_initialized",
                extra={
                    "provider": self.model_provider,
                    "model": self.model_name,
                    "component": "multimodal",
                }
            )
        except ImportError as e:
            logger.warning(
                "multimodal_init_failed",
                extra={
                    "provider": self.model_provider,
                    "error": str(e),
                    "falling_back_to_mock": True,
                }
            )
            self.model_provider = "mock"
            self._initialized = True

    # ─────────────────────────────────────────────────────────────────
    # 核心处理方法
    # ─────────────────────────────────────────────────────────────────

    def analyze_image(
        self,
        image_data: str,
        query: str = "请描述这张图片的内容",
    ) -> Dict[str, Any]:
        """
        通用图像分析。

        参数：
        - image_data: Base64 编码的图片（可带或不带前缀）
        - query: 分析提示词

        返回：包含描述和结构化数据的字典
        """
        self._ensure_initialized()

        if self.model_provider == "mock":
            return self._mock_analyze(image_data, query)

        try:
            if self.model_provider == "openai":
                return self._openai_vision(image_data, query)
            elif self.model_provider == "qwen":
                return self._qwen_vl(image_data, query)
            elif self.model_provider == "claude":
                return self._claude_vision(image_data, query)
        except Exception as e:
            logger.error(
                "multimodal_analysis_failed",
                extra={"provider": self.model_provider, "error": str(e)}
            )
            return self._error_result(str(e))

    # ─────────────────────────────────────────────────────────────────
    # 各模型实现
    # ─────────────────────────────────────────────────────────────────

    def _openai_vision(self, image_data: str, query: str) -> Dict[str, Any]:
        """OpenAI GPT-4o Vision 实现。"""
        # 处理 Base64 前缀
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )

        description = response.choices[0].message.content

        return {
            "description": description,
            "extracted_data": {},
            "raw_text": description,
            "confidence": 0.9,
        }

    def _qwen_vl(self, image_data: str, query: str) -> Dict[str, Any]:
        """vLLM 部署的 Qwen3.5-VL 实现（OpenAI 兼容接口）。"""
        # 处理 Base64 前缀
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=2048,
        )

        description = response.choices[0].message.content

        return {
            "description": description,
            "extracted_data": {},
            "raw_text": description,
            "confidence": 0.85,
        }

    def _claude_vision(self, image_data: str, query: str) -> Dict[str, Any]:
        """Claude 3 Vision 实现。"""
        # 处理 Base64 前缀
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]

        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": query},
                    ],
                }
            ],
        )

        description = response.content[0].text

        return {
            "description": description,
            "extracted_data": {},
            "raw_text": description,
            "confidence": 0.9,
        }

    # ─────────────────────────────────────────────────────────────────
    # Mock 实现（无 API Key 时）
    # ─────────────────────────────────────────────────────────────────

    def _mock_analyze(self, image_data: str, query: str) -> Dict[str, Any]:
        """Mock 图像分析。"""
        return {
            "description": f"[Mock] 图像分析结果。原始查询：{query}",
            "extracted_data": {},
            "raw_text": "Mock OCR: ERR_CONNECTION_REFUSED at line 42",
            "confidence": 0.5,
        }

    # ─────────────────────────────────────────────────────────────────
    # 错误处理
    # ─────────────────────────────────────────────────────────────────

    def _error_result(self, error_msg: str) -> Dict[str, Any]:
        """错误结果。"""
        return {
            "description": "",
            "extracted_data": {},
            "raw_text": "",
            "confidence": 0.0,
            "error": error_msg,
        }


# =============================================================================
# 全局单例
# =============================================================================

_multimodal_processor: Optional[MultimodalProcessor] = None


def get_multimodal_processor() -> MultimodalProcessor:
    """获取多模态处理器单例。"""
    global _multimodal_processor

    if _multimodal_processor is None:
        from config import settings

        # 从配置读取模型提供商
        provider = getattr(settings, 'multimodal_provider', 'mock')
        model_name = getattr(settings, 'multimodal_model', None)

        _multimodal_processor = MultimodalProcessor(
            model_provider=provider,
            model_name=model_name,
        )

    return _multimodal_processor
