# ============================================================
# 智能服务台 Agent 微服务 — Dockerfile
# ============================================================
# 多阶段构建：
#   Stage 1 (builder): 安装依赖，编译
#   Stage 2 (runtime): 运行最小化镜像
# ============================================================

# ─────────────────────────────────────────
# Stage 1: Builder（依赖安装）
# ─────────────────────────────────────────
FROM python:3.11-slim as builder

WORKDIR /app

# 安装编译依赖（psycopg2、lxml 等需要编译的包）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 先安装依赖（利用 Docker 缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 复制源码
COPY . .

# ─────────────────────────────────────────
# Stage 2: Runtime（生产运行）
# ─────────────────────────────────────────
FROM python:3.11-slim as runtime

# 安全：使用非 root 用户运行
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# 复制依赖安装结果
COPY --from=builder /root/.local /home/appuser/.local

# 复制源码（排除 .pyc 和 __pycache__）
COPY --from=builder /app /app

# 设置 PYTHONPATH
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安全：切换到非 root 用户
USER appuser

# 健康检查：使用 FastAPI 的 /health/live 端点
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')" || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
# 生产环境建议使用 gunicorn + uvicorn workers 替代单进程
# CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "main.py"]
