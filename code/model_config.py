import os

# 统一模型配置：只改这里即可影响前后端
FIXED_PLATFORM = os.getenv("APP_LLM_PLATFORM", "aliyun").strip() or "aliyun"
FIXED_MODEL_NAME = os.getenv("APP_LLM_MODEL_NAME", "qwen3.6-plus").strip() or "qwen3.6-plus"
