"""Resolve dedicated Qwen3.8 credentials with DashScope-compatible fallback."""

from __future__ import annotations


DEFAULT_QWEN38_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def qwen_credentials(env: dict, default_model: str) -> tuple[str, str, str]:
    dedicated_key = str(env.get("QWEN38_API_KEY") or "").strip()
    dedicated_url = str(env.get("QWEN38_BASE_URL") or "").strip()
    dedicated_model = str(env.get("QWEN38_MODEL") or "").strip()
    dedicated = (dedicated_key, dedicated_url, dedicated_model)
    if any(dedicated):
        if not all(dedicated):
            raise RuntimeError(
                "QWEN38_API_KEY, QWEN38_BASE_URL and QWEN38_MODEL "
                "must be configured together"
            )
        return dedicated_model, dedicated_key, dedicated_url

    key = str(env.get("DASHSCOPE_API_KEY") or "").strip()
    base_url = str(env.get("DASHSCOPE_BASE_URL") or DEFAULT_QWEN38_BASE_URL).strip()
    model = str(default_model).strip()
    if not key:
        raise RuntimeError("QWEN38_API_KEY or DASHSCOPE_API_KEY is required")
    return model, key, base_url
