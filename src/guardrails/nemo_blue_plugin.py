"""
Blue — lớp NVIDIA NeMo Guardrails dưới dạng ADK plugin (tuỳ chọn, bật bằng use_nemo=True).

Vị trí trong pipeline:  RateLimit -> InputGuardrail -> **NemoBluePlugin** -> LLM -> Output
  - on_user_message_callback : NeMo input rails  (trước khi gọi LLM)
  - after_model_callback     : NeMo output rails (sau khi LLM trả lời)

Mặc định chỉ chạy các rail **tất định** (không gọi LLM): kiểm tra giá trị bí mật trong
input và rò rỉ trong output, kể cả khi bị mã hoá base64 / hex / ROT13 / đảo ngược.
Nhờ vậy không tốn thêm quota của model free OpenRouter.

Bật thêm "self check" (LLM phân loại) bằng biến môi trường ``NEMO_BLUE_SELF_CHECK=1``.
Lưu ý: model Blue là LFM 1.7B free nên chất lượng phân loại thấp và tốn thêm 1-2 request
mỗi lượt, dễ gây 429.

Fail-closed: NeMo lỗi thì request bị chặn (giống Red Advance).
Tái sử dụng NemoRailsGuard / Colang / custom actions của ``agents.red_advance_nemo``.
"""
from __future__ import annotations

import contextvars
import os

from google.adk.plugins import base_plugin
from google.genai import types

from agents.red_advance_nemo import (
    INPUT_RAIL_PROTECTED,
    INPUT_RAIL_SELF_CHECK,
    OUTPUT_RAIL_LEAK,
    OUTPUT_RAIL_SELF_CHECK,
    SELF_CHECK_INPUT_PROMPT,
    SELF_CHECK_OUTPUT_PROMPT,
    COLANG_CONTENT,
    NemoRailsGuard,
    require_nemo,
)
from core.config import blue_client_kwargs, get_blue_model

# Câu hỏi của người dùng ở lượt hiện tại: input và output callback chạy trong cùng một
# task asyncio, nên ContextVar không bị lẫn giữa các người dùng đồng thời.
_current_user_text: contextvars.ContextVar[str] = contextvars.ContextVar(
    "nemo_blue_user_text", default=""
)


def self_check_enabled() -> bool:
    return os.environ.get("NEMO_BLUE_SELF_CHECK", "").strip().lower() in {"1", "true", "yes"}


def build_blue_rails_dict(*, self_check: bool) -> dict:
    """Config NeMo cho Blue (dạng dict, tương đương config.yml)."""
    input_flows = [INPUT_RAIL_PROTECTED]
    output_flows = [OUTPUT_RAIL_LEAK]
    config: dict = {"rails": {"input": {"flows": input_flows}, "output": {"flows": output_flows}}}
    if self_check:
        input_flows.append(INPUT_RAIL_SELF_CHECK)
        output_flows.append(OUTPUT_RAIL_SELF_CHECK)
        kwargs = blue_client_kwargs()
        config["models"] = [{
            "type": "main",
            "engine": "openai",
            "model": get_blue_model(),
            "api_key_env_var": "OPENROUTER_API_KEY",
            "parameters": {"base_url": kwargs["base_url"]},
        }]
        config["prompts"] = [
            {"task": "self_check_input", "content": SELF_CHECK_INPUT_PROMPT},
            {"task": "self_check_output", "content": SELF_CHECK_OUTPUT_PROMPT},
        ]
    return config


def build_blue_guard(*, self_check: bool | None = None) -> NemoRailsGuard:
    require_nemo()
    from nemoguardrails import RailsConfig

    if self_check is None:
        self_check = self_check_enabled()
    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT, config=build_blue_rails_dict(self_check=self_check)
    )
    return NemoRailsGuard(config)


def _content_text(content) -> str:
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(p, "text", "") or "" for p in parts)


class NemoBluePlugin(base_plugin.BasePlugin):
    """Chạy NeMo input/output rails quanh LLM của Blue."""

    def __init__(self, guard=None, *, self_check: bool | None = None):
        super().__init__(name="nemo_blue_guard")
        self.guard = guard if guard is not None else build_blue_guard(self_check=self_check)
        self.total_count = 0
        self.blocked_count = 0
        self.output_blocked_count = 0

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Trả Content để chặn, None để cho qua."""
        self.total_count += 1
        text = _content_text(user_message)
        _current_user_text.set(text)
        verdict = await self.guard.check_input(text)
        if verdict.allowed:
            return None
        self.blocked_count += 1
        return types.Content(role="model", parts=[types.Part.from_text(text=verdict.message)])

    async def after_model_callback(self, *, callback_context, llm_response):
        """Thay câu trả lời bằng thông báo cố định nếu output rail chặn."""
        text = _content_text(getattr(llm_response, "content", None))
        if not text:
            return llm_response
        verdict = await self.guard.check_output(_current_user_text.get(), text)
        if not verdict.allowed:
            self.output_blocked_count += 1
            llm_response.content = types.Content(
                role="model", parts=[types.Part.from_text(text=verdict.message)]
            )
        return llm_response
