from __future__ import annotations

from typing import Any

from crucible.cache_routing import TtlMode, call_anthropic_with_ttl_fallback
from crucible.mpsr import strip_markdown_fences_for_json
from crucible.schemas import MpsrPanelOutput


class AnthropicAdapter:
    """Real Anthropic API calls with TTL fallback."""

    def __init__(self, api_key: str, model_string: str, ttl_mode: TtlMode) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as e:
            raise ImportError("anthropic package required: pip install anthropic") from e
        self._client = _anthropic.Anthropic(api_key=api_key)
        self.model_string = model_string
        self.ttl_mode = ttl_mode
        self.provider_name = "anthropic"
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0

    def _telemetry_stub(self, event: str, payload: dict) -> None:
        pass

    def _extract_text(self, response: Any) -> str:
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            self.last_input_tokens = getattr(usage, "input_tokens", 0) or 0
            self.last_output_tokens = getattr(usage, "output_tokens", 0) or 0

    def _build_payload(self, system: str, user: str, max_tokens: int) -> dict:
        return {
            "model": self.model_string,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

    def call_mpsr(self, system: str, user: str, max_tokens: int = 8192) -> MpsrPanelOutput:
        schema = MpsrPanelOutput.model_json_schema()
        payload = {
            "model": self.model_string,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "tools": [{
                "name": "mpsr_panel_output",
                "description": "Output the MPSR panel scoping charter",
                "input_schema": schema,
            }],
            "tool_choice": {"type": "tool", "name": "mpsr_panel_output"},
        }
        response = call_anthropic_with_ttl_fallback(
            client=self._client,
            request_payload=payload,
            ttl_mode=self.ttl_mode,
            telemetry_logger=self._telemetry_stub,
        )
        self._record_usage(response)
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "mpsr_panel_output":
                return MpsrPanelOutput.model_validate(block.input)
        # fallback: try text
        text = self._extract_text(response)
        try:
            return MpsrPanelOutput.model_validate_json(text)
        except Exception:
            cleaned = strip_markdown_fences_for_json(text)
            return MpsrPanelOutput.model_validate_json(cleaned)

    def _call_with_tool(self, schema_cls: type, system: str, user: str, max_tokens: int) -> Any:
        """Use Anthropic tool_use to enforce schema compliance."""
        schema = schema_cls.model_json_schema()
        tool_name = schema_cls.__name__.lower()
        payload = {
            "model": self.model_string,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "tools": [{
                "name": tool_name,
                "description": f"Output the {schema_cls.__name__} structured result",
                "input_schema": schema,
            }],
            "tool_choice": {"type": "tool", "name": tool_name},
        }
        response = call_anthropic_with_ttl_fallback(
            client=self._client,
            request_payload=payload,
            ttl_mode=self.ttl_mode,
            telemetry_logger=self._telemetry_stub,
        )
        self._record_usage(response)
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
                return schema_cls.model_validate(block.input)
        text = self._extract_text(response)
        try:
            return schema_cls.model_validate_json(text)
        except Exception:
            cleaned = strip_markdown_fences_for_json(text)
            return schema_cls.model_validate_json(cleaned)

    def call_structured(self, schema_cls: type, payload: dict) -> Any:
        response = call_anthropic_with_ttl_fallback(
            client=self._client,
            request_payload=payload,
            ttl_mode=self.ttl_mode,
            telemetry_logger=self._telemetry_stub,
        )
        self._record_usage(response)
        text = self._extract_text(response)
        try:
            return schema_cls.model_validate_json(text)
        except Exception:
            cleaned = strip_markdown_fences_for_json(text)
            return schema_cls.model_validate_json(cleaned)

    def call_structured_su(self, schema_cls: type, system: str, user: str,
                           max_tokens: int = 4096) -> Any:
        return self._call_with_tool(schema_cls, system, user, max_tokens)

    def call_text(self, system: str, user: str, max_tokens: int = 4096) -> str:
        payload = self._build_payload(system, user, max_tokens)
        response = call_anthropic_with_ttl_fallback(
            client=self._client,
            request_payload=payload,
            ttl_mode=self.ttl_mode,
            telemetry_logger=self._telemetry_stub,
        )
        self._record_usage(response)
        return self._extract_text(response)
