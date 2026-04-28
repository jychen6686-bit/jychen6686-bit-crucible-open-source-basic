from __future__ import annotations

from typing import Any

from crucible.mpsr import strip_markdown_fences_for_json
from crucible.schemas import MpsrPanelOutput


class OpenAIAdapter:
    """Real OpenAI API calls with structured JSON output."""

    def __init__(self, api_key: str, model_string: str,
                 reasoning_effort: str = "high") -> None:
        try:
            import openai as _openai
        except ImportError as e:
            raise ImportError("openai package required: pip install openai") from e
        self._client = _openai.OpenAI(api_key=api_key)
        self.model_string = model_string
        self.reasoning_effort = reasoning_effort
        self.provider_name = "openai"
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            self.last_input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            self.last_output_tokens = getattr(usage, "completion_tokens", 0) or 0

    def _build_messages(self, system: str, user: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def call_mpsr(self, system: str, user: str, max_tokens: int = 8192) -> MpsrPanelOutput:
        response = self._client.beta.chat.completions.parse(
            model=self.model_string,
            max_tokens=max_tokens,
            messages=self._build_messages(system, user),
            response_format=MpsrPanelOutput,
        )
        self._record_usage(response)
        parsed = response.choices[0].message.parsed
        if parsed is not None:
            return parsed
        text = response.choices[0].message.content or ""
        try:
            return MpsrPanelOutput.model_validate_json(text)
        except Exception:
            cleaned = strip_markdown_fences_for_json(text)
            return MpsrPanelOutput.model_validate_json(cleaned)

    def call_structured(self, schema_cls: type, payload: dict) -> Any:
        response = self._client.chat.completions.create(**payload)
        self._record_usage(response)
        text = response.choices[0].message.content or ""
        try:
            return schema_cls.model_validate_json(text)
        except Exception:
            cleaned = strip_markdown_fences_for_json(text)
            return schema_cls.model_validate_json(cleaned)

    def call_structured_su(self, schema_cls: type, system: str, user: str,
                           max_tokens: int = 4096) -> Any:
        response = self._client.beta.chat.completions.parse(
            model=self.model_string,
            max_tokens=max_tokens,
            messages=self._build_messages(system, user),
            response_format=schema_cls,
        )
        self._record_usage(response)
        parsed = response.choices[0].message.parsed
        if parsed is not None:
            return parsed
        text = response.choices[0].message.content or ""
        try:
            return schema_cls.model_validate_json(text)
        except Exception:
            cleaned = strip_markdown_fences_for_json(text)
            return schema_cls.model_validate_json(cleaned)

    def call_text(self, system: str, user: str, max_tokens: int = 4096) -> str:
        response = self._client.chat.completions.create(
            model=self.model_string,
            max_tokens=max_tokens,
            messages=self._build_messages(system, user),
        )
        self._record_usage(response)
        return response.choices[0].message.content or ""
