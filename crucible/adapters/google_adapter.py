from __future__ import annotations

import copy
from typing import Any

from crucible.mpsr import strip_markdown_fences_for_json
from crucible.schemas import MpsrPanelOutput


def _make_google_response_schema(pydantic_cls: type) -> dict:
    """Convert a Pydantic model JSON schema into a Google API-compatible schema dict.

    Google's generation_config.response_schema does not accept:
    - additionalProperties
    - title
    - $ref / $defs (must be inlined)
    """
    raw = copy.deepcopy(pydantic_cls.model_json_schema())
    defs = raw.pop("$defs", {})

    def resolve_refs(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                return resolve_refs(copy.deepcopy(defs.get(ref_name, {})))
            return {k: resolve_refs(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve_refs(v) for v in obj]
        return obj

    schema = resolve_refs(raw)

    def strip_unsupported(obj: Any) -> None:
        if isinstance(obj, dict):
            obj.pop("additionalProperties", None)
            obj.pop("title", None)
            for v in obj.values():
                strip_unsupported(v)
        elif isinstance(obj, list):
            for v in obj:
                strip_unsupported(v)

    strip_unsupported(schema)
    return schema


class GoogleAdapter:
    """Google Generative AI calls using the google-genai SDK."""

    def __init__(self, api_key: str, model_string: str) -> None:
        try:
            from google import genai
            from google.genai import types as _types
        except ImportError as e:
            raise ImportError(
                "google-genai package required: pip install google-genai"
            ) from e
        self._client = genai.Client(api_key=api_key)
        self._types = _types
        self.model_string = model_string
        self.provider_name = "google"
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0

    def _generate(self, system: str, user: str, max_tokens: int = 4096,
                  json_mode: bool = False) -> str:
        config_kwargs: dict = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        response = self._client.models.generate_content(
            model=self.model_string,
            contents=user,
            config=self._types.GenerateContentConfig(**config_kwargs),
        )
        usage = response.usage_metadata
        if usage:
            self.last_input_tokens = usage.prompt_token_count or 0
            self.last_output_tokens = usage.candidates_token_count or 0
        return response.text or ""

    def call_mpsr(self, system: str, user: str,
                  max_tokens: int = 8192) -> MpsrPanelOutput:
        config_kwargs: dict = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
            "response_schema": _make_google_response_schema(MpsrPanelOutput),
        }
        response = self._client.models.generate_content(
            model=self.model_string,
            contents=user,
            config=self._types.GenerateContentConfig(**config_kwargs),
        )
        usage = response.usage_metadata
        if usage:
            self.last_input_tokens = usage.prompt_token_count or 0
            self.last_output_tokens = usage.candidates_token_count or 0
        text = response.text or ""
        try:
            return MpsrPanelOutput.model_validate_json(text)
        except Exception:
            cleaned = strip_markdown_fences_for_json(text)
            return MpsrPanelOutput.model_validate_json(cleaned)

    def call_structured(self, schema_cls: type, system: str, user: str,
                        max_tokens: int = 4096) -> Any:
        config_kwargs: dict = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
            "response_schema": _make_google_response_schema(schema_cls),
        }
        response = self._client.models.generate_content(
            model=self.model_string,
            contents=user,
            config=self._types.GenerateContentConfig(**config_kwargs),
        )
        usage = response.usage_metadata
        if usage:
            self.last_input_tokens = usage.prompt_token_count or 0
            self.last_output_tokens = usage.candidates_token_count or 0
        text = response.text or ""
        try:
            return schema_cls.model_validate_json(text)
        except Exception:
            cleaned = strip_markdown_fences_for_json(text)
            return schema_cls.model_validate_json(cleaned)

    # Alias for orchestrator uniformity
    def call_structured_su(self, schema_cls: type, system: str, user: str,
                           max_tokens: int = 4096) -> Any:
        return self.call_structured(schema_cls, system, user, max_tokens)

    def call_verdict(self, system: str, user: str, max_tokens: int = 8192) -> str:
        return self._generate(system, user, max_tokens, json_mode=False)

    def call_text(self, system: str, user: str, max_tokens: int = 4096) -> str:
        return self._generate(system, user, max_tokens, json_mode=False)
