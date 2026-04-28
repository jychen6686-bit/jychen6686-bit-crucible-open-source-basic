from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from crucible.schemas import MpsrPanelOutput


@runtime_checkable
class MpsrProvider(Protocol):
    """Callable protocol for MPSR panel dispatch_fn."""
    def __call__(self, provider_config: Any) -> MpsrPanelOutput: ...


@runtime_checkable
class StructuredOutputProvider(Protocol):
    """Calls a model and returns a validated Pydantic model instance."""
    def call_structured(self, schema_cls: type, system: str, user: str) -> Any: ...


@runtime_checkable
class TextOutputProvider(Protocol):
    """Calls a model and returns raw text."""
    def call_text(self, system: str, user: str) -> str: ...
