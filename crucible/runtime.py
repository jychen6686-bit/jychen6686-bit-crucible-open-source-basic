from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal

from crucible.schemas import (
    EngagementAssumptions,
    RagQueryInput, RagQueryResult, RateLimitBucket, RateLimitConfig, TelemetrySnapshot,
)
from crucible.normalization import normalize_citation
from crucible.cache_routing import TtlMode


# ===========================================================================
# In-run RAG cache
# ===========================================================================

class RagCache:
    def __init__(self) -> None:
        self._cache: dict[tuple, RagQueryResult] = {}
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def _key(self, query: RagQueryInput, jurisdiction: str) -> tuple:
        normalized_keywords = tuple(sorted(
            normalize_citation(k, jurisdiction) if any(c.isdigit() for c in k) else k.lower().strip()
            for k in query.keywords
        ))
        return (query.layer, normalized_keywords, frozenset(query.scope_tags), query.follow_references)

    def get(self, query: RagQueryInput, jurisdiction: str) -> RagQueryResult | None:
        with self._lock:
            result = self._cache.get(self._key(query, jurisdiction))
            if result is not None:
                self.hits += 1
            else:
                self.misses += 1
            return result

    def put(self, query: RagQueryInput, jurisdiction: str, result: RagQueryResult) -> None:
        with self._lock:
            self._cache[self._key(query, jurisdiction)] = result

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0


# ===========================================================================
# Run configuration
# ===========================================================================

@dataclass
class RunConfig:
    jurisdiction: str
    jurisdiction_language: str
    report_language_default: str | None
    budget_cap_usd: float = 25.0
    budget_cap_overdraft_allowance_usd: float = 5.0
    assumptions: EngagementAssumptions = field(default_factory=EngagementAssumptions)
    wall_clock_cap_seconds: int = 7200
    subagent_invocations_cap: int = 40
    auditor_phase1_dispatches_cap: int = 100
    same_subagent_cap: int = 20
    consecutive_same_subagent_cap: int = 25
    tool_calls_cap: int = 200
    output_tokens_per_call_cap: int = 16000
    max_retries_per_call: int = 2
    backoff_base_seconds: float = 2.0
    backoff_cap_seconds: float = 30.0
    subagent_soft_timeout_seconds: int = 480
    subagent_hard_timeout_seconds: int = 720
    safety_margin: float = 1.5


# ===========================================================================
# Run state
# ===========================================================================

@dataclass
class RunState:
    config: RunConfig
    pricing_table: dict[tuple[str, str], dict[str, float]]
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.monotonic)
    spend_usd: float = 0.0
    subagent_invocations: int = 0
    same_subagent_counts: dict[str, int] = field(default_factory=dict)
    last_subagent: str | None = None
    consecutive_same_subagent: int = 0
    tool_calls: int = 0
    last_checkpoint: str | None = None
    rag_cache: RagCache = field(default_factory=RagCache)
    rate_buckets: dict[str, RateLimitBucket] = field(default_factory=dict)
    ttl_mode: TtlMode = field(default_factory=TtlMode)
    cache_rebuilds_count: int = 0
    cache_rebuild_overdraft_total_usd: float = 0.0
    auditor_phase1_dispatches: int = 0
    track: Literal["A", "B"] = "A"
    artifacts: dict[str, Any] = field(default_factory=dict)
    wall_clock_started_at: float = field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp()
    )
    cumulative_active_seconds: float = 0.0

    def telemetry_log(self, event: str, payload: dict) -> None:
        """Stub. Host implementation writes to JSONL log."""
        pass

    def elapsed(self) -> float:
        """Total active seconds in this run, summed across all process sessions."""
        session_elapsed = time.monotonic() - self.started_at
        return self.cumulative_active_seconds + session_elapsed

    def snapshot(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            spend_usd=self.spend_usd,
            elapsed_seconds=self.elapsed(),
            subagent_invocations=self.subagent_invocations,
            tool_calls=self.tool_calls,
            last_subagent=self.last_subagent,
            last_checkpoint=self.last_checkpoint,
            cap_usd=self.config.budget_cap_usd,
            cap_wall_seconds=self.config.wall_clock_cap_seconds,
            rag_cache_hits=self.rag_cache.hits,
            rag_cache_misses=self.rag_cache.misses,
            cache_rebuilds_count=self.cache_rebuilds_count,
            cache_rebuild_overdraft_total_usd=self.cache_rebuild_overdraft_total_usd,
            ttl_mode_degraded=self.ttl_mode.degraded,
        )

    def record_dispatch(self, subagent: str, cost_usd: float) -> None:
        self.subagent_invocations += 1
        self.same_subagent_counts[subagent] = self.same_subagent_counts.get(subagent, 0) + 1
        self.spend_usd += cost_usd
        if self.last_subagent == subagent:
            self.consecutive_same_subagent += 1
        else:
            self.consecutive_same_subagent = 1
        self.last_subagent = subagent

    def check_loop_and_caps(self) -> str | None:
        hard_ceiling = self.config.budget_cap_usd + self.config.budget_cap_overdraft_allowance_usd
        if self.spend_usd > hard_ceiling:
            return f"Hard ceiling exceeded: spent ${self.spend_usd:.2f} > ${hard_ceiling:.2f} (cap + overdraft)."
        if self.elapsed() >= self.config.wall_clock_cap_seconds:
            return f"Wall clock cap {self.config.wall_clock_cap_seconds}s reached."
        if self.subagent_invocations >= self.config.subagent_invocations_cap:
            return f"Subagent invocation cap {self.config.subagent_invocations_cap} reached."
        if self.auditor_phase1_dispatches >= self.config.auditor_phase1_dispatches_cap:
            return f"Auditor Phase 1 dispatch cap {self.config.auditor_phase1_dispatches_cap} reached."
        if self.tool_calls >= self.config.tool_calls_cap:
            return f"Tool call cap {self.config.tool_calls_cap} reached."
        for name, n in self.same_subagent_counts.items():
            if n >= self.config.same_subagent_cap:
                return f"Subagent {name!r} invoked {n} times (cap {self.config.same_subagent_cap})."
        if self.consecutive_same_subagent >= self.config.consecutive_same_subagent_cap:
            return f"Loop: {self.last_subagent!r} called {self.consecutive_same_subagent}× consecutively."
        return None

    def record_artifact(self, name: str, content: Any) -> None:
        """Stash a structured output in RunState so it survives crashes."""
        self.artifacts[name] = content

    def persist(self, checkpoint_dir: Any) -> Any:
        """Serialize RunState to checkpoint. Imported from persistence module."""
        from crucible.persistence import _persist_run_state
        return _persist_run_state(self, checkpoint_dir)

    def flush_partial_artifacts(self, output_dir: Any) -> list:
        """On hard halt, dump every artifact to disk."""
        import json
        from pathlib import Path
        output_dir = Path(output_dir)
        written: list = []
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.telemetry_log("flush_artifacts_mkdir_failed", {"error": str(exc)[:300]})
            return written

        for name, content in self.artifacts.items():
            try:
                if isinstance(content, str):
                    path = output_dir / f"{name}_partial.md"
                    path.write_text(content, encoding="utf-8")
                else:
                    path = output_dir / f"{name}_partial.json"
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(content, f, indent=2, ensure_ascii=False, default=str)
                written.append(path)
            except Exception as exc:
                self.telemetry_log("flush_artifact_failed", {
                    "name": name,
                    "error": str(exc)[:300],
                })
                continue

        self.telemetry_log("flush_partial_artifacts_complete", {
            "files_written": [str(p) for p in written],
            "total_artifacts": len(self.artifacts),
        })
        return written
