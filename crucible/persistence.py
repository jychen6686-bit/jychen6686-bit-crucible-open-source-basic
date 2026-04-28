from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from crucible.cache_routing import TtlMode
from crucible.schemas import RateLimitBucket, RateLimitConfig
from crucible.mpsr import _shutdown_executor_safely, create_managed_executor


# ===========================================================================
# Storage backend abstraction
# ===========================================================================

@runtime_checkable
class StorageBackend(Protocol):
    def write_bytes(self, key: str, data: bytes) -> None: ...
    def read_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def atomic_write(self, key: str, data: bytes) -> None: ...


class LocalStorageBackend:
    """Default: write to local filesystem with atomic rename."""
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_bytes(self, key: str, data: bytes) -> None:
        (self.base_dir / key).write_bytes(data)

    def read_bytes(self, key: str) -> bytes:
        return (self.base_dir / key).read_bytes()

    def exists(self, key: str) -> bool:
        return (self.base_dir / key).exists()

    def atomic_write(self, key: str, data: bytes) -> None:
        final = self.base_dir / key
        tmp = self.base_dir / f"{key}.tmp"
        tmp.write_bytes(data)
        tmp.replace(final)


class S3StorageBackend:
    """Optional production backend. Requires boto3."""
    def __init__(self, bucket: str, prefix: str) -> None:
        try:
            import boto3  # type: ignore
        except ImportError as e:
            raise ImportError("S3StorageBackend requires boto3: pip install boto3") from e
        self.s3 = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def write_bytes(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self._full_key(key), Body=data)

    def read_bytes(self, key: str) -> bytes:
        resp = self.s3.get_object(Bucket=self.bucket, Key=self._full_key(key))
        return resp["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._full_key(key))
            return True
        except Exception:
            return False

    def atomic_write(self, key: str, data: bytes) -> None:
        self.write_bytes(key, data)


# ===========================================================================
# Persist field list
# ===========================================================================

_PERSIST_FIELDS = (
    "run_id",
    "spend_usd",
    "subagent_invocations",
    "same_subagent_counts",
    "last_subagent",
    "consecutive_same_subagent",
    "tool_calls",
    "last_checkpoint",
    "cache_rebuilds_count",
    "cache_rebuild_overdraft_total_usd",
    "track",
    "wall_clock_started_at",
)


def _persist_run_state(run_state: Any, checkpoint_dir: Any) -> Path:
    """Serialize RunState to {checkpoint_dir}/checkpoint.json atomically."""
    from crucible.runtime import RunState
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    session_elapsed = time.monotonic() - run_state.started_at
    cumulative = run_state.cumulative_active_seconds + session_elapsed

    cached_prefix_metadata: list[dict] = []

    config_dict = {}
    for f_name in run_state.config.__dataclass_fields__:
        val = getattr(run_state.config, f_name)
        # Serialize Pydantic models nested inside the dataclass
        config_dict[f_name] = val.model_dump() if hasattr(val, "model_dump") else val

    pricing_serialized = {
        f"{provider}|{tier}": pricing
        for (provider, tier), pricing in run_state.pricing_table.items()
    }

    payload: dict[str, Any] = {
        "schema_version": "crucible_runstate_v1",
        "persisted_at_iso": datetime.now(timezone.utc).isoformat(),
        "cumulative_active_seconds": cumulative,
        "ttl_mode_value": run_state.ttl_mode.mode,
        "ttl_mode_degraded": run_state.ttl_mode.degraded,
        "config": config_dict,
        "pricing_table": pricing_serialized,
        "rag_cache_stats": {
            "hits": run_state.rag_cache.hits,
            "misses": run_state.rag_cache.misses,
        },
        "cached_prefix_metadata": cached_prefix_metadata,
        "artifacts": run_state.artifacts,
    }
    for name in _PERSIST_FIELDS:
        payload[name] = getattr(run_state, name)

    final_path = checkpoint_dir / "checkpoint.json"
    temp_path = checkpoint_dir / "checkpoint.json.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    temp_path.replace(final_path)

    run_state.telemetry_log("checkpoint_persisted", {
        "path": str(final_path),
        "artifact_count": len(run_state.artifacts),
        "spend_usd": round(run_state.spend_usd, 4),
        "cumulative_active_seconds": round(cumulative, 1),
    })
    return final_path


def restore_from_checkpoint(checkpoint_path: Any) -> Any:
    """Read a checkpoint.json and rebuild a RunState ready to resume."""
    from crucible.runtime import RagCache, RunConfig, RunState

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")

    with open(checkpoint_path, encoding="utf-8") as f:
        payload = json.load(f)

    schema = payload.get("schema_version")
    if schema != "crucible_runstate_v1":
        raise ValueError(
            f"Unknown checkpoint schema_version: {schema!r}. "
            f"Expected 'crucible_runstate_v1'. Refusing to restore — "
            f"silent schema drift would corrupt run state."
        )

    config_dict = dict(payload["config"])
    config_dict.pop("keepalive_limits", None)  # v2.x compat: ignore if present
    # Reconstruct nested Pydantic models
    if "assumptions" in config_dict and isinstance(config_dict["assumptions"], dict):
        from crucible.schemas import EngagementAssumptions
        config_dict["assumptions"] = EngagementAssumptions(**config_dict["assumptions"])
    config = RunConfig(**config_dict)

    pricing_table: dict[tuple[str, str], dict[str, float]] = {}
    for joined_key, pricing in payload["pricing_table"].items():
        provider, tier = joined_key.split("|", 1)
        pricing_table[(provider, tier)] = pricing

    rate_buckets: dict[str, RateLimitBucket] = {}

    rag_cache = RagCache()
    rag_cache.hits = payload["rag_cache_stats"]["hits"]
    rag_cache.misses = payload["rag_cache_stats"]["misses"]

    ttl_mode = TtlMode(mode=payload["ttl_mode_value"])
    if payload["ttl_mode_degraded"]:
        ttl_mode.degrade_to_5m()

    state = RunState(
        config=config,
        pricing_table=pricing_table,
        run_id=payload["run_id"],
        started_at=time.monotonic(),
        spend_usd=payload["spend_usd"],
        subagent_invocations=payload["subagent_invocations"],
        same_subagent_counts=payload["same_subagent_counts"],
        last_subagent=payload["last_subagent"],
        consecutive_same_subagent=payload["consecutive_same_subagent"],
        tool_calls=payload["tool_calls"],
        last_checkpoint=payload["last_checkpoint"],
        rag_cache=rag_cache,
        rate_buckets=rate_buckets,
        ttl_mode=ttl_mode,
        cache_rebuilds_count=payload["cache_rebuilds_count"],
        cache_rebuild_overdraft_total_usd=payload["cache_rebuild_overdraft_total_usd"],
        track=payload["track"],
        artifacts=payload["artifacts"],
        wall_clock_started_at=payload["wall_clock_started_at"],
        cumulative_active_seconds=payload["cumulative_active_seconds"],
    )

    state.telemetry_log("checkpoint_restored", {
        "path": str(checkpoint_path),
        "spend_usd": state.spend_usd,
        "cumulative_active_seconds": state.cumulative_active_seconds,
        "artifact_count": len(state.artifacts),
        "ttl_mode_degraded": ttl_mode.degraded,
    })
    return state


def run_pipeline_with_persistence(
    config: Any,
    pricing_table: dict[tuple[str, str], dict[str, float]],
    output_base_path: Any,
    resume_from: Path | None = None,
    mpsr_max_workers: int = 3,
) -> Any:
    """Top-level orchestrator wrapper with bulletproof try/finally persistence."""
    from crucible.runtime import RunState
    output_base = Path(output_base_path)

    if resume_from is not None and resume_from.exists():
        run_state = restore_from_checkpoint(resume_from)
        run_dir = output_base / "runs" / run_state.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_state = RunState(config=config, pricing_table=pricing_table)
        run_dir = output_base / "runs" / run_state.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir
    artifacts_dir = run_dir / "artifacts"

    executor = create_managed_executor(max_workers=mpsr_max_workers)

    try:
        return run_state

    except BaseException:
        _shutdown_executor_safely(executor, wait=False, cancel_futures=True)
        raise

    finally:
        _shutdown_executor_safely(executor, wait=True, cancel_futures=False)

        try:
            run_state.persist(checkpoint_dir)
        except Exception as exc:
            run_state.telemetry_log("persist_failed_in_finally", {
                "error": str(exc)[:500],
            })
        try:
            run_state.flush_partial_artifacts(artifacts_dir)
        except Exception as exc:
            run_state.telemetry_log("flush_failed_in_finally", {
                "error": str(exc)[:500],
            })
