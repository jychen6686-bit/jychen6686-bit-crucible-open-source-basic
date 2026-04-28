from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Literal

from crucible.enums import CheckpointId, Decision, ModelTier
from crucible.schemas import (
    CachedPrefix, PauseForHumanInput, PauseForHumanResult,
    RebuildDecision, TelemetrySnapshot,
)


# ===========================================================================
# TTL mode (per-run, one-way degradation)
# ===========================================================================

def _is_bad_request_400(exc: Exception) -> bool:
    """Structural 400 detection without inspecting vendor error wording."""
    try:
        import anthropic  # type: ignore
        BadRequestError = getattr(anthropic, "BadRequestError", None)
        if BadRequestError is not None and isinstance(exc, BadRequestError):
            return True
    except ImportError:
        pass

    if getattr(exc, "status_code", None) == 400:
        return True

    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 400:
        return True

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", {})
        if isinstance(err, dict) and err.get("type") == "invalid_request_error":
            return True

    msg = str(exc).lower()
    if "400" in msg:
        return True
    if "invalid_request" in msg or "invalid request" in msg:
        return True
    return False


def _request_contains_cache_control(payload: dict) -> bool:
    """Recursively walk the request payload; True if any cache_control exists."""
    def walk(obj: Any) -> bool:
        if isinstance(obj, dict):
            if "cache_control" in obj:
                return True
            return any(walk(v) for v in obj.values())
        if isinstance(obj, list):
            return any(walk(v) for v in obj)
        return False
    return walk(payload)


def _strip_cache_ttl_param(payload: dict) -> dict:
    """Remove the ttl field from any cache_control blocks in the payload."""
    stripped = copy.deepcopy(payload)

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            cc = obj.get("cache_control")
            if isinstance(cc, dict) and "ttl" in cc:
                del cc["ttl"]
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(stripped)
    return stripped


@dataclass
class TtlMode:
    """Per-run TTL mode tracker. Supports one-way degradation: once degraded to 5m, stays 5m."""
    mode: Literal["1h", "5m"] = "1h"
    _degraded: bool = False
    _lock: Lock = field(default_factory=Lock)

    def degrade_to_5m(self) -> None:
        with self._lock:
            self.mode = "5m"
            self._degraded = True

    @property
    def degraded(self) -> bool:
        return self._degraded


def call_anthropic_with_ttl_fallback(
    client: Any,
    request_payload: dict,
    ttl_mode: TtlMode,
    telemetry_logger: Callable[[str, dict], None],
) -> Any:
    """Wraps an Anthropic API call with one-way TTL degradation."""
    if ttl_mode.mode == "5m":
        request_payload = _strip_cache_ttl_param(request_payload)
        return client.messages.create(**request_payload)

    try:
        return client.messages.create(**request_payload)
    except Exception as exc:
        if not _is_bad_request_400(exc):
            raise
        if not _request_contains_cache_control(request_payload):
            raise
        telemetry_logger("ttl_param_degraded", {
            "error_class": type(exc).__name__,
            "error_excerpt": str(exc)[:300],
            "previous_mode": ttl_mode.mode,
        })
        ttl_mode.degrade_to_5m()
        stripped = _strip_cache_ttl_param(request_payload)
        return client.messages.create(**stripped)


# ===========================================================================
# Cache freshness probe + rebuild
# ===========================================================================

def probe_cache_freshness(
    prefix: CachedPrefix,
    probe_executor: Callable[[CachedPrefix], dict],
) -> bool:
    """Send a 1-token probe; returns True if cache hit, False if miss."""
    try:
        result = probe_executor(prefix)
        return result.get("cache_hit", False)
    except Exception:
        return False


def estimate_rebuild_cost_usd(
    prefix: CachedPrefix,
    pricing: dict[str, float],
    write_premium: float = 1.25,
) -> float:
    """Cost of rewriting the cache prefix."""
    return (prefix.cached_token_count / 1_000_000) * pricing["input"] * write_premium


# ===========================================================================
# Economic routing: remaining pipeline cost estimator
# ===========================================================================

REMAINING_STEP_ESTIMATES: dict[str, tuple[str, str, int, int, bool]] = {
    "framework":               ("anthropic", "flagship", 5000,  4000, True),
    "red_team":                ("openai",    "flagship", 8000,  6000, False),
    "charter_delta_residual":  ("anthropic", "fast",     3000,  1000, True),
    "verdict":                 ("google",    "flagship", 12000, 5000, True),
    "auditor_phase1_per_cite": ("anthropic", "fast",     2000,  500,  True),
    "drafter_b2":              ("google",    "flagship", 10000, 8000, True),
    "drafter_b4":              ("google",    "flagship", 12000, 8000, True),
    "intake":                  ("anthropic", "fast",     3000,  2000, True),
}

PIPELINE_TRACK_A = [
    "mpsr", "framework", "red_team", "charter_delta_residual",
    "verdict", "auditor_phase1_per_cite",
]
PIPELINE_TRACK_B = [
    "mpsr", "intake", "framework", "drafter_b2", "red_team",
    "charter_delta_residual", "drafter_b4", "auditor_phase1_per_cite",
]


def estimate_remaining_pipeline_cost(
    track: Literal["A", "B"],
    current_position: str,
    pricing_table: dict[tuple[str, str], dict[str, float]],
    assume_cache_hit: bool,
    estimated_citation_count: int = 15,
) -> float:
    """Sum estimated costs for steps after current_position in the pipeline."""
    pipeline = PIPELINE_TRACK_A if track == "A" else PIPELINE_TRACK_B
    try:
        current_idx = pipeline.index(current_position)
    except ValueError:
        return 0.0

    remaining = pipeline[current_idx + 1:]
    total = 0.0

    for step_name in remaining:
        if step_name not in REMAINING_STEP_ESTIMATES:
            continue
        provider, tier, in_tok, out_tok, uses_cache = REMAINING_STEP_ESTIMATES[step_name]
        pricing = pricing_table.get((provider, tier))
        if pricing is None or pricing.get("input", -1) < 0:
            continue

        multiplier = estimated_citation_count if step_name == "auditor_phase1_per_cite" else 1

        if uses_cache and assume_cache_hit:
            effective_input_cost = (in_tok / 1_000_000) * pricing["input"] * 0.30
        else:
            effective_input_cost = (in_tok / 1_000_000) * pricing["input"]

        output_cost = (out_tok / 1_000_000) * pricing["output"]
        step_cost = (effective_input_cost + output_cost) * multiplier
        total += step_cost

    return total


def make_rebuild_decision(
    prefix: CachedPrefix,
    is_fresh: bool,
    rebuild_cost_usd: float,
    current_spend_usd: float,
    budget_cap_usd: float,
    overdraft_allowance_usd: float,
    track: Literal["A", "B"],
    current_position: str,
    pricing_table: dict[tuple[str, str], dict[str, float]],
    estimated_citation_count: int = 15,
) -> RebuildDecision:
    """Pure decision function. No side effects. Returns the action to take."""
    if is_fresh:
        return RebuildDecision(cache_key=prefix.cache_key, action="fresh_no_action")

    future_cached = estimate_remaining_pipeline_cost(
        track, current_position, pricing_table,
        assume_cache_hit=True,
        estimated_citation_count=estimated_citation_count,
    )
    future_uncached = estimate_remaining_pipeline_cost(
        track, current_position, pricing_table,
        assume_cache_hit=False,
        estimated_citation_count=estimated_citation_count,
    )

    with_rebuild_total = current_spend_usd + rebuild_cost_usd + future_cached
    without_rebuild_total = current_spend_usd + future_uncached

    if with_rebuild_total > without_rebuild_total:
        return RebuildDecision(
            cache_key=prefix.cache_key,
            action="skip_rebuild_uneconomical",
            rebuild_cost_usd=rebuild_cost_usd,
            with_rebuild_total=with_rebuild_total,
            without_rebuild_total=without_rebuild_total,
        )

    if with_rebuild_total <= budget_cap_usd:
        return RebuildDecision(
            cache_key=prefix.cache_key,
            action="rebuild",
            rebuild_cost_usd=rebuild_cost_usd,
            with_rebuild_total=with_rebuild_total,
            without_rebuild_total=without_rebuild_total,
        )

    overdraft_amount = with_rebuild_total - budget_cap_usd
    if overdraft_amount <= overdraft_allowance_usd:
        return RebuildDecision(
            cache_key=prefix.cache_key,
            action="rebuild_with_overdraft",
            rebuild_cost_usd=rebuild_cost_usd,
            with_rebuild_total=with_rebuild_total,
            without_rebuild_total=without_rebuild_total,
            overdraft_amount_usd=overdraft_amount,
        )

    return RebuildDecision(
        cache_key=prefix.cache_key,
        action="pause_before_rebuild",
        rebuild_cost_usd=rebuild_cost_usd,
        with_rebuild_total=with_rebuild_total,
        without_rebuild_total=without_rebuild_total,
        overdraft_amount_usd=overdraft_amount,
    )


def on_resume_from_checkpoint(
    run_state: Any,
    active_prefixes: list[CachedPrefix],
    probe_executor_by_provider: dict[str, Callable[[CachedPrefix], dict]],
    rebuild_executor_by_provider: dict[str, Callable[[CachedPrefix], dict]],
    pause_handler: Callable[[PauseForHumanInput], PauseForHumanResult],
    track: Literal["A", "B"],
    current_position: str,
) -> list[RebuildDecision]:
    """Called by the orchestrator after pause_for_human returns from any checkpoint."""
    decisions: list[RebuildDecision] = []

    for prefix in active_prefixes:
        probe_exec = probe_executor_by_provider.get(prefix.provider)
        if probe_exec is None:
            continue

        is_fresh = probe_cache_freshness(prefix, probe_exec)

        # Bug 3 fix: rebuild cost at the tier of the DOWNSTREAM CONSUMER
        pricing_key = (prefix.provider, prefix.tier.value)
        pricing = run_state.pricing_table.get(pricing_key)
        if pricing is None:
            continue

        rebuild_cost = estimate_rebuild_cost_usd(
            prefix, pricing,
            write_premium=2.0 if prefix.ttl_mode == "1h" else 1.25,
        )

        decision = make_rebuild_decision(
            prefix=prefix,
            is_fresh=is_fresh,
            rebuild_cost_usd=rebuild_cost,
            current_spend_usd=run_state.spend_usd,
            budget_cap_usd=run_state.config.budget_cap_usd,
            overdraft_allowance_usd=run_state.config.budget_cap_overdraft_allowance_usd,
            track=track,
            current_position=current_position,
            pricing_table=run_state.pricing_table,
        )
        decisions.append(decision)

        rebuild_exec = rebuild_executor_by_provider.get(prefix.provider)

        if decision.action == "fresh_no_action":
            continue

        elif decision.action == "skip_rebuild_uneconomical":
            run_state.telemetry_log("rebuild_skipped_uneconomical", {
                "cache_key": prefix.cache_key,
                "with_rebuild_total": decision.with_rebuild_total,
                "without_rebuild_total": decision.without_rebuild_total,
            })

        elif decision.action == "rebuild":
            if rebuild_exec is not None:
                rebuild_exec(prefix)
                run_state.spend_usd += decision.rebuild_cost_usd
                run_state.cache_rebuilds_count += 1

        elif decision.action == "rebuild_with_overdraft":
            if rebuild_exec is not None:
                rebuild_exec(prefix)
                run_state.spend_usd += decision.rebuild_cost_usd
                run_state.cache_rebuilds_count += 1
                run_state.cache_rebuild_overdraft_total_usd += decision.overdraft_amount_usd
                run_state.telemetry_log("cache_rebuild_overdraft", {
                    "cache_key": prefix.cache_key,
                    "rebuild_cost_usd": round(decision.rebuild_cost_usd, 4),
                    "overdraft_amount_usd": round(decision.overdraft_amount_usd, 4),
                    "total_spend_usd": round(run_state.spend_usd, 4),
                    "budget_cap_usd": run_state.config.budget_cap_usd,
                })

        elif decision.action == "pause_before_rebuild":
            result = pause_handler(PauseForHumanInput(
                checkpoint_id=CheckpointId.CACHE_REBUILD_OVERDRAFT_AUTHORIZE,
                summary=(
                    f"Cache rebuild for {prefix.cache_key} would cost "
                    f"${decision.rebuild_cost_usd:.2f}, which would push total spend "
                    f"${decision.overdraft_amount_usd:.2f} over the "
                    f"${run_state.config.budget_cap_usd:.2f} cap — exceeding the "
                    f"${run_state.config.budget_cap_overdraft_allowance_usd:.2f} "
                    f"overdraft allowance. Approve rebuild + extend budget, "
                    f"continue uncached, or abort?"
                ),
                artifact_refs=[],
                decision_options=[
                    Decision.EXTEND_BUDGET,
                    Decision.CONTINUE_UNCACHED,
                    Decision.ABORT,
                ],
                edits_allowed_on=[],
                telemetry=run_state.snapshot(),
            ))
            if result.decision == Decision.EXTEND_BUDGET and rebuild_exec is not None:
                rebuild_exec(prefix)
                run_state.spend_usd += decision.rebuild_cost_usd
                run_state.cache_rebuilds_count += 1
                run_state.config.budget_cap_usd += (result.budget_extension_usd or 0.0)

    return decisions
