from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError,
    as_completed,
)
from typing import Any, Callable, Literal

from crucible.enums import CheckpointId, ConsensusLevel, Decision, ModelTier, RetryClass
from crucible.normalization import normalize_citation
from crucible.schemas import (
    BudgetCheckInput, BudgetCheckResult,
    CharterEntry, CharterDeltaOutput,
    MpsrPanelDispatchResult, MpsrPanelOutput, MpsrProviderResult,
    PauseForHumanInput, PauseForHumanResult,
    RedTeamFinding,
)
from crucible.cache_routing import TtlMode


# ===========================================================================
# Charter merge
# ===========================================================================

def compute_consensus_level(raised_by_count: int, panel_size: int) -> ConsensusLevel:
    if raised_by_count == panel_size:
        return ConsensusLevel.UNANIMOUS
    if raised_by_count > panel_size / 2:
        return ConsensusLevel.MAJORITY
    return ConsensusLevel.MINORITY


def _slugify(text: str) -> str:
    """Lowercase, strip, collapse whitespace to underscore."""
    return re.sub(r"\s+", "_", text.strip().lower())


def merge_mpsr_panel_responses(
    panel_responses: list[MpsrPanelOutput],
    panel_size: int,
    jurisdiction: str,
) -> tuple[list[CharterEntry], list[dict[str, Any]]]:
    """Merge MPSR panel responses into Charter entries. All six categories merged."""
    entries_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    def add(category: str, canonical: str, payload: dict) -> None:
        key = (category, canonical)
        entry = entries_by_key.setdefault(key, {
            "type": category,
            "canonical_reference": canonical,
            "payload": payload,
            "raised_by": 0,
        })
        entry["raised_by"] += 1

    for response in panel_responses:
        for statute in response.applicable_statutes:
            canonical = normalize_citation(
                f"{statute.name}{statute.article}", jurisdiction
            )
            add("statute", canonical, statute.model_dump())

        for theme in response.applicable_doctrine_themes:
            canonical = _slugify(theme.topic)
            add("doctrine_theme", canonical, theme.model_dump())

        for mc in response.mandatory_clauses:
            canonical = _slugify(mc.clause)
            add("mandatory_clause", canonical, mc.model_dump())

        for domain in response.risk_domains:
            canonical = _slugify(domain.domain)
            add("risk_domain", canonical, domain.model_dump())

        for trap in response.jurisdictional_traps:
            canonical = _slugify(trap.trap)
            add("jurisdictional_trap", canonical, trap.model_dump())

        for seed in response.rag_query_seeds:
            normalized_kw = sorted(
                k.strip().lower() for k in seed.keywords if k.strip()
            )
            canonical = f"{seed.layer}:" + "_".join(normalized_kw)
            add("rag_seed", canonical, seed.model_dump())

    charter_entries = [
        CharterEntry(
            type=v["type"],
            canonical_reference=v["canonical_reference"],
            payload=v["payload"],
            consensus_level=compute_consensus_level(v["raised_by"], panel_size),
            raised_by_count=v["raised_by"],
            panel_size=panel_size,
        )
        for v in entries_by_key.values()
    ]

    disagreement_log = [
        {"canonical_reference": e.canonical_reference, "type": e.type,
         "raised_by_count": e.raised_by_count, "consensus_level": e.consensus_level.value}
        for e in charter_entries
        if e.consensus_level != ConsensusLevel.UNANIMOUS
    ]

    return charter_entries, disagreement_log


def check_panel_degradation(
    panel_responses: "list[MpsrPanelOutput] | MpsrPanelDispatchResult",
    target_panel_size: int,
) -> bool:
    """Returns True if panel has fewer responses than target_panel_size."""
    if isinstance(panel_responses, MpsrPanelDispatchResult):
        return panel_responses.effective_panel_size < target_panel_size
    return len(panel_responses) < target_panel_size


# ===========================================================================
# Concurrency-safe dispatch
# ===========================================================================

_concurrency_logger = logging.getLogger("crucible.concurrency")


def _get_provider_name(provider: Any) -> str:
    if isinstance(provider, str):
        return provider
    if hasattr(provider, "name"):
        return provider.name
    if hasattr(provider, "provider"):
        return str(provider.provider)
    return str(provider)


def _is_retries_exhausted(exc: Exception) -> bool:
    if hasattr(exc, "retries_exhausted"):
        return bool(exc.retries_exhausted)
    exc_name = type(exc).__name__.lower()
    return "retry" in exc_name or "retries" in exc_name


def dispatch_mpsr_panel(
    providers: list[Any],
    dispatch_fn: Callable[[Any], Any],
    timeout_seconds: float = 120.0,
    executor: ThreadPoolExecutor | None = None,
) -> MpsrPanelDispatchResult:
    """Dispatch MPSR panel calls concurrently. Guarantees future.result() on every future."""
    if not providers:
        return MpsrPanelDispatchResult(results=[], total_elapsed_seconds=0.0)

    own_executor = executor is None
    if own_executor:
        executor = ThreadPoolExecutor(
            max_workers=len(providers),
            thread_name_prefix="mpsr_panel",
        )

    dispatch_start = time.monotonic()
    future_to_provider: dict[Future, Any] = {}
    provider_start_times: dict[str, float] = {}
    results: list[MpsrProviderResult] = []

    try:
        for provider in providers:
            provider_name = _get_provider_name(provider)
            provider_start_times[provider_name] = time.monotonic()
            future = executor.submit(dispatch_fn, provider)
            future_to_provider[future] = provider

        completed_futures: set[Future] = set()

        try:
            for future in as_completed(
                future_to_provider.keys(),
                timeout=timeout_seconds,
            ):
                completed_futures.add(future)
                provider = future_to_provider[future]
                provider_name = _get_provider_name(provider)
                elapsed = time.monotonic() - provider_start_times[provider_name]

                try:
                    response = future.result()
                    results.append(MpsrProviderResult(
                        provider=provider_name,
                        success=True,
                        response=response,
                        elapsed_seconds=elapsed,
                    ))
                    _concurrency_logger.info(
                        "MPSR provider %s succeeded in %.1fs",
                        provider_name, elapsed,
                    )
                except Exception as exc:
                    results.append(MpsrProviderResult(
                        provider=provider_name,
                        success=False,
                        error=exc,
                        elapsed_seconds=elapsed,
                        retries_exhausted=_is_retries_exhausted(exc),
                    ))
                    _concurrency_logger.error(
                        "MPSR provider %s failed after %.1fs: %s: %s",
                        provider_name, elapsed,
                        type(exc).__name__, str(exc),
                    )

        except TimeoutError:
            for future, provider in future_to_provider.items():
                if future not in completed_futures:
                    provider_name = _get_provider_name(provider)
                    elapsed = time.monotonic() - provider_start_times[provider_name]
                    results.append(MpsrProviderResult(
                        provider=provider_name,
                        success=False,
                        error=TimeoutError(
                            f"Provider {provider_name} did not respond within "
                            f"{timeout_seconds}s timeout"
                        ),
                        elapsed_seconds=elapsed,
                        retries_exhausted=False,
                    ))
                    _concurrency_logger.error(
                        "MPSR provider %s timed out after %.1fs",
                        provider_name, elapsed,
                    )
                    future.cancel()

    finally:
        if own_executor:
            _shutdown_executor_safely(executor, wait=False, cancel_futures=True)

    total_elapsed = time.monotonic() - dispatch_start

    return MpsrPanelDispatchResult(
        results=results,
        total_elapsed_seconds=total_elapsed,
    )


def format_panel_failure_summary(dispatch_result: MpsrPanelDispatchResult) -> str:
    """Format a human-readable summary of panel failures for the PANEL_DEGRADED checkpoint."""
    total = dispatch_result.target_panel_size
    ok = dispatch_result.effective_panel_size
    failed = dispatch_result.failed

    lines = [
        f"MPSR panel degraded: {ok}/{total} providers responded successfully.",
        "",
    ]

    if not failed:
        lines.append("No failures recorded (this should not happen — investigate).")
        return "\n".join(lines)

    lines.append("Failed providers:")
    lines.append("")

    for i, f in enumerate(failed, 1):
        error_type = type(f.error).__name__ if f.error else "Unknown"
        error_msg = str(f.error)[:200] if f.error else "No error message"
        retry_status = "retries exhausted" if f.retries_exhausted else "not retried / retry available"

        lines.append(f"  {i}. {f.provider}")
        lines.append(f"     Error: {error_type}: {error_msg}")
        lines.append(f"     Elapsed: {f.elapsed_seconds:.1f}s")
        lines.append(f"     Retry status: {retry_status}")
        lines.append("")

    if ok == 0:
        lines.append(
            "\u26a0 ALL providers failed. Proceeding would produce an empty Charter. "
            "RETRY or ABORT recommended."
        )
    elif ok == 1:
        lines.append(
            "\u26a0 Only 1 provider responded. MPSR consensus is meaningless with a "
            "single provider — all entries will be MINORITY consensus. "
            "RETRY recommended."
        )
    elif ok == 2:
        lines.append(
            "2/3 panel: MAJORITY consensus is unreachable (mathematically "
            "requires >50% of 2, i.e., 2/2 = UNANIMOUS). Consensus space "
            "collapses from 3 buckets to 2 (UNANIMOUS / MINORITY only)."
        )

    return "\n".join(lines)


def merge_dispatch_results(
    original: MpsrPanelDispatchResult,
    retry: MpsrPanelDispatchResult,
) -> MpsrPanelDispatchResult:
    """Merge retry dispatch results into the original."""
    retried_providers = {r.provider for r in retry.results}

    merged_results = []
    for r in original.results:
        if r.provider in retried_providers:
            continue
        merged_results.append(r)

    merged_results.extend(retry.results)

    return MpsrPanelDispatchResult(
        results=merged_results,
        total_elapsed_seconds=original.total_elapsed_seconds + retry.total_elapsed_seconds,
    )


# ===========================================================================
# Thread-local TTL mode (Issue 9 minimal fix for MPSR dispatch)
# ===========================================================================

_ttl_mode_local = threading.local()


def get_ttl_mode_threadsafe(run_state: Any) -> TtlMode:
    """Get the effective TTL mode for the current thread."""
    local_mode = getattr(_ttl_mode_local, "mode", None)
    if local_mode is not None:
        return local_mode
    return run_state.ttl_mode


def set_ttl_degraded_threadsafe(run_state: Any) -> None:
    """Set TTL mode to degraded in both thread-local storage and the shared RunState."""
    _ttl_mode_local.mode = TtlMode(mode="5m", _degraded=True)
    run_state.ttl_mode.degrade_to_5m()
    _concurrency_logger.warning(
        "TTL mode degraded to 5min (set from thread %s)",
        threading.current_thread().name,
    )


def reset_thread_local_ttl() -> None:
    """Clear thread-local TTL override. Call at the start of each dispatch."""
    if hasattr(_ttl_mode_local, "mode"):
        del _ttl_mode_local.mode


# ===========================================================================
# Executor management
# ===========================================================================

def _shutdown_executor_safely(
    executor: ThreadPoolExecutor,
    wait: bool = True,
    cancel_futures: bool = False,
) -> None:
    """Shut down a ThreadPoolExecutor with safe defaults."""
    try:
        executor.shutdown(wait=wait, cancel_futures=cancel_futures)
    except TypeError:
        _concurrency_logger.warning(
            "ThreadPoolExecutor.shutdown(cancel_futures=True) not supported "
            "(Python < 3.9). Falling back to shutdown(wait=%s).", wait,
        )
        executor.shutdown(wait=wait)


def create_managed_executor(
    max_workers: int = 3,
    thread_name_prefix: str = "mpsr",
) -> ThreadPoolExecutor:
    """Create a ThreadPoolExecutor with canonical MPSR dispatch settings."""
    return ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix=thread_name_prefix,
    )


# ===========================================================================
# Statute delta + auditor phase 2
# ===========================================================================

def compute_statute_deltas(
    red_team_findings: list[RedTeamFinding],
    red_team_narrative: str,
    charter_entries: list[CharterEntry],
    jurisdiction: str,
    citation_extractor: Callable[[str, str], list[str]],
) -> list[dict[str, str]]:
    charter_statute_canonicals = {
        e.canonical_reference for e in charter_entries if e.type == "statute"
    }
    delta: list[dict[str, str]] = []
    seen: set[str] = set()

    for finding in red_team_findings:
        if not finding.cited_authority:
            continue
        canonical = normalize_citation(finding.cited_authority, jurisdiction)
        if canonical not in charter_statute_canonicals and canonical not in seen:
            delta.append({
                "citation": finding.cited_authority,
                "canonical": canonical,
                "source_finding_ref": finding.clause_ref,
            })
            seen.add(canonical)

    for cite in citation_extractor(red_team_narrative, jurisdiction):
        canonical = normalize_citation(cite, jurisdiction)
        if canonical not in charter_statute_canonicals and canonical not in seen:
            delta.append({
                "citation": cite, "canonical": canonical,
                "source_finding_ref": "narrative",
            })
            seen.add(canonical)

    return delta


OMISSION_TAG_PATTERN = re.compile(
    r"\[OMISSION:(?P<ref>.*?)\](?P<reason>.*?)\[/OMISSION\]",
    flags=re.DOTALL,
)


def extract_omission_tags(
    verdict_text: str,
    jurisdiction: str,
) -> dict[str, str]:
    """Parse [OMISSION:<ref>]...[/OMISSION] tags from verdict text."""
    result: dict[str, str] = {}
    for match in OMISSION_TAG_PATTERN.finditer(verdict_text):
        raw_ref = match.group("ref").strip()
        reason = match.group("reason").strip()
        canonical = normalize_citation(raw_ref, jurisdiction)
        result[canonical] = reason
    return result


def auditor_phase2(
    verdict_text: str,
    charter_entries: list[CharterEntry],
    jurisdiction: str,
    citation_extractor: Callable[[str, str], list[str]],
) -> dict[str, list[dict[str, Any]]]:
    """Cross-check Charter statutes against the Verdict."""
    omission_tags = extract_omission_tags(verdict_text, jurisdiction)

    stripped_text = OMISSION_TAG_PATTERN.sub("", verdict_text)
    verdict_canonicals = {
        normalize_citation(c, jurisdiction)
        for c in citation_extractor(stripped_text, jurisdiction)
    }

    covered, with_explanation, omitted = [], [], []

    for entry in charter_entries:
        if entry.type != "statute":
            continue

        if entry.canonical_reference in verdict_canonicals:
            covered.append({
                "canonical": entry.canonical_reference,
                "consensus_level": entry.consensus_level.value,
            })
            continue

        if entry.canonical_reference in omission_tags:
            with_explanation.append({
                "canonical": entry.canonical_reference,
                "consensus_level": entry.consensus_level.value,
                "charter_rationale": entry.payload.get("why", ""),
                "verdict_omission_reason": omission_tags[entry.canonical_reference],
            })
            continue

        omitted.append({
            "canonical": entry.canonical_reference,
            "consensus_level": entry.consensus_level.value,
            "charter_rationale": entry.payload.get("why", ""),
        })

    return {
        "covered": covered,
        "omitted_with_explanation": with_explanation,
        "omitted_from_verdict": omitted,
    }


# ===========================================================================
# Retry classification + cross-provider fallback
# ===========================================================================

def classify_error(exc: Exception) -> RetryClass:
    msg = str(exc).lower()
    if any(token in msg for token in ("429", "rate limit", "too many requests")):
        return RetryClass.RATE_LIMIT_RETRY
    if any(token in msg for token in (
        "401", "403", "unauthorized", "forbidden", "invalid api key",
    )):
        return RetryClass.HALT_AUTH
    if any(token in msg for token in ("content_policy", "policy violation", "refusal", "blocked")):
        return RetryClass.HALT_POLICY
    return RetryClass.RETRY


def cross_provider_fallback_chain(subagent_name: str, original_provider: str) -> list[str]:
    """Cross-provider fallback for content policy refusals. Same prompt, different provider."""
    if subagent_name == "red_team" and original_provider == "openai":
        return ["anthropic"]
    return []


# ===========================================================================
# Markdown fence stripping
# ===========================================================================

_FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL | re.IGNORECASE)


def strip_markdown_fences_for_json(text: str) -> str:
    text = text.strip()
    fence_match = _FENCE_PATTERN.match(text)
    if fence_match:
        return fence_match.group(1).strip()
    text = re.sub(r"^[^{\[]*", "", text, flags=re.DOTALL)
    text = re.sub(r"[^}\]]*$", "", text, flags=re.DOTALL)
    return text.strip()


# ===========================================================================
# Claude Agent SDK tool registration + pause_for_human stub
# ===========================================================================

def register_tools(agent_builder) -> None:
    """Adapt to actual SDK surface area at setup time."""
    agent_builder.add_tool(
        name="pause_for_human",
        description="Suspend run and surface decision card. Agent loop MUST NOT proceed until this returns.",
        input_schema=PauseForHumanInput.model_json_schema(),
        handler=pause_for_human,
    )
    agent_builder.add_tool(
        name="budget_check",
        description="Pessimistic budget check called BEFORE every subagent dispatch.",
        input_schema=BudgetCheckInput.model_json_schema(),
        handler=NotImplemented,
    )


def pause_for_human(input: PauseForHumanInput) -> PauseForHumanResult:
    raise NotImplementedError(
        "Wired to host UI. Agent loop MUST suspend until this returns. No timeout."
    )


# ===========================================================================
# Budget check + pricing
# ===========================================================================

PRICING_USD_PER_MTOK: dict[tuple[str, str], dict[str, float]] = {
    ("anthropic", "flagship"): {"input": -1, "output": -1},
    ("anthropic", "fast"):     {"input": -1, "output": -1},
    ("openai",    "flagship"): {"input": -1, "output": -1},
    ("google",    "flagship"): {"input": -1, "output": -1},
}


def estimate_call_cost_usd_pessimistic(
    provider: str,
    tier: ModelTier,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Pessimistic cost estimation. Assumes ZERO cache hit."""
    pricing = PRICING_USD_PER_MTOK.get((provider, tier.value))
    if pricing is None or pricing["input"] < 0 or pricing["output"] < 0:
        raise RuntimeError(
            f"Pricing for ({provider!r}, {tier.value!r}) unset. Configure "
            f"PRICING_USD_PER_MTOK in setup before running any pipeline."
        )
    return (
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"]
    )


def budget_check(input: BudgetCheckInput, run_state: Any) -> BudgetCheckResult:
    """Pessimistic budget check. Cache discount NOT subtracted from projection."""
    estimated = estimate_call_cost_usd_pessimistic(
        input.provider, input.tier,
        input.estimated_input_tokens, input.estimated_output_tokens,
    )
    safety_margin = run_state.config.safety_margin
    projected = run_state.spend_usd + (estimated * safety_margin)
    cap = run_state.config.budget_cap_usd
    allowed = projected <= cap
    return BudgetCheckResult(
        allowed=allowed,
        projected_total_usd=projected,
        cap_usd=cap,
        remaining_usd=max(cap - run_state.spend_usd, 0.0),
        safety_margin=safety_margin,
        reason=None if allowed else (
            f"Projected spend ${projected:.2f} would exceed cap ${cap:.2f}. "
            f"Halting before dispatching {input.next_subagent}."
        ),
    )
