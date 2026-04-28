"""CRUCIBLE v3.0 — public compatibility re-export layer.

All imports from this module are re-exported from the modular crucible/ package.
Existing tests import from `tools` directly; this file maintains that API.
"""

# ---- enums ----
from crucible.enums import (
    Severity, ConsensusLevel, ClauseStatus, AuditClassification,
    Decision, CheckpointId, ModelTier, RetryClass,
    _TokenType,
)

# ---- schemas ----
from crucible.schemas import (
    MpsrStatute, MpsrDoctrineTheme, MpsrMandatoryClause, MpsrRiskDomain,
    MpsrJurisdictionalTrap, MpsrRagSeed, MpsrPanelOutput,
    FrameworkClause, FrameworkMissingClauseProposal, FrameworkOutput,
    RedTeamFinding, RedTeamOutput,
    CharterDeltaConceptualEntry, CharterDeltaOutput,
    IntakeConflict, IntakeOutput, AuditCitationResult,
    CharterEntry, CachedPrefix, RebuildDecision,
    RateLimitConfig, RateLimitBucket,
    RagQueryInput, RagQueryResult,
    BrowserFetchInput, BrowserFetchResult,
    CiteExtractInput, CiteExtractResult,
    LoadTemplateInput, LoadTemplateResult,
    WriteArtifactInput, WriteArtifactResult,
    PauseForHumanInput, PauseForHumanResult,
    BudgetCheckInput, BudgetCheckResult,
    TelemetrySnapshot,
    MpsrProviderResult, MpsrPanelDispatchResult,
    ArticleRef, Document, ExpansionResult,
)

# ---- normalization ----
from crucible.normalization import normalize_citation, NORMALIZATION_RULES, dedupe_citations

# ---- mpsr ----
from crucible.mpsr import (
    compute_consensus_level, merge_mpsr_panel_responses, check_panel_degradation,
    dispatch_mpsr_panel, format_panel_failure_summary, merge_dispatch_results,
    get_ttl_mode_threadsafe, set_ttl_degraded_threadsafe, reset_thread_local_ttl,
    create_managed_executor, classify_error, cross_provider_fallback_chain,
    strip_markdown_fences_for_json, register_tools, pause_for_human, budget_check,
    estimate_call_cost_usd_pessimistic, compute_statute_deltas,
    extract_omission_tags, auditor_phase2,
    PRICING_USD_PER_MTOK, OMISSION_TAG_PATTERN,
    _slugify, _ttl_mode_local,
)

# ---- semantic_dedup ----
from crucible.semantic_dedup import (
    semantic_dedup_charter_entries,
    _SEMANTIC_DEDUP_CATEGORIES, _cosine_similarity, _cluster_by_similarity,
)

# ---- cache_routing ----
from crucible.cache_routing import (
    TtlMode, call_anthropic_with_ttl_fallback,
    probe_cache_freshness, estimate_rebuild_cost_usd,
    estimate_remaining_pipeline_cost, make_rebuild_decision,
    REMAINING_STEP_ESTIMATES, PIPELINE_TRACK_A, PIPELINE_TRACK_B,
    _is_bad_request_400, _request_contains_cache_control,
)
# on_resume_from_checkpoint is defined here (not re-exported from cache_routing)
# so that test patches of tools.probe_cache_freshness and tools.make_rebuild_decision apply.
def on_resume_from_checkpoint(
    run_state,
    active_prefixes,
    probe_executor_by_provider,
    rebuild_executor_by_provider,
    pause_handler,
    track,
    current_position,
):
    import sys
    _tools = sys.modules[__name__]
    _probe = getattr(_tools, "probe_cache_freshness")
    _make = getattr(_tools, "make_rebuild_decision")

    decisions = []
    for prefix in active_prefixes:
        probe_exec = probe_executor_by_provider.get(prefix.provider)
        if probe_exec is None:
            continue
        is_fresh = _probe(prefix, probe_exec)
        pricing_key = (prefix.provider, prefix.tier.value)
        pricing = run_state.pricing_table.get(pricing_key)
        if pricing is None:
            continue
        rebuild_cost = estimate_rebuild_cost_usd(
            prefix, pricing,
            write_premium=2.0 if prefix.ttl_mode == "1h" else 1.25,
        )
        decision = _make(
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

# ---- runtime ----
from crucible.runtime import (
    RagCache, RunConfig, RunState,
)

# ---- persistence ----
from crucible.persistence import (
    restore_from_checkpoint, run_pipeline_with_persistence,
    _PERSIST_FIELDS,
)

# ---- rag_expander ----
from crucible.rag_expander import (
    parse_canonical_ref, extract_referenced_articles,
    expand_references_bfs, expand_document_dict,
    _tokenize_japan, _format_expansion_japan,
)
