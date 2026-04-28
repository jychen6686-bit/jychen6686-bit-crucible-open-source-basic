from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from crucible.enums import CheckpointId, Decision
from crucible.mpsr import (
    auditor_phase2,
    budget_check,
    check_panel_degradation,
    compute_consensus_level,
    compute_statute_deltas,
    dispatch_mpsr_panel,
    format_panel_failure_summary,
    merge_mpsr_panel_responses,
    strip_markdown_fences_for_json,
)
from crucible.adapters.egov_adapter import fetch_egov_article, resolve_unknown_statutes_batch
from crucible.normalization import dedupe_citations
from crucible.prompts import PromptBuilder
from crucible.rag_expander import parse_canonical_ref
from crucible.runtime import RunState
from crucible.schemas import (
    AuditCitationResult,
    BudgetCheckInput,
    CharterDeltaOutput,
    CharterEntry,
    FrameworkOutput,
    IntakeOutput,
    PauseForHumanInput,
    PauseForHumanResult,
    RedTeamFinding,
    RedTeamOutput,
)
from crucible.semantic_dedup import semantic_dedup_charter_entries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JP_CITE_RE = re.compile(r"第\d+条(?:第\d+項)?(?:第\d+号)?")
_US_CITE_RE = re.compile(r"(?:§|\bSection\b|\bSec\.\b)\s*\d+[\w.-]*")
_EU_CITE_RE = re.compile(r"\bArticle\s+\d+\b")


def _make_citation_extractor(jurisdiction: str) -> Callable[[str, str], list[str]]:
    pattern = _JP_CITE_RE if jurisdiction.lower() == "japan" else (
        _US_CITE_RE if jurisdiction.lower() in ("us", "usa") else _EU_CITE_RE
    )
    return lambda text, _jur: pattern.findall(text)


def _accrue_cost(run_state: RunState, adapter: Any,
                 provider: str, tier: str) -> None:
    input_tok = getattr(adapter, "last_input_tokens", 0)
    output_tok = getattr(adapter, "last_output_tokens", 0)
    pricing = run_state.pricing_table.get((provider, tier))
    if pricing and (input_tok or output_tok):
        cost = (input_tok * pricing["input"] + output_tok * pricing["output"]) / 1_000_000
        run_state.spend_usd += cost


def _charter_json(run_state: RunState) -> str:
    return json.dumps(run_state.artifacts.get("charter", []),
                      ensure_ascii=False, indent=2)


def _reconstruct_charter_entries(run_state: RunState) -> list[CharterEntry]:
    return [CharterEntry(**d) for d in run_state.artifacts.get("charter", [])]


def _reconstruct_rt_findings(run_state: RunState) -> list[RedTeamFinding]:
    rt = run_state.artifacts.get("red_team", {})
    findings_raw = rt.get("findings", []) if isinstance(rt, dict) else []
    return [RedTeamFinding(**d) for d in findings_raw]


def _format_engagement_context(assumptions: Any) -> str:
    lines = [
        f"- Party role: {assumptions.party_role}",
        f"- Contract type: {assumptions.contract_type}",
    ]
    if assumptions.business_context:
        lines.append(f"- Business context: {assumptions.business_context}")
    if assumptions.counterparty_type:
        lines.append(f"- Counterparty: {assumptions.counterparty_type}")
    if assumptions.cross_border:
        lines.append("- Cross-border: yes")
    if assumptions.personal_data:
        lines.append("- Personal data involved: yes")
    if assumptions.personnel_involved:
        lines.append("- Personnel involved: yes")
    for k, v in (assumptions.additional or {}).items():
        lines.append(f"- {k}: {v}")
    body = "\n".join(lines)
    return (
        "---\n"
        "ENGAGEMENT CONTEXT (practitioner-supplied deal facts — inform analysis "
        "but do not treat as legal requirements):\n"
        f"{body}\n"
        "---"
    )


# ---------------------------------------------------------------------------
# Charter consolidation (Phase 3)
# ---------------------------------------------------------------------------

def _consolidate_charter(
    entries: list[CharterEntry],
    consolidator: Any,
    run_state: RunState,
) -> list[CharterEntry]:
    """Merge semantically duplicate Charter entries without losing unique coverage."""
    entries_json = json.dumps(
        [e.model_dump(mode="json") for e in entries], ensure_ascii=False, indent=2
    )
    system_prompt = (
        "You are a legal scope deduplication specialist.\n\n"
        "You receive a Charter JSON array. Entries may contain semantic duplicates \u2014 "
        "items raised by different MPSR panel members that describe the same legal concept "
        "in different words.\n\n"
        "RULES:\n"
        "1. MERGE entries that are semantically identical (same legal concept, different wording). "
        "When merging, SUM their raised_by_count values and keep the higher panel_size.\n"
        "2. PRESERVE all entries that represent genuinely distinct legal concepts. "
        "When in doubt, DO NOT merge \u2014 false preservation is safe, false merging destroys coverage.\n"
        "3. Output valid JSON array using the exact same schema as the input. "
        "Every output entry must have all original fields.\n"
        "4. Do NOT add new entries. Only merge or pass through existing ones.\n"
        "5. Respond ONLY with the JSON array, no other text."
    )
    user_prompt = (
        f"Consolidate this Charter ({len(entries)} entries):\n\n{entries_json}"
    )
    try:
        text = consolidator.call_text(system_prompt, user_prompt, max_tokens=8192)
        cleaned = strip_markdown_fences_for_json(text)
        raw_list = json.loads(cleaned)
        consolidated = [CharterEntry(**d) for d in raw_list]
        _accrue_cost(run_state, consolidator, "anthropic", "mid")
        return consolidated
    except Exception:
        # Consolidation failed \u2014 return originals unchanged
        return entries


def _default_pause_handler(checkpoint_dir: Path) -> Any:
    from crucible.adapters.managed_agent_ui import pause_for_human_managed_agent
    def _handler(inp: PauseForHumanInput) -> PauseForHumanResult:
        return pause_for_human_managed_agent(inp, None, checkpoint_dir)
    return _handler


# ---------------------------------------------------------------------------
# RAG phase (treatise context enrichment)
# ---------------------------------------------------------------------------

def _run_rag_phase(
    charter_entries: list[CharterEntry],
    adapters: dict[str, Any],
) -> tuple[str, str]:
    """Query the RAG adapter with charter rag_query_seeds.

    Returns (statutory_treatise_ctx, caselaw_ctx) as formatted strings.
    Both are empty strings if no rag_adapter is present in adapters or if
    the adapter returns no results.  Always safe to call; never raises.

    Two separate queries are issued:
    - Treatise query: uses rag_seed entries with layer in ("treatise", "statutory")
    - Case law query: uses rag_seed entries with layer="case" plus doctrine_theme
      topics from the charter (doctrine terminology is well-suited for embedding
      retrieval against case law volumes indexed under factual/principle terms)
    """
    rag_adapter = adapters.get("rag_adapter")
    if rag_adapter is None:
        return "", ""

    # Extract keywords by layer from rag_seed charter entries
    treatise_kw: list[str] = []
    case_kw: list[str] = []
    for entry in charter_entries:
        if entry.type == "rag_seed":
            layer = entry.payload.get("layer", "")
            kws = entry.payload.get("keywords", [])
            if layer in ("treatise", "statutory"):
                treatise_kw.extend(kws)
            elif layer == "case":
                case_kw.extend(kws)
        elif entry.type == "doctrine_theme":
            topic = entry.payload.get("topic", "")
            if topic:
                case_kw.append(topic)

    if not treatise_kw and not case_kw:
        return "", ""

    def _dedup(kws: list[str], cap: int) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for kw in kws:
            kw = kw.strip()
            if kw and kw not in seen:
                seen.add(kw)
                out.append(kw)
                if len(out) >= cap:
                    break
        return out

    def _fmt_docs(doc_list: list) -> str:
        if not doc_list:
            return ""
        parts = []
        for d in doc_list:
            heading = d.expansion_metadata.get("heading_path", "")
            ref = f"{d.statute} — {heading}" if d.statute and heading else (d.statute or d.canonical_ref)
            parts.append(f"【{ref}】\n{d.text}")
        return "\n\n---\n\n".join(parts)

    # --- Treatise query ---
    treatise_ctx = ""
    treatise_kw_deduped = _dedup(treatise_kw, 20)
    if treatise_kw_deduped:
        try:
            docs = rag_adapter.query_treatises(treatise_kw_deduped, max_results=12)
            treatise_docs = [
                d for d in docs
                if "case" not in d.expansion_metadata.get("legal_authority", "")
            ]
            treatise_ctx = _fmt_docs(treatise_docs)
        except Exception:
            pass

    # --- Case law query (second, separate embedding call) ---
    caselaw_ctx = ""
    case_kw_deduped = _dedup(case_kw, 10)
    if case_kw_deduped:
        try:
            docs = rag_adapter.query_treatises(case_kw_deduped, max_results=6)
            caselaw_docs = [
                d for d in docs
                if "判例" in (d.statute or "")
                or "case" in d.expansion_metadata.get("legal_authority", "")
            ]
            caselaw_ctx = _fmt_docs(caselaw_docs)
        except Exception:
            pass

    return treatise_ctx, caselaw_ctx


# ---------------------------------------------------------------------------
# MPSR phase (shared by Track A and B)
# ---------------------------------------------------------------------------

def _run_mpsr_phase(
    run_state: RunState,
    context: dict,
    adapters: dict[str, Any],
    checkpoint_dir: Path,
    pause_handler: Any,
) -> list[CharterEntry]:
    jurisdiction = run_state.config.jurisdiction
    panel_providers = context.get("mpsr_providers", [])
    dispatch_fn = adapters.get("mpsr_dispatch_fn", lambda p: p(p))

    if not panel_providers:
        return _reconstruct_charter_entries(run_state)

    # Skip MPSR if charter already computed (resume path)
    if run_state.artifacts.get("charter"):
        return _reconstruct_charter_entries(run_state)

    dispatch_result = dispatch_mpsr_panel(
        providers=panel_providers,
        dispatch_fn=dispatch_fn,
        timeout_seconds=120.0,
    )

    if check_panel_degradation(dispatch_result, target_panel_size=3):
        run_state.persist(checkpoint_dir)
        result = pause_handler(PauseForHumanInput(
            checkpoint_id=CheckpointId.PANEL_DEGRADED,
            summary=format_panel_failure_summary(dispatch_result),
            artifact_refs=[],
            decision_options=[Decision.RETRY, Decision.PROCEED_WITH_DEGRADED_PANEL, Decision.ABORT],
            edits_allowed_on=[],
            telemetry=run_state.snapshot(),
        ))
        if result.decision == Decision.ABORT:
            raise SystemExit("Aborted at PANEL_DEGRADED checkpoint.")

    panel_size = dispatch_result.effective_panel_size or 1
    charter_entries, _ = merge_mpsr_panel_responses(
        dispatch_result.succeeded, panel_size, jurisdiction
    )
    charter_entries, _ = semantic_dedup_charter_entries(
        charter_entries, panel_size, embed_fn=None
    )

    # Optional consolidation pass for large charters
    consolidator = adapters.get("consolidator")
    if consolidator is not None and len(charter_entries) > 80:
        original_count = len(charter_entries)
        charter_entries = _consolidate_charter(charter_entries, consolidator, run_state)
        # Recompute consensus levels after merge
        for entry in charter_entries:
            entry.consensus_level = compute_consensus_level(entry.raised_by_count, panel_size)
        print(f"   Charter consolidated: {original_count} \u2192 {len(charter_entries)} entries")

    run_state.record_artifact("charter", [e.model_dump(mode="json") for e in charter_entries])
    run_state.persist(checkpoint_dir)
    return charter_entries


# ---------------------------------------------------------------------------
# Track A — Review pipeline
# ---------------------------------------------------------------------------

def run_track_a(
    run_state: RunState,
    contract_text: str,
    context: dict,
    adapters: dict[str, Any],
    checkpoint_dir: Path,
    pause_handler: Any = None,
) -> Path:
    """Execute Track A: MPSR→Charter→Framework→RedTeam→CharterDelta→Verdict→Audit."""
    checkpoint_dir = Path(checkpoint_dir)
    run_state.track = "A"

    if pause_handler is None:
        pause_handler = _default_pause_handler(checkpoint_dir)

    jurisdiction = run_state.config.jurisdiction
    jurisdiction_language = run_state.config.jurisdiction_language
    report_language = context.get("report_language", "English")
    contract_type = context.get("contract_type", "Commercial Agreement")
    assumptions = run_state.config.assumptions
    engagement_ctx = _format_engagement_context(assumptions)

    cite_extractor = _make_citation_extractor(jurisdiction)

    # --- Phase 1: MPSR ---
    _run_mpsr_phase(run_state, context, adapters, checkpoint_dir, pause_handler)

    # SETUP_CONFIRM checkpoint
    run_state.persist(checkpoint_dir)
    result = pause_handler(PauseForHumanInput(
        checkpoint_id=CheckpointId.SETUP_CONFIRM,
        summary=(
            f"MPSR panel complete. Charter assembled. "
            f"Role: {assumptions.party_role} | Type: {assumptions.contract_type} | "
            f"Jurisdiction: {jurisdiction}. Proceed with framework analysis?"
        ),
        artifact_refs=["charter"],
        decision_options=[Decision.APPROVE, Decision.ABORT],
        edits_allowed_on=[],
        telemetry=run_state.snapshot(),
    ))
    if result.decision == Decision.ABORT:
        raise SystemExit("Aborted at SETUP_CONFIRM.")

    cap_msg = run_state.check_loop_and_caps()
    if cap_msg:
        raise RuntimeError(f"Cap reached: {cap_msg}")

    # --- RAG phase: enrich context with treatise passages ---
    rag_statutory_ctx, rag_caselaw_ctx = _run_rag_phase(
        _reconstruct_charter_entries(run_state), adapters
    )

    charter_json_str = _charter_json(run_state)

    # --- Phase 2: Framework ---
    anth_mid = adapters.get("anthropic_mid", adapters.get("anthropic_flagship"))
    fw_system = PromptBuilder.framework_system(
        jurisdiction, jurisdiction_language,
        contract_type=contract_type, mode_verb="reviewed",
        charter_json=charter_json_str,
    )
    fw_user = PromptBuilder.framework_user(
        mode="review", contract_text=contract_text,
        rag_statutory_context=rag_statutory_ctx,
    ) + "\n\n" + engagement_ctx
    fw_result: FrameworkOutput = anth_mid.call_structured_su(
        FrameworkOutput, fw_system, fw_user, max_tokens=8192
    )
    _accrue_cost(run_state, anth_mid, "anthropic", "mid")
    run_state.record_artifact("framework", fw_result.model_dump(mode="json"))
    run_state.persist(checkpoint_dir)

    # SCOPE_CONFIRM checkpoint
    result = pause_handler(PauseForHumanInput(
        checkpoint_id=CheckpointId.SCOPE_CONFIRM,
        summary="Framework analysis complete. Review framework clauses, then proceed to Red Team?",
        artifact_refs=["framework"],
        decision_options=[Decision.APPROVE, Decision.ABORT],
        edits_allowed_on=["framework"],
        telemetry=run_state.snapshot(),
    ))
    if result.decision == Decision.ABORT:
        raise SystemExit("Aborted at SCOPE_CONFIRM.")

    cap_msg = run_state.check_loop_and_caps()
    if cap_msg:
        raise RuntimeError(f"Cap reached: {cap_msg}")

    # --- Phase 3: Red Team ---
    oai_red = adapters.get("openai_red_team", adapters.get("openai_flagship"))
    rt_system = PromptBuilder.red_team_system(
        jurisdiction, jurisdiction_language,
        contract_type=contract_type, target_label="contract",
        charter_json=charter_json_str,
    )
    rt_user = PromptBuilder.red_team_user(
        mode="review", contract_text=contract_text, target_label="contract",
        rag_statutory_context=rag_statutory_ctx,
        rag_treatise_context=rag_caselaw_ctx,
        jurisdiction_language=jurisdiction_language,
    ) + "\n\n" + engagement_ctx
    try:
        rt_result: RedTeamOutput = oai_red.call_structured_su(
            RedTeamOutput, rt_system, rt_user, max_tokens=8192
        )
        _accrue_cost(run_state, oai_red, "openai", "flagship")
    except Exception:
        # Cross-provider fallback to Anthropic on content policy or failure
        rt_result = anth_mid.call_structured_su(
            RedTeamOutput, rt_system, rt_user, max_tokens=8192
        )
        _accrue_cost(run_state, anth_mid, "anthropic", "mid")

    run_state.record_artifact("red_team", rt_result.model_dump(mode="json"))
    run_state.persist(checkpoint_dir)

    # POST_RED checkpoint
    result = pause_handler(PauseForHumanInput(
        checkpoint_id=CheckpointId.POST_RED,
        summary="Red Team analysis complete. Proceeding to Charter Delta computation.",
        artifact_refs=["red_team"],
        decision_options=[Decision.APPROVE, Decision.ABORT],
        edits_allowed_on=[],
        telemetry=run_state.snapshot(),
    ))
    if result.decision == Decision.ABORT:
        raise SystemExit("Aborted at POST_RED.")

    cap_msg = run_state.check_loop_and_caps()
    if cap_msg:
        raise RuntimeError(f"Cap reached: {cap_msg}")

    # --- Phase 4: Charter Delta ---
    charter_entries = _reconstruct_charter_entries(run_state)
    rt_findings = _reconstruct_rt_findings(run_state)
    rt_narrative = run_state.artifacts.get("red_team", {}).get("narrative", "")

    statute_deltas = compute_statute_deltas(
        rt_findings, rt_narrative, charter_entries, jurisdiction, cite_extractor
    )

    anth_fast = adapters.get("anthropic_fast")
    cd_system = PromptBuilder.charter_delta_residual_system(
        jurisdiction, jurisdiction_language=jurisdiction_language,
        charter_json=charter_json_str,
        precomputed_statute_deltas_json=json.dumps(statute_deltas, ensure_ascii=False),
    )
    cd_user = PromptBuilder.charter_delta_residual_user(
        red_team_findings_json=json.dumps(
            [f.model_dump(mode="json") for f in rt_findings], ensure_ascii=False
        )
    )
    cd_result: CharterDeltaOutput = anth_fast.call_structured_su(
        CharterDeltaOutput, cd_system, cd_user, max_tokens=2048
    )
    _accrue_cost(run_state, anth_fast, "anthropic", "fast")
    run_state.record_artifact("charter_delta", cd_result.model_dump(mode="json"))
    run_state.persist(checkpoint_dir)

    if cd_result.conceptual_delta_found:
        result = pause_handler(PauseForHumanInput(
            checkpoint_id=CheckpointId.DELTA_CONFIRM,
            summary=(
                f"Charter Delta found {len(cd_result.new_conceptual_entries)} new entries. "
                "Approve to incorporate into Verdict scope."
            ),
            artifact_refs=["charter_delta"],
            decision_options=[Decision.APPROVE, Decision.ABORT],
            edits_allowed_on=["charter_delta"],
            telemetry=run_state.snapshot(),
        ))
        if result.decision == Decision.ABORT:
            raise SystemExit("Aborted at DELTA_CONFIRM.")

    cap_msg = run_state.check_loop_and_caps()
    if cap_msg:
        raise RuntimeError(f"Cap reached: {cap_msg}")

    # --- Phase 5: Verdict ---
    fw_artifact = run_state.artifacts.get("framework", {})
    google_flag = adapters.get("google_flagship")
    vd_system = PromptBuilder.verdict_system(
        jurisdiction, jurisdiction_language,
        report_language=report_language,
        charter_json=charter_json_str,
    )
    vd_user = PromptBuilder.verdict_user(
        framework_clauses_json=json.dumps(fw_artifact.get("clauses", []), ensure_ascii=False),
        framework_missing_clauses_proposed_language_json=json.dumps(
            fw_artifact.get("missing_clauses_proposed_language", []), ensure_ascii=False
        ),
        red_team_findings_json=json.dumps(
            [f.model_dump(mode="json") for f in rt_findings], ensure_ascii=False
        ),
        charter_delta_findings_json=json.dumps(
            cd_result.model_dump(mode="json"), ensure_ascii=False
        ),
        rag_treatise_context=rag_caselaw_ctx,
        contract_text=contract_text,
        report_language=report_language,
        jurisdiction_language=jurisdiction_language,
    )
    verdict_text: str = google_flag.call_verdict(vd_system, vd_user)
    _accrue_cost(run_state, google_flag, "google", "flagship")
    run_state.record_artifact("verdict", verdict_text)
    run_state.persist(checkpoint_dir)

    # POST_VERDICT checkpoint
    result = pause_handler(PauseForHumanInput(
        checkpoint_id=CheckpointId.POST_VERDICT,
        summary="Verdict complete. Proceeding to citation audit.",
        artifact_refs=["verdict"],
        decision_options=[Decision.APPROVE, Decision.ABORT],
        edits_allowed_on=[],
        telemetry=run_state.snapshot(),
    ))
    if result.decision == Decision.ABORT:
        raise SystemExit("Aborted at POST_VERDICT.")

    cap_msg = run_state.check_loop_and_caps()
    if cap_msg:
        raise RuntimeError(f"Cap reached: {cap_msg}")

    # --- Phase 6: Audit ---
    aud_system = PromptBuilder.auditor_system(
        jurisdiction, jurisdiction_language=jurisdiction_language,
        report_language=report_language,
    )
    raw_cites = [f.cited_authority for f in rt_findings if f.cited_authority]
    unique_cites, _ = dedupe_citations(raw_cites, jurisdiction)

    # Pre-audit: batch-resolve any statute names unknown to the lookup table
    _audit_statutes = [
        parse_canonical_ref(c).statute for c in unique_cites if parse_canonical_ref(c) is not None
    ]
    resolve_unknown_statutes_batch(list(set(_audit_statutes)), anth_fast)

    audit_phase1: list[dict] = []
    _aud_cap = run_state.config.auditor_phase1_dispatches_cap
    for cite in unique_cites:
        if run_state.auditor_phase1_dispatches >= _aud_cap:
            break
        # Attempt live e-Gov retrieval for this citation
        ref = parse_canonical_ref(cite)
        if ref is not None:
            egov_doc = fetch_egov_article(ref)
            retrieved_text = egov_doc.text if egov_doc is not None else "[retrieval not available]"
        else:
            retrieved_text = "[retrieval not available]"

        # Hard guard: skip LLM when no source text is available — prevents parametric verification
        if retrieved_text == "[retrieval not available]":
            audit_phase1.append({
                "citation": cite,
                "classification": "UNCERTAIN",
                "rationale": "retrieval_failed",
                "retrieved_text_quote": "",
            })
            run_state.auditor_phase1_dispatches += 1
            continue

        aud_user = PromptBuilder.auditor_user(
            citation_string=cite,
            document_claim_excerpt="",
            retrieved_source_text=retrieved_text,
        )
        aud_result: AuditCitationResult = anth_fast.call_structured_su(
            AuditCitationResult, aud_system, aud_user, max_tokens=1024
        )
        run_state.auditor_phase1_dispatches += 1
        _accrue_cost(run_state, anth_fast, "anthropic", "fast")
        audit_phase1.append({**aud_result.model_dump(mode="json"), "citation": cite})

    audit_phase2 = auditor_phase2(verdict_text, charter_entries, jurisdiction, cite_extractor)

    run_state.record_artifact("audit", {"phase1": audit_phase1, "phase2": audit_phase2})
    run_state.persist(checkpoint_dir)

    # POST_AUDIT checkpoint
    omitted = audit_phase2.get("omitted_from_verdict", [])
    result = pause_handler(PauseForHumanInput(
        checkpoint_id=CheckpointId.POST_AUDIT,
        summary=(
            f"Audit complete. Phase 1: {len(audit_phase1)}/{len(unique_cites)} citations checked"
            + (f" (capped at {_aud_cap})" if len(audit_phase1) < len(unique_cites) else "")
            + f". Phase 2: {len(omitted)} charter statutes omitted from verdict without explanation."
        ),
        artifact_refs=["audit", "verdict"],
        decision_options=[Decision.APPROVE, Decision.ABORT],
        edits_allowed_on=[],
        telemetry=run_state.snapshot(),
    ))
    if result.decision == Decision.ABORT:
        raise SystemExit("Aborted at POST_AUDIT.")

    return checkpoint_dir / "checkpoint.json"


# ---------------------------------------------------------------------------
# Track B — Draft pipeline
# ---------------------------------------------------------------------------

def run_track_b(
    run_state: RunState,
    termsheet_text: str,
    context: dict,
    adapters: dict[str, Any],
    checkpoint_dir: Path,
    pause_handler: Any = None,
) -> Path:
    """Execute Track B: MPSR→Intake→Framework→DrafterB2→RedTeam→CharterDelta→DrafterB4→Audit."""
    checkpoint_dir = Path(checkpoint_dir)
    run_state.track = "B"

    if pause_handler is None:
        pause_handler = _default_pause_handler(checkpoint_dir)

    jurisdiction = run_state.config.jurisdiction
    jurisdiction_language = run_state.config.jurisdiction_language
    report_language = context.get("report_language", "English")
    contract_type = context.get("contract_type", "Commercial Agreement")
    termsheet_format = context.get("termsheet_format", "plain_text")
    assumptions = run_state.config.assumptions
    engagement_ctx = _format_engagement_context(assumptions)

    cite_extractor = _make_citation_extractor(jurisdiction)

    # --- Phase 1: MPSR ---
    _run_mpsr_phase(run_state, context, adapters, checkpoint_dir, pause_handler)

    # POST_INTAKE checkpoint
    result = pause_handler(PauseForHumanInput(
        checkpoint_id=CheckpointId.POST_INTAKE,
        summary=(
            f"MPSR complete. Role: {assumptions.party_role} | "
            f"Type: {assumptions.contract_type} | Jurisdiction: {jurisdiction}. "
            "Proceed with intake analysis?"
        ),
        artifact_refs=["charter"],
        decision_options=[Decision.APPROVE, Decision.ABORT],
        edits_allowed_on=[],
        telemetry=run_state.snapshot(),
    ))
    if result.decision == Decision.ABORT:
        raise SystemExit("Aborted at POST_INTAKE.")

    cap_msg = run_state.check_loop_and_caps()
    if cap_msg:
        raise RuntimeError(f"Cap reached: {cap_msg}")

    # --- RAG phase: enrich context with treatise passages ---
    rag_statutory_ctx, rag_caselaw_ctx = _run_rag_phase(
        _reconstruct_charter_entries(run_state), adapters
    )

    charter_json_str = _charter_json(run_state)

    # --- Phase 2: Intake ---
    anth_fast = adapters.get("anthropic_fast")
    in_system = PromptBuilder.intake_system(jurisdiction, charter_json=charter_json_str)
    in_user = PromptBuilder.intake_user(termsheet_text, termsheet_format=termsheet_format)
    intake_result: IntakeOutput = anth_fast.call_structured_su(
        IntakeOutput, in_system, in_user, max_tokens=4096
    )
    _accrue_cost(run_state, anth_fast, "anthropic", "fast")
    run_state.record_artifact("intake", intake_result.model_dump(mode="json"))
    run_state.persist(checkpoint_dir)

    # SCOPE_CONFIRM checkpoint
    result = pause_handler(PauseForHumanInput(
        checkpoint_id=CheckpointId.SCOPE_CONFIRM,
        summary=(
            "Intake analysis complete."
            + (" BLOCKING CONFLICTS detected — human review required."
               if intake_result.requires_human_review else "")
            + " Proceed with framework?"
        ),
        artifact_refs=["intake"],
        decision_options=[Decision.APPROVE, Decision.ABORT],
        edits_allowed_on=["intake"],
        telemetry=run_state.snapshot(),
    ))
    if result.decision == Decision.ABORT:
        raise SystemExit("Aborted at SCOPE_CONFIRM.")

    cap_msg = run_state.check_loop_and_caps()
    if cap_msg:
        raise RuntimeError(f"Cap reached: {cap_msg}")

    # --- Phase 3: Framework ---
    anth_mid = adapters.get("anthropic_mid", adapters.get("anthropic_flagship"))
    fw_system = PromptBuilder.framework_system(
        jurisdiction, jurisdiction_language,
        contract_type=contract_type, mode_verb="drafted",
        charter_json=charter_json_str,
    )
    fw_user = PromptBuilder.framework_user(
        mode="draft", contract_text=termsheet_text,
        rag_statutory_context=rag_statutory_ctx,
    ) + "\n\n" + engagement_ctx
    fw_result: FrameworkOutput = anth_mid.call_structured_su(
        FrameworkOutput, fw_system, fw_user, max_tokens=8192
    )
    _accrue_cost(run_state, anth_mid, "anthropic", "mid")
    run_state.record_artifact("framework", fw_result.model_dump(mode="json"))
    run_state.persist(checkpoint_dir)

    cap_msg = run_state.check_loop_and_caps()
    if cap_msg:
        raise RuntimeError(f"Cap reached: {cap_msg}")

    # --- Phase 4: Drafter B2 ---
    requirements_spec_json = json.dumps(
        intake_result.model_dump(mode="json"), ensure_ascii=False
    )
    skeleton_json = json.dumps(fw_result.model_dump(mode="json"), ensure_ascii=False)

    dr_system = PromptBuilder.drafter_system(
        jurisdiction, jurisdiction_language,
        contract_type=contract_type, report_language=report_language,
        charter_json=charter_json_str,
    )
    dr_b2_user = PromptBuilder.drafter_user_b2(
        requirements_spec_json=requirements_spec_json,
        skeleton_clauses_json=skeleton_json,
        jurisdiction_language=jurisdiction_language,
    )
    draft_b2_text: str = anth_mid.call_text(dr_system, dr_b2_user, max_tokens=8192)
    _accrue_cost(run_state, anth_mid, "anthropic", "mid")
    run_state.record_artifact("draft_b2", draft_b2_text)
    run_state.persist(checkpoint_dir)

    cap_msg = run_state.check_loop_and_caps()
    if cap_msg:
        raise RuntimeError(f"Cap reached: {cap_msg}")

    # --- Phase 5: Red Team (self-adversarial) ---
    oai_red = adapters.get("openai_red_team", adapters.get("openai_flagship"))
    rt_system = PromptBuilder.red_team_system(
        jurisdiction, jurisdiction_language,
        contract_type=contract_type, target_label="draft contract",
        charter_json=charter_json_str,
    )
    rt_user = PromptBuilder.red_team_user(
        mode="self_adversarial", contract_text=draft_b2_text,
        target_label="draft contract",
        rag_statutory_context=rag_statutory_ctx,
        rag_treatise_context=rag_caselaw_ctx,
        jurisdiction_language=jurisdiction_language,
    ) + "\n\n" + engagement_ctx
    try:
        rt_result: RedTeamOutput = oai_red.call_structured_su(
            RedTeamOutput, rt_system, rt_user, max_tokens=8192
        )
        _accrue_cost(run_state, oai_red, "openai", "flagship")
    except Exception:
        rt_result = anth_mid.call_structured_su(
            RedTeamOutput, rt_system, rt_user, max_tokens=8192
        )
        _accrue_cost(run_state, anth_mid, "anthropic", "mid")

    run_state.record_artifact("red_team", rt_result.model_dump(mode="json"))
    run_state.persist(checkpoint_dir)

    # POST_SELF_RED checkpoint
    result = pause_handler(PauseForHumanInput(
        checkpoint_id=CheckpointId.POST_SELF_RED,
        summary="Self-adversarial Red Team complete. Proceeding to Charter Delta.",
        artifact_refs=["red_team", "draft_b2"],
        decision_options=[Decision.APPROVE, Decision.ABORT],
        edits_allowed_on=[],
        telemetry=run_state.snapshot(),
    ))
    if result.decision == Decision.ABORT:
        raise SystemExit("Aborted at POST_SELF_RED.")

    cap_msg = run_state.check_loop_and_caps()
    if cap_msg:
        raise RuntimeError(f"Cap reached: {cap_msg}")

    # --- Phase 6: Charter Delta ---
    charter_entries = _reconstruct_charter_entries(run_state)
    rt_findings = _reconstruct_rt_findings(run_state)
    rt_narrative = run_state.artifacts.get("red_team", {}).get("narrative", "")

    statute_deltas = compute_statute_deltas(
        rt_findings, rt_narrative, charter_entries, jurisdiction, cite_extractor
    )
    cd_system = PromptBuilder.charter_delta_residual_system(
        jurisdiction, jurisdiction_language=jurisdiction_language,
        charter_json=charter_json_str,
        precomputed_statute_deltas_json=json.dumps(statute_deltas, ensure_ascii=False),
    )
    cd_user = PromptBuilder.charter_delta_residual_user(
        red_team_findings_json=json.dumps(
            [f.model_dump(mode="json") for f in rt_findings], ensure_ascii=False
        )
    )
    cd_result: CharterDeltaOutput = anth_fast.call_structured_su(
        CharterDeltaOutput, cd_system, cd_user, max_tokens=2048
    )
    _accrue_cost(run_state, anth_fast, "anthropic", "fast")
    run_state.record_artifact("charter_delta", cd_result.model_dump(mode="json"))
    run_state.persist(checkpoint_dir)

    if cd_result.conceptual_delta_found:
        result = pause_handler(PauseForHumanInput(
            checkpoint_id=CheckpointId.DELTA_CONFIRM,
            summary=f"Charter Delta: {len(cd_result.new_conceptual_entries)} new entries.",
            artifact_refs=["charter_delta"],
            decision_options=[Decision.APPROVE, Decision.ABORT],
            edits_allowed_on=["charter_delta"],
            telemetry=run_state.snapshot(),
        ))
        if result.decision == Decision.ABORT:
            raise SystemExit("Aborted at DELTA_CONFIRM.")

    cap_msg = run_state.check_loop_and_caps()
    if cap_msg:
        raise RuntimeError(f"Cap reached: {cap_msg}")

    # --- Phase 7: Drafter B4 ---
    hardenings = [
        f.model_dump(mode="json") for f in rt_findings
        if f.severity in ("CRITICAL", "HIGH")
    ]
    dr_b4_user = PromptBuilder.drafter_user_b4(
        requirements_spec_json=requirements_spec_json,
        skeleton_clauses_json=skeleton_json,
        previous_draft=draft_b2_text,
        accepted_hardenings_json=json.dumps(hardenings, ensure_ascii=False),
        charter_delta_findings_json=json.dumps(cd_result.model_dump(mode="json"), ensure_ascii=False),
        jurisdiction_language=jurisdiction_language,
        report_language=report_language,
    )
    draft_b4_text: str = anth_mid.call_text(dr_system, dr_b4_user, max_tokens=8192)
    _accrue_cost(run_state, anth_mid, "anthropic", "mid")
    run_state.record_artifact("draft_b4", draft_b4_text)
    run_state.persist(checkpoint_dir)

    # POST_FINAL_DRAFT checkpoint
    result = pause_handler(PauseForHumanInput(
        checkpoint_id=CheckpointId.POST_FINAL_DRAFT,
        summary="Final draft (B4 synthesis) complete. Proceeding to audit.",
        artifact_refs=["draft_b4", "charter_delta"],
        decision_options=[Decision.APPROVE, Decision.ABORT],
        edits_allowed_on=["draft_b4"],
        telemetry=run_state.snapshot(),
    ))
    if result.decision == Decision.ABORT:
        raise SystemExit("Aborted at POST_FINAL_DRAFT.")

    cap_msg = run_state.check_loop_and_caps()
    if cap_msg:
        raise RuntimeError(f"Cap reached: {cap_msg}")

    # --- Phase 8: Audit ---
    aud_system = PromptBuilder.auditor_system(
        jurisdiction, jurisdiction_language=jurisdiction_language,
        report_language=report_language,
    )
    raw_cites = [f.cited_authority for f in rt_findings if f.cited_authority]
    unique_cites, _ = dedupe_citations(raw_cites, jurisdiction)

    # Pre-audit: batch-resolve any statute names unknown to the lookup table
    _audit_statutes = [
        parse_canonical_ref(c).statute for c in unique_cites if parse_canonical_ref(c) is not None
    ]
    resolve_unknown_statutes_batch(list(set(_audit_statutes)), anth_fast)

    audit_phase1_results: list[dict] = []
    _aud_cap = run_state.config.auditor_phase1_dispatches_cap
    for cite in unique_cites:
        if run_state.auditor_phase1_dispatches >= _aud_cap:
            break
        ref = parse_canonical_ref(cite)
        if ref is not None:
            egov_doc = fetch_egov_article(ref)
            retrieved_text = egov_doc.text if egov_doc is not None else "[retrieval not available]"
        else:
            retrieved_text = "[retrieval not available]"

        # Hard guard: skip LLM when no source text is available — prevents parametric verification
        if retrieved_text == "[retrieval not available]":
            audit_phase1_results.append({
                "citation": cite,
                "classification": "UNCERTAIN",
                "rationale": "retrieval_failed",
                "retrieved_text_quote": "",
            })
            run_state.auditor_phase1_dispatches += 1
            continue

        aud_user = PromptBuilder.auditor_user(
            citation_string=cite,
            document_claim_excerpt="",
            retrieved_source_text=retrieved_text,
        )
        aud_result: AuditCitationResult = anth_fast.call_structured_su(
            AuditCitationResult, aud_system, aud_user, max_tokens=1024
        )
        run_state.auditor_phase1_dispatches += 1
        _accrue_cost(run_state, anth_fast, "anthropic", "fast")
        audit_phase1_results.append({**aud_result.model_dump(mode="json"), "citation": cite})

    audit_p2 = auditor_phase2(draft_b4_text, charter_entries, jurisdiction, cite_extractor)
    run_state.record_artifact("audit", {"phase1": audit_phase1_results, "phase2": audit_p2})
    run_state.persist(checkpoint_dir)

    # POST_AUDIT checkpoint
    omitted = audit_p2.get("omitted_from_verdict", [])
    result = pause_handler(PauseForHumanInput(
        checkpoint_id=CheckpointId.POST_AUDIT,
        summary=(
            f"Audit complete. Phase 1: {len(audit_phase1_results)}/{len(unique_cites)} citations checked"
            + (f" (capped at {_aud_cap})" if len(audit_phase1_results) < len(unique_cites) else "")
            + f". Phase 2: {len(omitted)} charter statutes not addressed in final draft."
        ),
        artifact_refs=["audit", "draft_b4"],
        decision_options=[Decision.APPROVE, Decision.ABORT],
        edits_allowed_on=[],
        telemetry=run_state.snapshot(),
    ))
    if result.decision == Decision.ABORT:
        raise SystemExit("Aborted at POST_AUDIT.")

    return checkpoint_dir / "checkpoint.json"
