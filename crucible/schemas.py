from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from crucible.enums import (
    AuditClassification, CheckpointId, ClauseStatus, ConsensusLevel,
    Decision, ModelTier, Severity,
)


# ===========================================================================
# MPSR panel output schemas
# ===========================================================================

class MpsrStatute(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    article: str
    why: str


class MpsrDoctrineTheme(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str
    why: str


class MpsrMandatoryClause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clause: str
    basis: str


class MpsrRiskDomain(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain: str
    trigger_keywords: list[str]
    severity: Literal["high", "medium", "low"]


class MpsrJurisdictionalTrap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trap: str
    why: str


class MpsrRagSeed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layer: Literal["statutory", "treatise", "case", "internal"]
    keywords: list[str]


class MpsrPanelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applicable_statutes: list[MpsrStatute]
    applicable_doctrine_themes: list[MpsrDoctrineTheme]
    mandatory_clauses: list[MpsrMandatoryClause]
    risk_domains: list[MpsrRiskDomain]
    jurisdictional_traps: list[MpsrJurisdictionalTrap]
    rag_query_seeds: list[MpsrRagSeed]


# ===========================================================================
# Framework schemas
# ===========================================================================

class FrameworkClause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    status: ClauseStatus
    deficiency_note: str | None = None
    basis_citation: str | None = None


class FrameworkMissingClauseProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clause_name: str
    proposed_language: str
    basis_citation: str


class FrameworkOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clauses: list[FrameworkClause]
    missing_clauses_proposed_language: list[FrameworkMissingClauseProposal]
    narrative: str = ""


# ===========================================================================
# Red team schemas
# ===========================================================================

class RedTeamFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clause_ref: str
    risk_summary: str = Field(..., max_length=400)
    worst_case_exposure: str
    cited_authority: str | None = None
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    charter_source: str
    charter_consensus_level: ConsensusLevel | None = None
    domain: str
    applicable: bool = True


class RedTeamOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: list[RedTeamFinding]
    narrative: str = ""


# ===========================================================================
# Charter delta schemas
# ===========================================================================

class CharterDeltaConceptualEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["risk_domain", "doctrine_theme", "doctrine"]
    name: str
    rationale: str = Field(..., max_length=500)
    red_team_finding_ref: str


class CharterDeltaOutput(BaseModel):
    """Strict binary classification. Default NOT_NEW (conceptual_delta_found=false)."""
    model_config = ConfigDict(extra="forbid")
    conceptual_delta_found: bool
    new_conceptual_entries: list[CharterDeltaConceptualEntry] = Field(default_factory=list)
    summary: str

    def model_post_init(self, __context) -> None:
        if not self.conceptual_delta_found and self.new_conceptual_entries:
            raise ValueError("conceptual_delta_found=False but new_conceptual_entries non-empty")
        if self.conceptual_delta_found and not self.new_conceptual_entries:
            raise ValueError("conceptual_delta_found=True but new_conceptual_entries empty")


# ===========================================================================
# Intake schemas
# ===========================================================================

class IntakeConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    termsheet_value: str
    conflict: str
    severity: Literal["BLOCKING", "WARNING"]
    suggested_resolution: str


class IntakeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirements_spec: dict[str, Any]
    conflicts: list[IntakeConflict]
    requires_human_review: bool


# ===========================================================================
# Audit citation result
# ===========================================================================

class AuditCitationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    classification: Literal["VERIFIED", "UNCERTAIN", "DISPUTED"]
    rationale: str = Field(..., max_length=200)
    retrieved_text_quote: str | None = None


# ===========================================================================
# Charter entry
# ===========================================================================

class CharterEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["statute", "doctrine_theme", "mandatory_clause", "risk_domain",
                  "jurisdictional_trap", "rag_seed"]
    canonical_reference: str
    payload: dict[str, Any]
    consensus_level: ConsensusLevel
    raised_by_count: int
    panel_size: int


# ===========================================================================
# Telemetry snapshot (must precede PauseForHumanInput)
# ===========================================================================

class TelemetrySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    spend_usd: float
    elapsed_seconds: float
    subagent_invocations: int
    tool_calls: int
    last_subagent: str | None
    last_checkpoint: str | None
    cap_usd: float
    cap_wall_seconds: int
    rag_cache_hits: int = 0
    rag_cache_misses: int = 0
    cache_rebuilds_count: int = 0
    cache_rebuild_overdraft_total_usd: float = 0.0
    ttl_mode_degraded: bool = False


# ===========================================================================
# RAG query schemas
# ===========================================================================

class RagQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layer: Literal["statutory", "case", "internal"]
    keywords: list[str] = Field(..., min_length=1)
    scope_tags: list[str]
    follow_references: bool = True
    max_recursion_depth: Literal[1] = Field(
        default=1,
        description="Hard-locked at 1. Single-hop reference following only.",
    )
    timeout_seconds: float = Field(default=60.0, le=60.0)
    max_results: int = Field(default=10, ge=1, le=50)
    max_referenced_docs: int = Field(default=20, ge=0, le=100)


class RagQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documents: list[dict[str, Any]]
    referenced_documents: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_time_ms: int
    truncated: bool
    referenced_truncated: bool = False
    cache_hit: bool = False


# ===========================================================================
# Engagement assumptions (practitioner-supplied deal context)
# ===========================================================================

class EngagementAssumptions(BaseModel):
    """Practitioner-supplied facts about the deal, injected into subagent prompts."""
    model_config = ConfigDict(extra="forbid")
    party_role: Literal["vendor", "customer", "neutral"] = "neutral"
    contract_type: str = "Commercial Agreement"
    business_context: str = ""
    counterparty_type: str = ""
    cross_border: bool = False
    personal_data: bool = False
    personnel_involved: bool = False
    additional: dict[str, str] = Field(default_factory=dict)


# ===========================================================================
# Other tool schemas
# ===========================================================================

class BrowserFetchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    selector: str | None = None
    timeout_seconds: float = Field(default=90.0, le=90.0)


class BrowserFetchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str
    fetched_at: datetime
    status_code: int


class CiteExtractInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    jurisdiction: str


class CiteExtractResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    citations: list[str]


class LoadTemplateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_type: str
    jurisdiction: str


class LoadTemplateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    template_text: str | None
    source: str


class WriteArtifactInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section_id: str
    content: str
    language: str


class WriteArtifactResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: str
    bytes_written: int


# ===========================================================================
# pause_for_human and budget_check schemas
# ===========================================================================

class PauseForHumanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint_id: CheckpointId
    summary: str = Field(..., max_length=2000)
    artifact_refs: list[str] = Field(default_factory=list)
    decision_options: list[Decision] = Field(..., min_length=1)
    edits_allowed_on: list[str] = Field(default_factory=list)
    telemetry: TelemetrySnapshot


class PauseForHumanResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Decision
    edits: dict[str, str] = Field(default_factory=dict)
    budget_extension_usd: float | None = None
    note: str | None = None


class BudgetCheckInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    next_subagent: str
    estimated_input_tokens: int = Field(..., ge=0)
    estimated_output_tokens: int = Field(..., ge=0)
    provider: Literal["anthropic", "openai", "google"]
    tier: ModelTier
    model_string: str


class BudgetCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed: bool
    projected_total_usd: float
    cap_usd: float
    remaining_usd: float
    safety_margin: float = 1.5
    reason: str | None = None


# ===========================================================================
# Concurrency-safe dispatch result dataclasses
# ===========================================================================

@dataclass
class MpsrProviderResult:
    """Result from a single MPSR provider dispatch. Always populated."""
    provider: str
    success: bool
    response: Any | None = None
    error: Exception | None = None
    elapsed_seconds: float = 0.0
    retries_exhausted: bool = False

    def __post_init__(self):
        if self.success and self.response is None:
            raise ValueError(
                f"MpsrProviderResult for {self.provider}: success=True but response is None"
            )
        if not self.success and self.error is None:
            raise ValueError(
                f"MpsrProviderResult for {self.provider}: success=False but error is None"
            )


@dataclass
class MpsrPanelDispatchResult:
    """Aggregated result from all MPSR provider dispatches."""
    results: list[MpsrProviderResult] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0

    @property
    def succeeded(self) -> list[Any]:
        return [r.response for r in self.results if r.success]

    @property
    def failed(self) -> list[MpsrProviderResult]:
        return [r for r in self.results if not r.success]

    @property
    def effective_panel_size(self) -> int:
        return len(self.succeeded)

    @property
    def all_succeeded(self) -> bool:
        return len(self.failed) == 0

    @property
    def target_panel_size(self) -> int:
        return len(self.results)


# ===========================================================================
# Rate limit dataclasses
# ===========================================================================

@dataclass
class RateLimitConfig:
    tokens_per_minute: int
    requests_per_minute: int


@dataclass
class RateLimitBucket:
    config: RateLimitConfig
    tokens_remaining: int
    requests_remaining: int
    last_refill: float = field(default_factory=time.monotonic)
    _lock: Lock = field(default_factory=Lock)

    @classmethod
    def from_config(cls, cfg: RateLimitConfig) -> "RateLimitBucket":
        return cls(
            config=cfg,
            tokens_remaining=cfg.tokens_per_minute,
            requests_remaining=cfg.requests_per_minute,
        )

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed >= 60.0:
            self.tokens_remaining = self.config.tokens_per_minute
            self.requests_remaining = self.config.requests_per_minute
            self.last_refill = now
        else:
            ratio = elapsed / 60.0
            self.tokens_remaining = min(
                self.config.tokens_per_minute,
                self.tokens_remaining + int(self.config.tokens_per_minute * ratio),
            )
            self.requests_remaining = min(
                self.config.requests_per_minute,
                self.requests_remaining + int(self.config.requests_per_minute * ratio),
            )
            self.last_refill = now

    def check_and_consume(self, tokens_needed: int) -> tuple[bool, float]:
        with self._lock:
            self._refill()
            if self.tokens_remaining >= tokens_needed and self.requests_remaining >= 1:
                self.tokens_remaining -= tokens_needed
                self.requests_remaining -= 1
                return (True, 0.0)
            tokens_short = max(tokens_needed - self.tokens_remaining, 0)
            wait_for_tokens = (tokens_short / self.config.tokens_per_minute) * 60.0 if tokens_short else 0.0
            wait_for_requests = 60.0 if self.requests_remaining < 1 else 0.0
            return (False, max(wait_for_tokens, wait_for_requests))


# ===========================================================================
# Cache prefix and rebuild decision dataclasses
# ===========================================================================

@dataclass
class CachedPrefix:
    """Represents an active cached prefix on a provider."""
    provider: Literal["anthropic", "google"]
    cache_key: str
    prefix_payload: dict
    cached_token_count: int
    tier: ModelTier = ModelTier.FLAGSHIP
    created_at: float = field(default_factory=time.monotonic)
    last_pinged_at: float = field(default_factory=time.monotonic)
    ttl_mode: Literal["1h", "5m"] = "1h"
    last_probe_status: Literal["fresh", "stale", "unknown"] = "unknown"


@dataclass
class RebuildDecision:
    cache_key: str
    action: Literal["fresh_no_action", "rebuild", "skip_rebuild_uneconomical",
                    "rebuild_with_overdraft", "pause_before_rebuild"]
    rebuild_cost_usd: float = 0.0
    with_rebuild_total: float = 0.0
    without_rebuild_total: float = 0.0
    overdraft_amount_usd: float = 0.0


# ===========================================================================
# RAG expander dataclasses
# ===========================================================================

@dataclass(frozen=True)
class ArticleRef:
    """A reference to a specific article in a statute. Frozen for set membership."""
    statute: str
    article_num: int
    paragraph_num: int | None = None
    item_num: int | None = None

    @property
    def canonical(self) -> str:
        parts = [self.statute, f"第{self.article_num}条"]
        if self.paragraph_num is not None:
            parts.append(f"第{self.paragraph_num}項")
        if self.item_num is not None:
            parts.append(f"第{self.item_num}号")
        return "".join(parts)

    def with_statute(self, statute: str) -> "ArticleRef":
        return ArticleRef(
            statute=statute,
            article_num=self.article_num,
            paragraph_num=self.paragraph_num,
            item_num=self.item_num,
        )


@dataclass
class Document:
    """An article document used by the expander."""
    canonical_ref: str
    text: str
    statute: str = ""
    article_num: int = 0
    expansion_metadata: dict = field(default_factory=dict)


@dataclass
class ExpansionResult:
    """Return value of expand_references_bfs."""
    document: Document
    expanded_count: int
    truncated: bool
    max_depth_reached: int
    expanded_canonicals: list[str]
    skipped_for_cycle: int = 0
    skipped_for_depth: int = 0
    skipped_for_article_cap: int = 0
    skipped_for_char_cap: int = 0
