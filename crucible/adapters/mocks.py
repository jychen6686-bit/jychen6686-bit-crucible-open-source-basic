from __future__ import annotations

from typing import Any

from crucible.enums import ConsensusLevel, Decision
from crucible.schemas import (
    AuditCitationResult,
    CharterDeltaOutput,
    FrameworkClause,
    FrameworkMissingClauseProposal,
    FrameworkOutput,
    IntakeConflict,
    IntakeOutput,
    MpsrDoctrineTheme,
    MpsrJurisdictionalTrap,
    MpsrMandatoryClause,
    MpsrPanelOutput,
    MpsrRagSeed,
    MpsrRiskDomain,
    MpsrStatute,
    PauseForHumanInput,
    PauseForHumanResult,
    RedTeamFinding,
    RedTeamOutput,
)


class MockMpsrProvider:
    """Mock MPSR panel dispatch_fn. Returns pre-canned MpsrPanelOutput."""

    def __init__(self, fail: bool = False, provider_name: str = "mock") -> None:
        self.fail = fail
        self.provider_name = provider_name
        self.name = provider_name

    def __call__(self, provider_config: Any) -> MpsrPanelOutput:
        if self.fail:
            raise RuntimeError("mock_provider_failure")
        return MpsrPanelOutput(
            applicable_statutes=[
                MpsrStatute(name="民法", article="第415条", why="Breach of contract damages"),
                MpsrStatute(name="民法", article="第709条", why="Tort liability"),
            ],
            applicable_doctrine_themes=[
                MpsrDoctrineTheme(topic="force_majeure_excuse", why="doctrine of impossibility of performance under 民法第415条"),
            ],
            mandatory_clauses=[
                MpsrMandatoryClause(clause="governing_law", basis="Private international law"),
                MpsrMandatoryClause(clause="dispute_resolution", basis="Civil procedure"),
            ],
            risk_domains=[
                MpsrRiskDomain(
                    domain="payment_default",
                    trigger_keywords=["payment", "default", "overdue"],
                    severity="high",
                ),
            ],
            jurisdictional_traps=[
                MpsrJurisdictionalTrap(
                    trap="japan_consumer_contract_act",
                    why="Limits liquidated damages to actual loss",
                ),
            ],
            rag_query_seeds=[
                MpsrRagSeed(layer="statutory", keywords=["民法", "契約"]),
                MpsrRagSeed(layer="treatise", keywords=["損害賠償", "履行不能"]),
            ],
        )


class MockAuditCitationProvider:
    """Mock audit citation verifier."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def __call__(self, citation: str, jurisdiction: str) -> AuditCitationResult:
        if self.fail:
            raise RuntimeError("mock_audit_failure")
        return AuditCitationResult(
            classification="VERIFIED",
            rationale="Mock: citation verified against mock corpus",
            retrieved_text_quote=None,
        )


class MockCharterDeltaProvider:
    """Mock charter delta classifier."""

    def __init__(self, fail: bool = False, delta_found: bool = False) -> None:
        self.fail = fail
        self.delta_found = delta_found

    def __call__(self, *args: Any, **kwargs: Any) -> CharterDeltaOutput:
        if self.fail:
            raise RuntimeError("mock_charter_delta_failure")
        return CharterDeltaOutput(
            conceptual_delta_found=False,
            new_conceptual_entries=[],
            summary="mock: no delta found",
        )


class MockCacheProbeExecutor:
    """Mock cache probe executor."""

    def __init__(self, cache_hit: bool = True) -> None:
        self.cache_hit = cache_hit

    def __call__(self, prefix: Any) -> dict[str, Any]:
        return {"cache_hit": self.cache_hit}


class MockRebuildExecutor:
    """Mock cache rebuild executor. Records that rebuild was called."""

    def __init__(self) -> None:
        self.rebuild_count = 0
        self.rebuilt_keys: list[str] = []

    def __call__(self, prefix: Any) -> dict[str, Any]:
        self.rebuild_count += 1
        self.rebuilt_keys.append(getattr(prefix, "cache_key", str(prefix)))
        return {"rebuilt": True}


class MockAdapter:
    """Unified mock adapter for all pipeline phases (no API calls)."""

    def __init__(self, provider_name: str = "mock") -> None:
        self.provider_name = provider_name
        self.name = provider_name
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0

    def call_mpsr(self, system: str, user: str, max_tokens: int = 4096) -> MpsrPanelOutput:
        return MockMpsrProvider(provider_name=self.provider_name)(None)

    def call_structured_su(self, schema_cls: type, system: str, user: str,
                           max_tokens: int = 4096) -> Any:
        return self._make_mock(schema_cls)

    def call_structured(self, schema_cls: type, system: str, user: str,
                        max_tokens: int = 4096) -> Any:
        return self._make_mock(schema_cls)

    def call_verdict(self, system: str, user: str, max_tokens: int = 8192) -> str:
        return (
            "## Overall Risk Rating\nMEDIUM — Mock verdict. No real analysis performed.\n\n"
            "## RED LINE Items\nNone identified in mock mode.\n\n"
            "## Negotiate Items\nNone in mock mode.\n\n"
            "## Accept Items\nAll clauses accepted (mock).\n\n"
            "## Missing Clauses\nNone in mock mode.\n\n"
            "## Legal Assessment\nMock verdict — engage real providers for actual analysis."
        )

    def call_text(self, system: str, user: str, max_tokens: int = 4096) -> str:
        return "Mock text output."

    def _make_mock(self, schema_cls: type) -> Any:
        name = schema_cls.__name__
        if name == "FrameworkOutput":
            return FrameworkOutput(
                clauses=[
                    FrameworkClause(
                        name="governing_law",
                        status="PRESENT",
                        basis_citation="民法第1条",
                    )
                ],
                missing_clauses_proposed_language=[],
                narrative="Mock framework analysis.",
            )
        if name == "RedTeamOutput":
            return RedTeamOutput(
                findings=[
                    RedTeamFinding(
                        clause_ref="§1",
                        risk_summary="Mock risk finding",
                        worst_case_exposure="Mock exposure scenario",
                        cited_authority="民法第415条",
                        severity="LOW",
                        charter_source="core_domain",
                        charter_consensus_level="majority",
                        domain="liability",
                    )
                ],
                narrative="Mock red team analysis.",
            )
        if name == "CharterDeltaOutput":
            return CharterDeltaOutput(
                conceptual_delta_found=False,
                new_conceptual_entries=[],
                summary="Mock: no conceptual delta found.",
            )
        if name == "AuditCitationResult":
            return AuditCitationResult(
                classification="VERIFIED",
                rationale="Mock verification",
                retrieved_text_quote=None,
            )
        if name == "IntakeOutput":
            return IntakeOutput(
                requirements_spec={},
                conflicts=[],
                requires_human_review=False,
            )
        raise NotImplementedError(f"MockAdapter: no mock for {name}")


class MockPauseHandler:
    """Mock pause_for_human handler. Records checkpoint was triggered."""

    def __init__(self, decision: Decision = Decision.APPROVE) -> None:
        self.decision = decision
        self.calls: list[PauseForHumanInput] = []

    def __call__(self, input: PauseForHumanInput) -> PauseForHumanResult:
        self.calls.append(input)
        return PauseForHumanResult(
            decision=self.decision,
            edits={},
            budget_extension_usd=None,
            note=None,
        )
