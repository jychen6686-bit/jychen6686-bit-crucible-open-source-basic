"""
CRUCIBLE v3.0 — Managed Agent entrypoint.

Usage:
  python managed_agent_main.py init <config_path>
  python managed_agent_main.py smoke [--mock]
  python managed_agent_main.py run --track A|B --contract <path> [--mock] [--config <path>]
  python managed_agent_main.py resume --run-dir <path> [--decision APPROVE|REJECT|...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_init(args: argparse.Namespace) -> None:
    """Load and validate config YAML, verify environment variables."""
    try:
        import yaml
    except ImportError:
        print("pyyaml required: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    fill_in_count = _count_fill_in(config)
    print(f"CRUCIBLE v3.0 config loaded: {config_path}")
    print(f"  Jurisdiction: {config.get('jurisdiction', {}).get('primary', '<missing>')}")
    print(f"  Crucible version: {config.get('deployment', {}).get('crucible_version', '<missing>')}")
    if fill_in_count > 0:
        print(f"  WARNING: {fill_in_count} <FILL_IN> placeholder(s) remain — set model strings and pricing before running.")
    else:
        print("  All values configured.")
    print("Init complete.")


def _count_fill_in(obj: object) -> int:
    if isinstance(obj, str):
        return obj.count("<FILL_IN>")
    if isinstance(obj, dict):
        return sum(_count_fill_in(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_fill_in(v) for v in obj)
    return 0


def cmd_smoke(args: argparse.Namespace) -> None:
    """Run a minimal 3-provider MPSR dispatch with mock providers."""
    from crucible.adapters.mocks import MockMpsrProvider
    from crucible.mpsr import (
        dispatch_mpsr_panel, check_panel_degradation,
        merge_mpsr_panel_responses, format_panel_failure_summary,
    )
    from crucible.semantic_dedup import semantic_dedup_charter_entries
    from crucible.cache_routing import TtlMode
    from crucible.runtime import RunConfig, RunState
    from crucible.persistence import restore_from_checkpoint
    import tempfile, os

    print("CRUCIBLE v3.0 smoke test (mock mode)")
    print("=" * 50)

    # 3-provider mock MPSR dispatch
    providers = [
        MockMpsrProvider(provider_name="anthropic"),
        MockMpsrProvider(provider_name="openai"),
        MockMpsrProvider(provider_name="google"),
    ]

    print("1. Dispatching MPSR panel (3 mock providers)...")
    dispatch_result = dispatch_mpsr_panel(
        providers=providers,
        dispatch_fn=lambda p: p(p),
        timeout_seconds=30.0,
    )
    print(f"   Succeeded: {dispatch_result.effective_panel_size}/3")
    assert dispatch_result.effective_panel_size == 3, "Expected 3/3 panel"

    # Test degraded panel path
    print("2. Testing degraded panel detection (1 failing provider)...")
    degraded_providers = [
        MockMpsrProvider(provider_name="anthropic"),
        MockMpsrProvider(provider_name="openai", fail=True),
        MockMpsrProvider(provider_name="google"),
    ]
    degraded_result = dispatch_mpsr_panel(
        providers=degraded_providers,
        dispatch_fn=lambda p: p(p),
        timeout_seconds=30.0,
    )
    assert check_panel_degradation(degraded_result, target_panel_size=3), "Expected degraded"
    print(f"   Degraded panel detected: {degraded_result.effective_panel_size}/3")
    print(f"   Summary:\n{format_panel_failure_summary(degraded_result)[:200]}")

    # Charter merge
    print("3. Merging panel responses into Charter...")
    panel_size = dispatch_result.effective_panel_size
    charter_entries, disagreement_log = merge_mpsr_panel_responses(
        dispatch_result.succeeded, panel_size, jurisdiction="Japan"
    )
    charter_entries, dedup_log = semantic_dedup_charter_entries(
        charter_entries, panel_size, embed_fn=None
    )
    categories = {e.type for e in charter_entries}
    print(f"   Charter entries: {len(charter_entries)} across categories: {categories}")
    assert len(charter_entries) > 0, "Expected charter entries"

    # Structured output validation
    print("4. Validating structured output schemas...")
    from crucible.schemas import TelemetrySnapshot
    from crucible.enums import CheckpointId
    snap = TelemetrySnapshot(
        spend_usd=0.0, elapsed_seconds=0.0, subagent_invocations=0,
        tool_calls=0, last_subagent=None, last_checkpoint=None,
        cap_usd=25.0, cap_wall_seconds=7200,
    )
    assert snap.ttl_mode_degraded is False
    print("   TelemetrySnapshot validated.")

    # TTL mode
    print("5. Checking TTL mode initialization...")
    ttl = TtlMode()
    assert ttl.mode == "1h"
    assert not ttl.degraded
    ttl.degrade_to_5m()
    assert ttl.mode == "5m"
    assert ttl.degraded
    print("   TTL mode: 1h → 5m degradation OK.")

    # Checkpoint persist + restore
    print("6. Testing checkpoint persist + restore...")
    config = RunConfig(
        jurisdiction="Japan",
        jurisdiction_language="Japanese",
        report_language_default="English",
    )
    pricing_table = {
        ("anthropic", "flagship"): {"input": 3.0, "output": 15.0},
        ("anthropic", "fast"): {"input": 0.25, "output": 1.25},
        ("openai", "flagship"): {"input": 5.0, "output": 15.0},
        ("google", "flagship"): {"input": 1.0, "output": 3.0},
    }
    run_state = RunState(config=config, pricing_table=pricing_table)
    run_state.record_artifact("charter", [e.model_dump(mode="json") for e in charter_entries])
    run_state.spend_usd = 1.23

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = run_state.persist(tmpdir)
        print(f"   Persisted to: {checkpoint_path}")
        restored = restore_from_checkpoint(checkpoint_path)
        assert restored.run_id == run_state.run_id
        assert abs(restored.spend_usd - 1.23) < 1e-9
        assert "charter" in restored.artifacts
        print(f"   Restored run_id={restored.run_id[:8]}... spend=${restored.spend_usd:.2f}")

    print()
    print("=" * 50)
    print("SMOKE TEST PASSED — all 6 checks OK")


def _load_yaml_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("pyyaml required: pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_pricing_table(cfg: dict) -> dict:
    pricing: dict = {}
    for provider, tiers in cfg.get("pricing_usd_per_mtok", {}).items():
        for tier, rates in tiers.items():
            if isinstance(rates, dict):
                inp = rates.get("input")
                out = rates.get("output")
                if isinstance(inp, (int, float)) and isinstance(out, (int, float)):
                    pricing[(provider, tier)] = {"input": float(inp), "output": float(out)}
    return pricing or {
        ("anthropic", "flagship"): {"input": 5.0, "output": 25.0},
        ("anthropic", "mid"): {"input": 3.0, "output": 15.0},
        ("anthropic", "fast"): {"input": 1.0, "output": 5.0},
        ("openai", "flagship"): {"input": 5.0, "output": 30.0},
        ("openai", "standard"): {"input": 2.5, "output": 15.0},
        ("google", "flagship"): {"input": 2.0, "output": 12.0},
        ("google", "standard"): {"input": 1.25, "output": 10.0},
    }


def _load_assumptions(path: str | None) -> "EngagementAssumptions":
    from crucible.schemas import EngagementAssumptions
    if not path:
        return EngagementAssumptions()
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    try:
        return EngagementAssumptions.model_validate(data)
    except Exception as e:
        print(f"Invalid assumptions file: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    """Start a Track A or B run."""
    import os
    from crucible.runtime import RunConfig, RunState
    from crucible.orchestrator import run_track_a, run_track_b
    from crucible.adapters.mocks import MockMpsrProvider, MockAdapter
    from crucible.adapters.egov_adapter import fetch_egov_article
    from crucible.cache_routing import TtlMode

    contract_path = Path(args.contract) if args.contract else None
    contract_text = ""
    if contract_path and contract_path.exists():
        contract_text = contract_path.read_text(encoding="utf-8")

    # Load config and assumptions
    cfg = _load_yaml_config(args.config) if args.config else {}
    jur_cfg = cfg.get("jurisdiction", {})
    guardrails = cfg.get("guardrails", {})
    models_cfg = cfg.get("models", {})

    assumptions = _load_assumptions(getattr(args, "assumptions", None))
    if assumptions.additional or assumptions.business_context or assumptions.party_role != "neutral":
        print(f"  Assumptions loaded: party_role={assumptions.party_role}, "
              f"contract_type={assumptions.contract_type!r}"
              + (f", {len(assumptions.additional)} additional" if assumptions.additional else ""))
    else:
        print("  No assumptions file provided — using defaults.")

    config = RunConfig(
        jurisdiction=jur_cfg.get("primary", "Japan"),
        jurisdiction_language=jur_cfg.get("language", "Japanese"),
        report_language_default="English",
        budget_cap_usd=float(guardrails.get("budget_cap_usd", 25.0)),
        budget_cap_overdraft_allowance_usd=float(
            guardrails.get("budget_cap_overdraft_allowance_usd", 5.0)
        ),
        assumptions=assumptions,
    )
    pricing_table = _build_pricing_table(cfg)

    run_state = RunState(config=config, pricing_table=pricing_table)
    run_dir = Path("./runs") / run_state.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting Track {args.track} run {run_state.run_id[:8]}...")
    print(f"Run directory: {run_dir}")

    if args.mock:
        providers = [
            MockMpsrProvider(provider_name="anthropic"),
            MockMpsrProvider(provider_name="openai"),
            MockMpsrProvider(provider_name="google"),
        ]
        mock_adapter = MockAdapter()
        adapters = {
            "mpsr_dispatch_fn": lambda p: p(p),
            "anthropic_flagship": mock_adapter,
            "anthropic_fast": mock_adapter,
            "openai_flagship": mock_adapter,
            "google_flagship": mock_adapter,
        }
        context = {
            "mpsr_providers": providers,
            "contract_type": assumptions.contract_type,
            "business_context": assumptions.business_context,
            "counterparty_type": assumptions.counterparty_type,
            "cross_border_flag": assumptions.cross_border,
            "data_flag": assumptions.personal_data,
            "personnel_flag": assumptions.personnel_involved,
            "report_language": "English",
            "article_fetcher": lambda ref: None,  # no network in mock mode
        }
    else:
        # Real provider mode
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        google_key = os.environ.get("GOOGLE_API_KEY", "")

        missing = [k for k, v in [
            ("ANTHROPIC_API_KEY", anthropic_key),
            ("OPENAI_API_KEY", openai_key),
            ("GOOGLE_API_KEY", google_key),
        ] if not v]
        if missing:
            print(f"Missing environment variables: {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)

        def _model(role: str, fallback: str) -> str:
            m = models_cfg.get(role, {}).get("model_string", fallback)
            if not m or m == "<FILL_IN>":
                print(f"WARNING: model_string for '{role}' not set, using '{fallback}'")
                return fallback
            return m

        ttl_mode = TtlMode()

        from crucible.adapters.anthropic_adapter import AnthropicAdapter
        from crucible.adapters.openai_adapter import OpenAIAdapter
        from crucible.adapters.google_adapter import GoogleAdapter

        anth_flag = AnthropicAdapter(anthropic_key, _model("mpsr_anthropic", "claude-opus-4-6"), ttl_mode)
        anth_mid = AnthropicAdapter(anthropic_key, _model("framework", "claude-sonnet-4-6"), ttl_mode)
        anth_consolidator = AnthropicAdapter(anthropic_key, _model("consolidator", "claude-sonnet-4-6"), ttl_mode)
        anth_fast = AnthropicAdapter(anthropic_key, _model("charter_delta_residual", "claude-haiku-4-5-20251001"), ttl_mode)
        oai_flag = OpenAIAdapter(openai_key, _model("mpsr_openai", "gpt-5.4"))
        oai_red_team = OpenAIAdapter(openai_key, _model("red_team", "gpt-5.5"))
        goog_flag = GoogleAdapter(google_key, _model("verdict", "gemini-3.1-pro-preview"))

        # Real MPSR providers
        class _RealMpsrProvider:
            def __init__(self, name, adapter):
                self.name = name
                self._adapter = adapter

            def __call__(self, _ignored):
                from crucible.prompts import PromptBuilder
                from crucible.orchestrator import _format_engagement_context
                sys_p = PromptBuilder.mpsr_panel_system(
                    config.jurisdiction, assumptions.contract_type, config.jurisdiction_language
                )
                usr_p = PromptBuilder.mpsr_panel_user(
                    assumptions.contract_type, args.track, config.jurisdiction,
                    engagement_context_block=_format_engagement_context(assumptions),
                )
                return self._adapter.call_mpsr(sys_p, usr_p)

        providers = [
            _RealMpsrProvider("anthropic", anth_flag),
            _RealMpsrProvider("openai", oai_flag),
            _RealMpsrProvider("google", GoogleAdapter(google_key, _model("mpsr_google", "gemini-2.5-pro"))),
        ]

        adapters = {
            "mpsr_dispatch_fn": lambda p: p(p),
            "anthropic_flagship": anth_flag,
            "anthropic_mid": anth_mid,
            "anthropic_fast": anth_fast,
            "consolidator": anth_consolidator,
            "openai_flagship": oai_flag,
            "openai_red_team": oai_red_team,
            "google_flagship": goog_flag,
        }

        # --- RAG adapter (optional; graceful degradation if index not built) ---
        _rag_db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "rag_index", "chroma_db",
        )
        _rag_db_path = os.path.normpath(_rag_db_path)
        if os.path.isdir(_rag_db_path):
            try:
                from crucible.adapters.chroma_rag_adapter import ChromaRagAdapter
                adapters["rag_adapter"] = ChromaRagAdapter(
                    chroma_db_path=_rag_db_path,
                    gemini_api_key=google_key,
                )
                print(f"  RAG adapter: {adapters['rag_adapter']}")
            except Exception as _rag_err:
                print(f"  RAG adapter unavailable: {_rag_err!r} — continuing without RAG")
        else:
            print(f"  RAG index not found at {_rag_db_path} — run rag_build_pipeline.py to build")
        context = {
            "mpsr_providers": providers,
            "contract_type": assumptions.contract_type,
            "business_context": assumptions.business_context,
            "counterparty_type": assumptions.counterparty_type,
            "cross_border_flag": assumptions.cross_border,
            "data_flag": assumptions.personal_data,
            "personnel_flag": assumptions.personnel_involved,
            "report_language": "English",
            "article_fetcher": fetch_egov_article,
        }

    if args.mock:
        from crucible.schemas import PauseForHumanResult
        from crucible.enums import Decision as _Decision

        def pause_handler(inp):
            print(f"\n  [mock] Checkpoint: {inp.checkpoint_id.value} — auto-approved")
            return PauseForHumanResult(decision=_Decision.APPROVE, edits={},
                                       budget_extension_usd=None, note=None)
    elif getattr(args, "auto_approve", False):
        from crucible.schemas import PauseForHumanResult
        from crucible.enums import Decision as _Decision

        def pause_handler(inp):
            print(f"\nCheckpoint: {inp.checkpoint_id.value} — auto-approved")
            return PauseForHumanResult(decision=_Decision.APPROVE, edits={},
                                       budget_extension_usd=None, note=None)
    else:
        from crucible.adapters.managed_agent_ui import pause_for_human_managed_agent

        def pause_handler(inp):
            result = pause_for_human_managed_agent(inp, run_state, run_dir)
            print(f"\nCheckpoint reached: {inp.checkpoint_id.value}")
            print(f"context_summary.md written to {run_dir}")
            print(f"Resume with: python managed_agent_main.py resume --run-dir {run_dir} --decision APPROVE")
            return result

    try:
        if args.track == "A":
            run_track_a(run_state, contract_text, context, adapters, run_dir, pause_handler)
        else:
            run_track_b(run_state, contract_text, context, adapters, run_dir, pause_handler)
        print(f"Run completed. Artifacts in: {run_dir}")
    except SystemExit as e:
        print(f"Run halted: {e}")
    except RuntimeError as e:
        print(f"Run error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_resume(args: argparse.Namespace) -> None:
    """Resume a run from checkpoint."""
    from crucible.persistence import restore_from_checkpoint
    from crucible.adapters.managed_agent_ui import (
        load_pending_checkpoint, load_context_summary, clear_pending_checkpoint
    )
    from crucible.enums import Decision

    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / "checkpoint.json"

    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # Load context summary for context injection
    context_summary = load_context_summary(run_dir)
    if context_summary:
        print("Context summary from last checkpoint:")
        print("-" * 40)
        print(context_summary[:1000])
        print("-" * 40)

    # Check for pending checkpoint
    pending = load_pending_checkpoint(run_dir)
    if pending:
        print(f"Pending checkpoint: {pending['checkpoint_id']}")
        print(f"Summary: {pending['summary'][:200]}")
        print(f"Options: {', '.join(pending['decision_options'])}")

    run_state = restore_from_checkpoint(checkpoint_path)
    print(f"Restored run {run_state.run_id[:8]}... spend=${run_state.spend_usd:.4f}")

    if args.decision:
        try:
            decision = Decision(args.decision.lower())
        except ValueError:
            print(f"Invalid decision: {args.decision}. Options: {[d.value for d in Decision]}", file=sys.stderr)
            sys.exit(1)
        print(f"Decision: {decision.value}")
        if pending:
            clear_pending_checkpoint(run_dir)
        print(f"Run state restored. Continue pipeline from checkpoint: {run_state.last_checkpoint}")
    else:
        print("No decision provided. State restored but pipeline not continued.")
        print(f"Resume with: python managed_agent_main.py resume --run-dir {run_dir} --decision APPROVE")


def main() -> None:
    parser = argparse.ArgumentParser(description="CRUCIBLE v3.0 Managed Agent entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init", help="Initialize from config file")
    p_init.add_argument("config_path", help="Path to config YAML")

    p_smoke = subparsers.add_parser("smoke", help="Run smoke test")
    p_smoke.add_argument("--mock", action="store_true", default=True,
                         help="Use mock providers (default)")

    p_run = subparsers.add_parser("run", help="Start a pipeline run")
    p_run.add_argument("--track", choices=["A", "B"], required=True)
    p_run.add_argument("--contract", help="Path to contract text file")
    p_run.add_argument("--mock", action="store_true", default=False,
                       help="Use mock providers")
    p_run.add_argument("--auto-approve", action="store_true", default=False,
                       help="Auto-approve all human checkpoints (for testing)")
    p_run.add_argument("--config", help="Path to config YAML")
    p_run.add_argument("--assumptions", help="Path to engagement assumptions JSON file")

    p_resume = subparsers.add_parser("resume", help="Resume from checkpoint")
    p_resume.add_argument("--run-dir", required=True, help="Run directory with checkpoint.json")
    p_resume.add_argument("--decision", help="Decision for pending checkpoint")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "smoke":
        cmd_smoke(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "resume":
        cmd_resume(args)


if __name__ == "__main__":
    main()
