from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crucible.enums import Decision
from crucible.schemas import PauseForHumanInput, PauseForHumanResult


def pause_for_human_managed_agent(
    input: PauseForHumanInput,
    run_state: Any,
    checkpoint_dir: Path,
) -> PauseForHumanResult:
    """
    Checkpoint emulation for Managed Agents.

    Writes pending_checkpoint.json + context_summary.md then returns ABORT
    so the current process halts cleanly. The entrypoint detects these files
    on resume and prompts the user for a decision.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Persist checkpoint.json first
    run_state.persist(checkpoint_dir)

    # Write pending_checkpoint.json
    pending = {
        "checkpoint_id": input.checkpoint_id.value,
        "summary": input.summary,
        "artifact_refs": input.artifact_refs,
        "decision_options": [d.value for d in input.decision_options],
        "edits_allowed_on": input.edits_allowed_on,
        "persisted_at_iso": datetime.now(timezone.utc).isoformat(),
    }
    pending_path = checkpoint_dir / "pending_checkpoint.json"
    with open(pending_path, "w", encoding="utf-8") as f:
        json.dump(pending, f, indent=2, ensure_ascii=False)

    # Write context_summary.md for Managed Agent context injection on resume
    snap = run_state.snapshot()
    artifacts_list = "\n".join(
        f"  - {name}" for name in run_state.artifacts
    ) or "  (none yet)"
    decision_opts = ", ".join(d.value for d in input.decision_options)

    summary_md = f"""# CRUCIBLE Run Context Summary

**Run ID:** {run_state.run_id}
**Track:** {run_state.track}
**Checkpoint:** {input.checkpoint_id.value}
**Persisted at:** {datetime.now(timezone.utc).isoformat()}

## Current Status

- Spend so far: ${snap.spend_usd:.4f} / ${snap.cap_usd:.2f} cap
- Elapsed: {snap.elapsed_seconds:.0f}s / {snap.cap_wall_seconds}s cap
- Subagent invocations: {snap.subagent_invocations}
- Last subagent: {snap.last_subagent or "(none)"}
- TTL mode degraded: {snap.ttl_mode_degraded}

## Artifacts Produced

{artifacts_list}

## Pending Decision

{input.summary}

**Decision options:** {decision_opts}

## How to Resume

```
python managed_agent_main.py resume --run-dir {checkpoint_dir} --decision <DECISION>
```

Replace `<DECISION>` with one of: {decision_opts}
"""

    summary_path = checkpoint_dir / "context_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    # Return ABORT so the current pipeline halts
    return PauseForHumanResult(
        decision=Decision.ABORT,
        edits={},
        budget_extension_usd=None,
        note="Checkpoint persisted. Resume with managed_agent_main.py resume.",
    )


def load_pending_checkpoint(checkpoint_dir: Path) -> dict | None:
    """Load pending_checkpoint.json if it exists. Returns None if no pending checkpoint."""
    path = Path(checkpoint_dir) / "pending_checkpoint.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_context_summary(checkpoint_dir: Path) -> str | None:
    """Load context_summary.md for injection into Managed Agent context on resume."""
    path = Path(checkpoint_dir) / "context_summary.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def clear_pending_checkpoint(checkpoint_dir: Path) -> None:
    """Remove pending_checkpoint.json after the human has provided their decision."""
    path = Path(checkpoint_dir) / "pending_checkpoint.json"
    if path.exists():
        path.unlink()
