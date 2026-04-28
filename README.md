# Crucible Open Source (Basic)

This is the **minimal open-source package** of Crucible.
It includes core orchestration code and mock-mode execution only.

## Included

- Core package: `crucible/`
- CLI entrypoint: `managed_agent_main.py`
- Compatibility exports: `tools.py`
- Prompt templates: `subagent_prompts.md`
- Config template: `config.template.yaml`
- Build config: `pyproject.toml`
- License: `LICENSE`

## Excluded on purpose

- Any local/private config (`config.yaml`, `.env`)
- Runtime artifacts (`runs/`, caches)
- Proprietary or copyrighted corpus data (`RAG READY/`, `rag_index/`)

## Quick Start

```bash
pip install -e .
python managed_agent_main.py smoke
```

Expected smoke result: `SMOKE TEST PASSED`.

## Mock Run

```bash
python managed_agent_main.py run --track A --mock
```

## Real-provider Mode

If you want to run with real providers, set API keys in your environment and provide a filled config file:

```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
cp config.template.yaml config.yaml
python managed_agent_main.py run --track A --contract path/to/contract.txt --assumptions path/to/assumptions.json --config config.yaml
```

## Notes

- This package is intended for open-source distribution of the framework layer.
- Data ingestion, private corpora, and built RAG indexes are intentionally not included.
