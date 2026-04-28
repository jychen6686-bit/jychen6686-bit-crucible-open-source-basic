# Crucible Open Source (Basic)

## English

This is the **minimal open-source package** of Crucible.
It includes core orchestration code and mock-mode execution only.

### Included

- Core package: `crucible/`
- CLI entrypoint: `managed_agent_main.py`
- Compatibility exports: `tools.py`
- Prompt templates: `subagent_prompts.md`
- Config template: `config.template.yaml`
- Build config: `pyproject.toml`
- License: `LICENSE`

### Excluded on purpose

- Any local/private config (`config.yaml`, `.env`)
- Runtime artifacts (`runs/`, caches)
- Proprietary or copyrighted corpus data (`RAG READY/`, `rag_index/`)

### Quick Start

```bash
pip install -e .
python managed_agent_main.py smoke
```

Expected smoke result: `SMOKE TEST PASSED`.

### Mock Run

```bash
python managed_agent_main.py run --track A --mock
```

### Real-provider Mode

```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
cp config.template.yaml config.yaml
python managed_agent_main.py run --track A --contract path/to/contract.txt --assumptions path/to/assumptions.json --config config.yaml
```

## 中文

这是 Crucible 的**最小开源包**，仅包含框架核心与 mock 模式运行能力。

### 包含内容

- 核心包：`crucible/`
- CLI 入口：`managed_agent_main.py`
- 兼容导出：`tools.py`
- 提示词模板：`subagent_prompts.md`
- 配置模板：`config.template.yaml`
- 构建配置：`pyproject.toml`
- 许可证：`LICENSE`

### 有意排除

- 本地/私密配置（`config.yaml`、`.env`）
- 运行产物（`runs/`、缓存）
- 私有或受版权保护的语料与索引（`RAG READY/`、`rag_index/`）

### 快速开始

```bash
pip install -e .
python managed_agent_main.py smoke
```

预期输出：`SMOKE TEST PASSED`。

### Mock 运行

```bash
python managed_agent_main.py run --track A --mock
```

### 真实模型模式

```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
cp config.template.yaml config.yaml
python managed_agent_main.py run --track A --contract path/to/contract.txt --assumptions path/to/assumptions.json --config config.yaml
```

## 日本語

これは Crucible の**最小オープンソース版**です。
フレームワークの中核コードと mock 実行機能のみを含みます。

### 含まれるもの

- コアパッケージ: `crucible/`
- CLI エントリ: `managed_agent_main.py`
- 互換エクスポート: `tools.py`
- プロンプトテンプレート: `subagent_prompts.md`
- 設定テンプレート: `config.template.yaml`
- ビルド設定: `pyproject.toml`
- ライセンス: `LICENSE`

### 意図的に除外したもの

- ローカル/機密設定（`config.yaml`、`.env`）
- 実行生成物（`runs/`、キャッシュ）
- 著作権または私有データ（`RAG READY/`、`rag_index/`）

### クイックスタート

```bash
pip install -e .
python managed_agent_main.py smoke
```

期待される結果: `SMOKE TEST PASSED`。

### Mock 実行

```bash
python managed_agent_main.py run --track A --mock
```

### 実プロバイダーモード

```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
cp config.template.yaml config.yaml
python managed_agent_main.py run --track A --contract path/to/contract.txt --assumptions path/to/assumptions.json --config config.yaml
```
