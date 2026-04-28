# Crucible Open Source (Basic)

## English

This is the **minimal open-source package** of Crucible.
It includes core orchestration code and mock-mode execution only.

### Current Scope

- This package is currently optimized for **Japanese-language, Japan-law workflows**.
- Track logic, citation normalization defaults, and prompt assumptions are JP-centric by default.

### How to Adapt to Other Jurisdictions

1. Update `config.template.yaml`:
	- set `jurisdiction.primary`
	- set `jurisdiction.language`
	- replace `citation_regex_patterns`
2. Replace prompts in `subagent_prompts.md`:
	- statute names and citation style
	- legal reasoning norms and report language
3. Implement/replace adapters as needed:
	- law-text API adapter (currently e-Gov oriented)
	- local RAG corpus and indexing pipeline for target jurisdiction
4. Rebuild evaluation set and regression tests for the target law domain before production use.

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

### About `subagent_prompts.md`

- Yes, prompt templates are open in this repository.
- This is acceptable for transparent OSS distribution, but prompts are also part of your product IP.
- If you want tighter commercial control, keep only a minimal public prompt set and maintain an internal/private prompt pack.

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

### License

Licensed under Apache License 2.0. See `LICENSE`.

## 中文

这是 Crucible 的**最小开源包**，仅包含框架核心与 mock 模式运行能力。

### 当前适用范围

- 当前默认针对**日语 + 日本法**场景优化。
- 轨道逻辑、引文规范默认值、提示词假设都以日本法为中心。

### 迁移到其他法域怎么改

1. 修改 `config.template.yaml`：
	- 设置 `jurisdiction.primary`
	- 设置 `jurisdiction.language`
	- 替换 `citation_regex_patterns`
2. 改写 `subagent_prompts.md`：
	- 法条名称与引文格式
	- 法律论证习惯与报告语言
3. 按目标法域替换适配器：
	- 法条 API 适配器（当前偏 e-Gov）
	- 本地 RAG 语料与索引流水线
4. 上线前按目标法域重建评测集与回归测试。

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

### 关于 `subagent_prompts.md`

- 是的，当前仓库里已开源提示词模板。
- 这样做有利于透明开源，但提示词本身也是产品 IP。
- 如果你希望更强商业控制，建议只公开最小提示词集合，把增强版提示词放到内部私有仓。

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

### 许可证

本项目采用 Apache License 2.0，详见 `LICENSE`。

## 日本語

これは Crucible の**最小オープンソース版**です。
フレームワークの中核コードと mock 実行機能のみを含みます。

### 現在の適用範囲

- 現在は **日本語 + 日本法** ワークフロー向けに最適化されています。
- トラックロジック、引用正規化の既定値、プロンプト前提は日本法中心です。

### 他法域へ移行する方法

1. `config.template.yaml` を更新:
	- `jurisdiction.primary` を設定
	- `jurisdiction.language` を設定
	- `citation_regex_patterns` を差し替え
2. `subagent_prompts.md` を差し替え:
	- 法令名と引用形式
	- 法的推論スタイルとレポート言語
3. 必要に応じてアダプターを置換:
	- 法令 API アダプター（現状は e-Gov 寄り）
	- 対象法域向け RAG コーパスと索引パイプライン
4. 本番投入前に対象法域向け評価セット/回帰テストを再構築。

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

### `subagent_prompts.md` について

- はい、現状はプロンプトテンプレートを公開しています。
- OSS の透明性には有効ですが、プロンプト自体は製品 IP でもあります。
- 商用コントロールを強めるなら、公開版は最小テンプレートのみとし、拡張版は社内非公開で運用するのが安全です。

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

### ライセンス

本プロジェクトは Apache License 2.0 で提供されます。詳細は `LICENSE` を参照してください。
