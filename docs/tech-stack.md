# Victor Tech Stack

> Canonical reference for technology choices. System ownership and import rules live in
> [Architecture](architecture.md); work status lives in the [Roadmap](roadmap.md).

**Version**: {{ victor_version }} | **Last reviewed**: 2026-09-07

## Technology stack

Dependency ranges below come from the root `pyproject.toml`; they are install requirements,
not a claim that these are the newest upstream releases. Optional extras remain optional.

| Concern | Technology / declared dependency | Implementation |
| --- | --- | --- |
| Runtime | Python 3.11+, asyncio | `victor/` |
| Contracts | `victor-contracts>=0.8.0,<1.0`; separately released SDK | `victor-contracts/` |
| Models and settings | Pydantic and pydantic-settings >=2.0 | `victor/config/` |
| Typed provider transport | `sandhi-gateway==0.5.0` | `victor/providers/` |
| Python HTTP interfaces | `httpx>=0.27`, `aiohttp>=3.14.3` | Provider and integration adapters |
| CLI | `typer>=0.15,<0.26`, `rich>=13.7`, prompt-toolkit | `victor/ui/cli.py`, `victor/ui/` |
| Interactive TUI | `textual>=0.89` | `victor/ui/tui/` |
| Token counting | `tiktoken>=0.7`, optional native `BpeTokenizer` | `victor/processing/`, `rust/` |
| YAML and Git | `pyyaml>=6.0`, `gitpython>=3.1.58` | Workflow loading and tools |
| Code analysis | `tree-sitter>=0.23`, optional language grammars | `victor-codegraph/`, `victor/core/` |
| Numeric support | `numpy>=1.24,<2.3` | Processing and retrieval helpers |
| Semantic index | Optional `lancedb>=0.6.0` | `victor/storage/` |
| ProximaDB integration | Optional `proximadb>=0.3,<0.4` | `victor/storage/graph/`, `victor/storage/vector_stores/` |

### Providers and surfaces

The provider registry and [provider comparison](reference/providers-comparison.md) describe
capabilities; [provider support tiers](https://github.com/anvai-labs/victor/blob/develop/SUPPORT.md#provider-support-tiers) describe support
commitments. The transport boundary is Sandhi for admitted providers, with Victor-specific
adapters for other execution models. Installed Anthropic/OpenAI SDK dependencies do not imply
that every adapter directly owns HTTP transport.

| Surface | Technology | Location |
| --- | --- | --- |
| CLI and REPL | Typer, Rich, prompt-toolkit | `victor/ui/` |
| Live TUI | Textual | `victor/ui/tui/` |
| HTTP API | Optional FastAPI / Uvicorn | `victor/integrations/api/fastapi_server.py` |
| MCP | FastMCP integration | `victor/integrations/mcp/` |
| Editor extension | TypeScript | `vscode-victor/` |

<a id="native-extensions-rust"></a>

### Native extensions

The Rust workspace contains `victor-protocol`, `victor-state`, `victor-tools`,
`victor-edge`, and the `victor_native` Python bindings. Native processing paths provide
Python fallbacks; the required native-parity job checks matching behavior with the built extension.

Use the [canonical native build instructions](development/setup.md#native-extension-build).
The [architecture native section](architecture.md#rust-native-extensions) describes ownership.

<a id="language-and-runtime"></a>

## Language and verification tools

| Concern | Tool / policy |
| --- | --- |
| Formatting | Black, 100-character Python line length |
| Linting | Ruff; repository configuration is in `pyproject.toml` |
| Type checking | Mypy with an advisory broad job and required strict checks for selected scopes |
| Tests | pytest, pytest-asyncio, respx for HTTP mocks |
| CI | `ci-fast.yml` aggregate on development PRs; main-targeting promotion tests use 3 Python versions × 12 shards |
| Docs | MkDocs Material, mkdocstrings, Mermaid diagrams |
| Native build | maturin / Cargo |
| Task entry points | Make and pre-commit |

Commands and environment setup belong in [Development Setup](development/setup.md),
including the [documentation build](development/setup.md#documentation-build).
See [PR Workflow](development/PR_WORKFLOW.md) for the authoritative gate and branch policy.

## Dependency map

The canonical [layer architecture](architecture.md#layer-architecture) and
[layer rules](architecture.md#layer-rules) describe dependencies and guard coverage.
Vertical definition files use `victor_contracts`; runtime extension allowances and outstanding
boundary-audit work are described by the [SDK boundary](architecture/CONTRACTS_BOUNDARY.md).

## Infrastructure

Database scope, paths, and storage ownership are documented once in
[Database Architecture](architecture.md#database-architecture). SQLite stores global and
project state; undo history has its own database to avoid indexer write-lock contention.
LanceDB and the optional ProximaDB backends serve different retrieval configurations.
The [ProximaDB backend design](architecture/proximadb-codegraph-backend.md) distinguishes
implemented opt-in capabilities from remaining default-graduation work.

<a id="active-items"></a>
<a id="tech-debt-timeline"></a>

## Technical debt register

The complete register, including TD namespaces, current statuses and historical planning
notes, moved to the [canonical roadmap register](roadmap.md#technical-debt-register).
Existing links to this section continue to resolve.

## Resolved debt

See [resolved debt](roadmap.md#resolved-debt) in the same register. Historical resolutions
remain visible alongside reopened items; they are not duplicated here.

## Architectural constraints

See [architecture layer rules](architecture.md#layer-rules) and the linked guard tests.
The canonical gate descriptions live in [PR Workflow](development/PR_WORKFLOW.md).

### Build verification

Follow [Development Setup](development/setup.md) and the
[testing guide](development/testing.md) for checks appropriate to the affected component.

## Architecture and delivery diagrams

- [System layering and guards](architecture.md#system-overview)
- [Unified streaming loop](architecture.md#agenticloop)
- [Single workflow engine](architecture.md#workflow-engine)
- [SQLite worker and workflow persistence](architecture.md#database-architecture)
- [Release trains](development/releasing/publishing.md#release-process-overview)
- [CI gates](development/PR_WORKFLOW.md#ci-gate-map)

Proposed runtime inversion, graph resume and RL relocation diagrams live in their FEPs,
linked from [planned runtime changes](architecture.md#planned-runtime-and-learning-changes).
