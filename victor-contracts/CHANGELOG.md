# Changelog

All notable changes to the `victor-contracts` package are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package adheres to [Semantic Versioning](https://semver.org/). Releases
are tagged independently of `victor-ai` as `sdk-v<version>`. Stability tiers and
the deprecation policy are defined in
[CONTRACT_STABILITY.md](CONTRACT_STABILITY.md).

## [Unreleased]

## [0.15.0] - release candidate

FEP-0042 Phase 3 (backlog F-019): removes the surfaces deprecated in 0.14.0.

### Upgrade notes

- Published first-party verticals **< 0.14.0** do not work with 0.15.0: victor-coding,
  victor-devops and victor-research 0.13.0 import `get_default_persona_registry` while loading
  `teams.personas`, and victor-coding 0.13.0 imports `chain_runtime` at module load, so they
  fail at import. Upgrade every first-party vertical to **>= 0.14.0** (they use none of the
  removed names). victor-ai >= 0.23.0 refuses first-party verticals < 0.14.0 before importing
  them, with a message naming the vertical and the required version.
- victor-ai **0.22.0** has no such gate: with first-party verticals < 0.14.0 installed, their
  activation hooks fail with `ImportError`. Upgrade the verticals (or victor-ai to >= 0.23.0,
  which explains the problem instead).
- victor-ai **<= 0.21.0** imports the removed `set_default_persona_registry` eagerly, so
  `import victor.framework.multi_agent` fails against 0.15.0; upgrade victor-ai to
  >= 0.23.0. (victor-ai 0.22.0 keeps working: the private helper its deprecated
  `FrameworkPersonaProvider` imports is retained as a no-op.)
- Release order: victor-contracts 0.15.0 is published only after victor-ai 0.23.0, so a fresh
  `pip install victor-ai` resolves a host that carries the gate.
- Third-party code that calls `register_vertical_personas(personas, policy)` without a
  provider now gets `TypeError`; pass the host persona registry as `provider`, or expose
  personas through the vertical's `get_personas()` hook instead.

### Removed

Each was deprecated with warnings in 0.13.0 or 0.14.0.

- `victor_contracts.chain_runtime` (`ChainRegistry`, `get_chain_registry`), the inert shim
  since 0.14.0. Importing it raises `ModuleNotFoundError`. Deliver chains through the
  vertical's `get_chains()` / `get_composed_chains()` hook; the host registers them in its
  flat `victor.framework.chain_registry` on activation.
- The default-registry fallback of `verticals.register_vertical_personas`: `provider` is now a
  required argument (typed `PersonaRegistryProtocol`), passing `None` raises `TypeError`, and
  the return type is `int` (was `Optional[int]`). Expose personas through `get_personas()`;
  a host that registers personas itself passes its registry as `provider`.
- `verticals.RegistrationPolicy.provider_getter` and `auto_register` (never read). Passing
  either now raises `TypeError`; drop the argument.
- `registries.set_default_persona_registry()` / `get_default_persona_registry()` and their
  top-level `victor_contracts` re-exports and `__all__` entries (plus the private
  `_install_default_persona_registry` / `_current_default_persona_registry` helpers). Use
  `get_personas()` in verticals and pass `provider=` in hosts.
  `set_default_team_registry()` / `get_default_team_registry()` are unaffected.

### Changed

- (internal, `testing/`) `assert_import_registers_no_personas` no longer installs a recording
  default persona registry; it replaces `register_vertical_personas` (on both
  `victor_contracts.verticals` and `victor_contracts.verticals.persona_kernel`) with a recorder
  before the probed imports and fails on any import-time call, as well as on SDK
  `DeprecationWarning`s.

## [0.14.0] - 2026-10-30

FEP-0042 Phase 2a (backlog F-019).

### Upgrade notes

- Published first-party verticals **0.13.0** keep working with 0.14.0 but emit deprecation
  warnings at import: victor-coding/devops/research call `get_default_persona_registry()` while
  loading `teams.personas`, and victor-coding registers its chains through the inert
  `chain_runtime` shim. Upgrade the verticals to **>= 0.14.0**, which use neither. Running with
  `-W error::DeprecationWarning` turns these into import errors.
- Published victor-dataanalysis 0.13.0's `register_data_analysis_personas()` (never called by
  the host) now logs a registration failure and returns 0, because
  `get_runtime_persona_provider` is gone.

### Removed

- `victor_contracts.multi_agent.get_runtime_persona_provider()` (and its
  `__all__` entry), deprecated in 0.13.0. Deliver personas through the
  vertical's `get_personas()` hook; the host registers them on activation and
  reads them from its flat `victor.framework.persona_registry`.

### Changed

- `victor_contracts.chain_runtime` (`ChainRegistry`, `get_chain_registry`),
  deprecated in 0.13.0, is now an **inert, host-free compatibility shim**
  instead of being removed: published victor-coding 0.13.0 registers chains
  through it at import, unguarded. It no longer imports
  `victor.framework.chains`; `ChainRegistry.register_chain(...)` accepts and
  discards registrations, lookups return `None` / `[]`, and
  `get_chain_registry()` returns a process-wide instance. Accessing either name
  still emits `DeprecationWarning`; removable no earlier than 0.15.0. Deliver
  chains through the vertical's `get_chains()` / `get_composed_chains()` hook.

### Deprecated

Each emits `DeprecationWarning` (attributed to the caller) and keeps working;
removable no earlier than 0.15.0. Nothing in the Victor host installs a default
persona registry, so these paths registered nothing.

- `verticals.register_vertical_personas(personas, policy)` without `provider=`
  (the default-registry fallback). In 0.15.0 `provider` becomes required.
  Verticals expose personas through `get_personas()`; a host that registers
  personas itself passes `provider=` explicitly (that path does not warn).
- `verticals.RegistrationPolicy.provider_getter` and `auto_register`: never
  read. Warns only when set to a non-default value; drop the argument.
- `registries.set_default_persona_registry()` /
  `get_default_persona_registry()` (also re-exported from `victor_contracts`).
  `set_default_team_registry()` / `get_default_team_registry()` are unaffected.

### Added

- (internal, `testing/`) `victor_contracts.testing.personas`:
  `assert_import_registers_no_personas(*modules)` imports modules in a fresh
  interpreter with `DeprecationWarning` as an error and fails on any import-time
  persona registration; `assert_explicit_persona_registration(register, personas,
  vertical)` checks an explicit-provider helper; `RecordingPersonaRegistry` is the
  recording double both use.

## [0.13.0] - 2026-10-30

### Deprecated

- `chain_runtime` (`ChainRegistry`, `get_chain_registry`): accessing either name
  emits `DeprecationWarning`; removable no earlier than 0.14.0. The versioned
  chain registry it bridges to has no runtime readers. Deliver chains through
  the vertical's `get_chains()` / `get_composed_chains()` hook; the host
  registers them in its chain registry on activation.
- `multi_agent.get_runtime_persona_provider()`: emits `DeprecationWarning`;
  removable no earlier than 0.14.0. The host persona provider has no runtime
  readers. Deliver personas through the vertical's `get_personas()` hook.
- The host's flat chain/persona registries (`victor.framework.chain_registry`,
  `victor.framework.persona_registry`) are unaffected.

### Added

- (internal) `_lazy.make_lazy_getattr(..., deprecated=...)`: bridge names listed in the
  optional mapping emit `DeprecationWarning` (attributed to the caller) before
  resolving.

## [0.12.0] - 2026-10-30

### Removed

- The six runtime bridge modules deprecated in 0.9.0 are removed; importing
  them raises `ModuleNotFoundError`. None had consumers in the Victor monorepo.
  Replacements:
  - `agent_spec_runtime` — use `victor.agent.specs.models` in host-runtime code
  - `graph_runtime` — use `victor.framework.graph` (`StateGraph`, `END`)
  - `handler_runtime` — use `victor_contracts.workflow_runtime`
    (`register_compute_handler`) or `victor.framework.handler_registry`
  - `subagent_runtime` — use `victor.agent.subagents.protocols`
  - `tool_runtime` — use `victor.framework.tools` (`RuntimeToolSet`)
  - `workflow_executor_runtime` — use `victor_contracts.workflow_runtime`
- The internal `_deprecation` module (the deprecated-bridge registry) is removed
  with them.

### Added

- `victor_contracts.chunking`: a dependency-free chunking core —
  `ChunkingConfig` (same fields and defaults as the framework and RAG copies),
  `TextChunker` (`chunk_text`, `chunk_markdown`, `chunk_code`,
  `estimate_chunks`) and the immutable `ChunkSpan(text, start_char, end_char)`.
  Spans are byte-for-byte equal to the existing framework and RAG chunkers,
  which can now delegate to it. Unlike those copies, `ChunkingConfig` rejects
  settings that made chunking loop forever (`chunk_size <= 0`, negative
  `min_chunk_size`, `chunk_overlap` outside `[0, chunk_size)` or greater than
  `min_chunk_size`) with `ValueError`.
- Internal `victor_contracts._lazy.make_lazy_getattr`, the shared PEP 562
  `__getattr__` factory now used by every bridge module. Unknown attributes
  raise `AttributeError("module '<bridge>' has no attribute '<name>'")`
  uniformly.

## [0.11.0] - 2026-10-29

### Added

- `TeamMemberSpec` gains the per-member execution overrides the runtime already
  supported: `provider`, `model`, `temperature`, `reasoning_effort` and
  `formation_role`. Vertical team definitions can now declare heterogeneous
  (cross-vendor, per-role sampling) teams; `to_team_member()` forwards them.
  `TeamSpec.to_dict()`/`from_dict()` round-trip them.
- `PIISeverity.to_severity()` maps onto the host's unified `SeverityLevel`,
  matching the other severity enums (requires the Victor runtime host).

### Changed

- `victor-ai` now re-exports `TeamMemberSpec`, `ROLE_MAPPING` and the PII
  module from the SDK instead of maintaining diverged copies. Runtime-built
  specs therefore gain the SDK behaviour: `allowed_tools` are canonicalized,
  `can_delegate` is an explicit field, and the `supervisor`, `manager`, `lead`
  and `coordinator` roles map to PLANNER (previously EXECUTOR at runtime).
  **Behaviour change for runtime API users:** a member with one of those roles
  and no explicit `allowed_tools` now gets the planner's default (read-only)
  tool set and a 15-call budget instead of the executor's; pass `role="executor"`
  or explicit `allowed_tools` for a coordinating member that must write.

## [0.10.0] - 2026-10-17

### Changed

- Require Python 3.12 or newer. This minor release raises the interpreter floor
  from the published 0.9.2 package; existing 0.9.2 artifacts remain unchanged.

### Fixed

- Runtime `MemoryConfig` and `TeamAgentCategory` now re-export their canonical SDK
  definitions, preserving identity and eliminating incompatible duplicate types.
- `processing_runtime` resolves `ChunkInfo` and `TextChunkerProtocol` against their
  actual host owner. The deprecated handler decorator keeps the decorated class
  type; the bridge no longer advertises `sync_handlers_with_executor`, which had
  no host implementation and failed on access.

### Added

- MyPy 2.4.0 developer pin aligned with core and first-party verticals; manifest
  extraction narrows from `object` without a checker-version-specific ignore.
- SDK and legacy-root `py.typed` wheel markers, strict SDK source checks and an
  installed-wheel consumer check with positive and negative typing examples.
  Runtime legacy submodule aliases remain available; static consumers should use
  `victor_contracts.*` (see the contract stability policy).

## [0.9.0] - 2026-08-01

### Added

- `CONTRACT_STABILITY.md`: contract maturity matrix (Stable / Stable bridge /
  Deprecated / Internal) and the deprecation policy for the package.
- This changelog.

### Deprecated

- The six consumer-less runtime bridge modules now emit `DeprecationWarning`
  on attribute access and will be removed no earlier than 0.10.0:
  - `agent_spec_runtime` — use `victor.agent.specs.models` in host-runtime code
  - `graph_runtime` — use `victor.framework.graph` (`StateGraph`, `END`)
  - `handler_runtime` — use `victor_contracts.workflow_runtime`
    (`register_compute_handler`) or `victor.framework.handler_registry`
  - `subagent_runtime` — use `victor.agent.subagents.protocols`
  - `tool_runtime` — use `victor.framework.tools` (`RuntimeToolSet`)
  - `workflow_executor_runtime` — use `victor_contracts.workflow_runtime`

## [0.8.0] - 2026-07-21

In-tree version bump (coordinated with victor-ai 0.8.0); not yet published to
PyPI as of this changelog's creation.

### Added

- `RuntimeEvaluationFeedback` shared runtime-evaluation feedback contract
  (`runtime_evaluation.py`), decoupling the `victor.evaluation` producer from
  `victor.framework` / `victor.agent` consumers.
- `ArgumentKind` enum and per-argument `argument_kinds` on the SDK Tool
  Contract (`tools.py`), supporting FEP-0024 pluggable code correction.

## [0.7.3] - 2026-07-17

Published to PyPI. Coordinated release with victor-ai 0.7.6.

### Added

- Team coordination protocol contracts (`framework/protocols/teams.py`).
- Team schema and multi-agent contract extensions (`team_schema.py`,
  `multi_agent.py`) for the supervisor-based teams refactor.

## [0.7.2] - 2026-07-04

First public PyPI release.

### Added

- Standalone importability without the `victor-ai` runtime installed.
- SDK Tool Contract (`ToolContract` plus trait enums, FEP-0009) and the
  contract-first vertical authoring surface (`VerticalBase`,
  `VerticalDefinition`, `ToolNames`, `CapabilityIds`).
- `victor-contracts check <package>` CLI for validating vertical packages.

---

Versions 0.7.1 and earlier existed only inside the monorepo (including the
`victor_sdk` → `victor_contracts` package rename) and were never published to
PyPI.
