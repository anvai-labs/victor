# Changelog

All notable changes to the `victor-contracts` package are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package adheres to [Semantic Versioning](https://semver.org/). Releases
are tagged independently of `victor-ai` as `sdk-v<version>`. Stability tiers and
the deprecation policy are defined in
[CONTRACT_STABILITY.md](CONTRACT_STABILITY.md).

## [Unreleased]

## [0.11.0] - release candidate

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
