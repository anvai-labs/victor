# Changelog

All notable changes to the `victor-contracts` package are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package adheres to [Semantic Versioning](https://semver.org/). Releases
are tagged independently of `victor-ai` as `sdk-v<version>`. Stability tiers and
the deprecation policy are defined in
[CONTRACT_STABILITY.md](CONTRACT_STABILITY.md).

## [Unreleased]

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
