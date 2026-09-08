# Framework Enhancement Proposals (FEPs)

This directory is the canonical proposal series for changes to Victor's framework, public APIs,
and cross-package contracts. Follow the [FEP process guide](../docs/FEP_PROCESS.md) for scope,
review, decision and implementation requirements; [FEP-0001](fep-0001-fep-process.md) records
the governance proposal.

## Proposal Index

Status values below match proposal frontmatter, reviewed 2026-09-07. A merged design document
can remain Draft; implementation state must not be inferred from its presence in this index.
The [FEP process guide](../docs/FEP_PROCESS.md) explains the contribution workflow.
The older `docs/feps/` 0001/0002 documents form a separate historical series.

| FEP | Title | Status |
|-----|-------|--------|
| [FEP-0001](fep-0001-fep-process.md) | Framework Enhancement Proposal Process | Accepted |
| [FEP-0002](fep-0002-documentation-style-guide.md) | Documentation Style Guide | Draft |
| [FEP-0003](fep-0003-progressive-tool-loading.md) | Progressive Tool Loading with Cost-Based Selection | Draft |
| [FEP-0004](fep-0004-provider-oauth.md) | Provider OAuth Authentication (Subscription-Based Access) | Draft |
| [FEP-0005](fep-0005-policy-engine.md) | Governance Policy Engine (ALLOW/DENY/ASK over tool execution) | Draft |
| [FEP-0006](fep-0006-external-harness-executors.md) | External-Harness Executors | Draft |
| [FEP-0007](fep-0007-unified-agentic-loop.md) | Unified Agentic Loop (single loop, two I/O modes) | Implemented |
| [FEP-0008](fep-0008-evaluation-centric-completion.md) | Evaluation-Centric Completion (calibrated, multi-dimensional, effect-grounded, judge-validated) | Draft |
| [FEP-0009](fep-0009-sdk-tool-contract.md) | SDK Tool Contract — promote tool metadata/traits into victor-contracts | Draft |
| [FEP-0010](fep-0010-shared-protocol-crate.md) | Shared protocol crate for edge + cloud (one contract surface) | Draft |
| [FEP-0011](fep-0011-provider-cache-cost-model.md) | Characterized provider cache cost model (replace boolean cache flags) | Draft |
| [FEP-0012](fep-0012-shipped-edge-classifier.md) | Shipped RL-trained edge classifier (replace the Ollama edge default) | Draft |
| [FEP-0013](fep-0013-shell-safety-policy.md) | Damage-scoped ShellSafetyPolicy — replace the readonly allowlist with a composable, context-aware, RL-seamed policy | Draft |
| [FEP-0014](fep-0014-canonical-validation-metrics-contracts.md) | Canonical validation and metrics contracts (deduplicate ValidationSeverity / ValidationResult / MetricsCollector) | Implemented |
| [FEP-0015](fep-0015-trim-internal-framework-exports.md) | Trim internal-only symbols from the framework public API (step_handlers exports) | Accepted |
| [FEP-0016](fep-0016-wire-initialization-phase-manager.md) | Wire the initialization phase manager (centralize orchestrator runtime init) | Implemented |
| [FEP-0017](fep-0017-prompt-optimization-reward-loop.md) | Close the prompt-optimization reward loop — serve, attribute, and reward evolved prompt candidates | Implemented |
| [FEP-0018](fep-0018-framework-verification-hook.md) | Framework verification hook — verify the agent's work after COMPLETE + retry on failure | Draft |
| [FEP-0019](fep-0019-lsp-integrated-verification.md) | LSP-integrated verification + real-time code feedback | Draft |
| [FEP-0020](fep-0020-ai-gateway-usage-attribution.md) | AI usage gateway: per-user/team attribution + shared-key metering | Draft |
| [FEP-0021](fep-0021-close-the-inner-loop.md) | Close the inner loop — generation + invalidation + reachability for framework self-consistency | Draft |
| [FEP-0022](fep-0022-measurement-driven-self-consistency.md) | Measurement-driven framework self-consistency — the runtime reachability oracle (FEP-0021 Probe A substrate) | Draft |
| [FEP-0023](fep-0023-context-management-subsystem-activation.md) | Activate the context-management subsystem (ledger-keystone, phased + gated) | Draft |
| [FEP-0024](fep-0024-pluggable-code-correction.md) | Pluggable code-correction — entry-point validators, MiddlewareChain routing, argument-kind trait | Draft |
| [FEP-0025](fep-0025-prompt-evolution-as-controlled-experiment.md) | Prompt evolution as a controlled experiment — seeding, evidence, and promotion into source | Draft |
| [FEP-0026](fep-0026-authenticated-control-plane-channel.md) | Authenticated control-plane channel for framework-authored guidance | Review |
| [FEP-0027](fep-0027-settle-stream-usage-on-abort.md) | Settle partial usage when a model stream aborts | Draft |
| [FEP-0028](fep-0028-team-node-durability-contract.md) | Team-Node Durability Contract | Accepted |
| [FEP-0029](fep-0029-single-agent-durable-chat-continuation.md) | Single-Agent Durable Chat Continuation (pause/resume on approval) | Draft |
| [FEP-0030](fep-0030-decoupled-completion-judge.md) | Decoupled Completion Judge (session-model-independent, calibrated, cheap-resident) | Draft |
| [FEP-0031](fep-0031-chat-runtime-inversion.md) | Chat Runtime Inversion — ChatService owns the turn lifecycle | Draft |
| [FEP-0032](fep-0032-interrupt-resume-semantics.md) | Graph Interrupt/Resume Semantics — interrupted signal, resume-at vs completed-at | Draft |
| [FEP-0033](fep-0033-rl-subsystem-relocation.md) | RL/Prompt-Evolution Subsystem Relocation — out of victor/framework, into an application-level victor.rl | Draft |

## Submit or Update a Proposal

1. Start from the [canonical template](fep-0000-template.md).
2. Follow the [submission and review process](../docs/FEP_PROCESS.md).
3. Keep the proposal's frontmatter, title, filename and this index consistent. Quote four-digit
   FEP identifiers in YAML frontmatter, for example `fep: "0031"`.

Do not infer implementation from a merged design document. Preserve its actual status until
the process requirements for changing that status are met.

## Historical Proposal Series

These older documents retain their original numbers in a separate `docs/feps/` series:

- [Legacy 0001: Edge Model](../docs/feps/fep-0001-edge-model.md), distinct from root FEP-0001.
- [Legacy 0002: RL Budget Calibration](../docs/feps/fep-0002-rl-budget-calibration.md), distinct
  from root FEP-0002.

Their banners point at related current decisions and proposals. Do not renumber or interpret
these historical identities as replacements for the canonical entries above.
