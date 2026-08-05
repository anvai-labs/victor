---
fep: 0006
title: "External-Harness Executors"
type: Standards Track
status: Draft
created: 2026-06-15
modified: 2026-06-15
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0006
---

# FEP-0006: External-Harness Executors

## Table of Contents

1. [Summary](#summary)
2. [Motivation](#motivation)
3. [Proposed Change](#proposed-change)
4. [Benefits](#benefits)
5. [Drawbacks and Alternatives](#drawbacks-and-alternatives)
6. [Unresolved Questions](#unresolved-questions)
7. [Implementation Plan](#implementation-plan)
8. [Migration Path](#migration-path)
9. [Compatibility](#compatibility)
10. [References](#references)
11. [Review Process](#review-process)
12. [Acceptance Criteria](#acceptance-criteria)

---

## Summary

Victor can drive 24 LLM providers, but every provider is modelled as a single
LLM round-trip: `BaseProvider.chat()` takes messages and returns one
`CompletionResponse`, and Victor's own `AgenticLoop`/`TurnExecutor` owns the
tool-calling loop. Several high-value targets — Claude Code, OpenAI Codex, and
OpenAI-Agents-style endpoints — are not single round-trips: they are *complete
agents* that run their own internal plan→tool→observe loop and return a final
result. This is the core capability of meta-harnesses like Omnigent, and Victor
cannot express it today.

This FEP proposes an **External-Harness Executor**: a way for Victor to delegate
a task (or a turn) to an external agent and incorporate its work. It defines two
delivery levels — a cheap **black-box adapter** (the external agent runs opaquely;
Victor sees only the final output) and a **streamed-step integration** (the
external agent's inner steps — reasoning, tool calls, tool results — are surfaced
to Victor as normalized events so observability, and eventually governance
(FEP-0005) and sandboxing, can apply). It surfaces the central design decision —
*where* the external agent plugs in (the provider seam vs. a new executor
strategy) — and recommends a phased rollout, opt-in and off by default.

Affected: framework runtime (`victor/agent`, `victor/providers`), and a new
public executor contract. No `victor-contracts` change in Phase 1.

## Motivation

### Problem Statement

Victor's runtime assumes **Victor owns the agentic loop**. The provider layer is
a thin "one prompt → one completion" boundary:

- `BaseProvider.chat()` (`victor/providers/base.py:898`) is abstract and returns
  a single `CompletionResponse`; `stream()` yields `StreamChunk`s for *one*
  model turn.
- `TurnExecutor` (`victor/agent/services/turn_execution_runtime.py:165`) is the
  single-turn primitive under `AgenticLoop`; it calls the injected
  `execution_provider.execute_turn(...)` (the legacy `ExecutionProvider`
  protocol, `victor/agent/coordinators/protocols.py:574`) once per iteration and
  Victor runs any tools the model requested.
- Providers are built through `ManagedProviderFactory.create(provider_name,
  model, ...)` (`victor/providers/factory.py:525`).

An **external harness** inverts this. Claude Code / Codex / an OpenAI-Agents
endpoint each runs its *own* loop: it plans, calls *its own* tools, observes
results, and iterates — possibly for minutes — before returning. There is no
clean "one round-trip" to map onto `chat()`. If we naively wrap such an agent as
a provider, two things break:

1. **Loop ownership conflict.** Victor's `AgenticLoop` wants to drive iterations,
   but the external agent already drove them. The outer loop becomes a no-op
   wrapper around a single giant "turn."
2. **Black-boxed inner steps.** The external agent's internal tool calls,
   file edits, and shell commands never pass through Victor's `ToolPipeline`,
   so they are invisible to observability and **ungoverned** by the FEP-0005
   policy engine and the sandbox isolation layer. From Victor's perspective a
   minute of autonomous file mutation looks like one opaque completion.

Real-world use cases:

- "Hand this refactor to Codex and integrate the result" (delegation).
- Cross-harness review teams ("let Claude Code implement, have Victor review")
  — the heterogeneous-teams work already lets members run on different
  providers; an external-agent member is the natural extension.
- Benchmarking Victor's orchestration against a raw external agent on the same
  task.

### Goals

1. Let Victor delegate a task/turn to an external agent (Claude Code, Codex,
   OpenAI-Agents endpoint) and incorporate its final output.
2. Define a clear, public **executor contract** that does not pretend an
   external agent is a single-round-trip provider.
3. Provide a cheap first increment (black-box) and a path to full
   **streamed-step** integration where inner steps are surfaced to Victor.
4. Make the surfaced inner steps eventually governable (FEP-0005) and
   sandboxable, and observable from day one of streamed-step mode.
5. Opt-in, off by default, fail-safe; no `victor-contracts` change in Phase 1;
   no new parallel abstraction layer inside `victor/agent`.

### Non-Goals

- Reimplementing the external agents themselves, or shipping their binaries.
- Governing the inner steps of a **black-box** external agent (impossible by
  definition — that is precisely what streamed-step mode is for).
- Multi-agent *orchestration* changes beyond exposing an external executor as a
  usable team member / execution strategy.
- Authentication/credential management for external harnesses beyond reusing the
  existing provider credential plumbing.

## Proposed Change

### High-Level Design

Introduce an **executor** abstraction distinct from a *provider*. A provider is
"one model turn"; an executor "runs a task to completion." External harnesses are
executors.

```python
from victor.framework.executors import ExternalAgentExecutor, ExternalAgentEvent

# Conceptual contract (framework-internal in Phase 1; promoted to a public
# contract only when stabilized).
class ExternalAgentExecutor(Protocol):
    name: str

    async def run(
        self,
        task: str,
        *,
        context: ExecutorContext,
    ) -> AsyncIterator[ExternalAgentEvent]:
        """Drive the external agent to completion, yielding normalized events.

        Events: AGENT_THINKING, TOOL_CALL, TOOL_RESULT, CONTENT, STEP, ERROR,
        FINAL. A black-box executor emits only CONTENT + FINAL.
        """
        ...
```

Two delivery levels share this contract:

- **Black-box adapter** — `run()` shells out to (or HTTP-calls) the external
  harness once, waits for completion, and yields a single `CONTENT`/`FINAL`
  event with the final output. Inner steps are not surfaced.
- **Streamed-step integration** — `run()` consumes the harness's native event
  stream (Claude Code `--output-format stream-json`, Codex event stream, OpenAI
  Agents SDK run events) and maps each into an `ExternalAgentEvent`. Victor's
  observability bus records them; later phases route surfaced `TOOL_CALL` events
  through the policy engine / sandbox before they are allowed to proceed (only
  possible for harnesses that support an approval/interactive protocol).

### Detailed Specification

#### Where it plugs in (the central decision)

Two integration points are viable; this FEP recommends supporting **both, in
order**, because they serve different needs:

**(A) Provider-seam black-box adapter (Phase 1).** Implement the legacy
`ExecutionProvider.execute_turn(...)` (or a thin `BaseProvider` subclass) that
runs the external agent and returns its final output as one `CompletionResponse`.
Built via `ManagedProviderFactory.create("external:<harness>", model, ...)`.
- Pro: minimal — TurnExecutor/AgenticLoop are unchanged; the external agent is
  "one turn that finishes the task."
- Con: Victor's loop is vestigial; inner steps are black-boxed; this is honest
  only as a delegation primitive, not a governed execution.

**(B) Executor strategy that bypasses AgenticLoop (Phase 2).** Add an execution
strategy at the `ChatService`/`TurnExecutor` selection seam so that, when the
target is an external executor, Victor does **not** run `AgenticLoop` at all —
it consumes the executor's event stream directly and renders it to the user
(reusing the streaming chunk path) while recording events. This is the correct
home for streamed-step mode: the external agent legitimately owns the loop, and
Victor is the host/observer/governor, not the driver.

The decision to surface for review: **do we ship (A) alone first (fast, limited),
or invest directly in (B)?** Recommendation: ship (A) behind a flag as a
delegation primitive, then build (B) for the governed/observable use case. (A) is
~2–3 hours; (B) is the 1–2 week effort.

#### Event model

A normalized `ExternalAgentEvent` (mirroring `victor/framework/events.py` event
kinds so existing observability consumers work): `kind`
(THINKING/TOOL_CALL/TOOL_RESULT/CONTENT/STEP/ERROR/FINAL), `text`,
`tool_name`/`arguments`/`result` (when applicable), `raw` (harness-native
payload), and correlation/trace IDs. Black-box mode emits only CONTENT + FINAL.

#### Adapters

One adapter per harness, each isolating the harness's transport and event format:
- **Claude Code** — subprocess, `claude -p --output-format stream-json` (or the
  Agent SDK); parse the JSON event stream.
- **Codex** — subprocess/CLI or API; parse its event stream.
- **OpenAI-Agents endpoint** — HTTP; consume run/step events. OpenAI-compatible
  patterns can reuse `victor/providers/openai_compat.py`.

Adapters degrade gracefully: if streamed-step events are unavailable, fall back
to black-box (final output only) with a one-time warning.

#### Governance / sandbox interaction (FEP-0005, sandbox isolation)

- Black-box mode: inner steps are **ungoverned and unsandboxed** — this MUST be
  surfaced to the user (a one-time warning + an explicit setting acknowledging
  it), because it is a real escape hatch around the policy engine and sandbox.
- Streamed-step mode: surfaced `TOOL_CALL` events MAY be routed through the
  policy engine before the host acknowledges them, *for harnesses that expose an
  interactive approval protocol*. Where the harness cannot pause for approval,
  governance is observe-only (audit), not enforce. This nuance is per-adapter.

#### Configuration

```yaml
external_executors:
  enabled: false               # master switch (off by default)
  harness: "claude_code"       # claude_code | codex | openai_agents
  mode: "black_box"            # black_box | streamed_steps
  command: null                # override binary/endpoint
  timeout_seconds: 1800
  acknowledge_ungoverned: false  # required True to use black_box (escape hatch)
```

Gated additionally by a feature flag `USE_EXTERNAL_EXECUTORS`
(`VICTOR_USE_EXTERNAL_EXECUTORS=true`), opt-in.

### API Changes

- New framework package `victor/framework/executors/` with the
  `ExternalAgentExecutor` protocol and `ExternalAgentEvent` type
  (framework-internal in Phase 1).
- Phase 1 reuses the existing `ExecutionProvider.execute_turn` seam (no new
  public API) for the black-box adapter.
- Phase 2 adds an execution-strategy selection point at the
  `ChatService`/`TurnExecutor` boundary (internal).
- A public `victor-contracts` executor contract is proposed **only** if/when an
  external package needs to ship its own harness adapter (deferred; needs its
  own contracts semver bump).

### Dependencies

None mandatory in core. Per-adapter optional dependencies (the external harness
binaries/SDKs) must degrade gracefully when absent, matching the optional-dep
policy.

## Benefits

### For Framework Users
- Delegate whole tasks to best-in-class external agents from inside Victor,
  without leaving the Victor session or losing its session/observability context.
- Cross-harness review/benchmark flows (implement with one agent, review with
  Victor) — composes directly with the heterogeneous-teams work, so an external
  agent can be a team member alongside native providers.

### For Vertical Developers
- A documented executor seam to add domain-specific external agents without
  faking a single-round-trip provider, with one isolated adapter per harness so
  format drift is contained.

### For the Ecosystem
- Closes Victor's biggest gap vs. meta-harnesses (Omnigent's core superpower):
  orchestrating *other agents*, not just models — while keeping the governance
  and sandbox story honest about what black-box vs. streamed-step can enforce.

## Drawbacks and Alternatives

### Drawbacks
- **Governance escape hatch (black-box).** Inner steps bypass FEP-0005 and the
  sandbox. Mitigation: require explicit `acknowledge_ungoverned: true`, emit a
  persistent warning, and recommend streamed-step mode for governed use.
- **Adapter maintenance.** Each harness has its own evolving event format.
  Mitigation: one isolated adapter per harness; fall back to black-box on
  format drift.
- **Latency/observability mismatch.** A black-box run can take minutes with no
  intermediate signal. Mitigation: stream a heartbeat; prefer streamed-step.

### Alternatives Considered

1. **Wrap every external agent as a normal `BaseProvider`.**
   - Pros: zero new abstraction.
   - Cons: lies about the contract (one round-trip), makes `AgenticLoop`
     vestigial, hides the governance gap. Rejected — the dishonesty is the bug.

2. **Only ever do streamed-step integration (skip black-box).**
   - Pros: always governed/observable.
   - Cons: large up-front cost; some harnesses don't expose pausable step
     streams; blocks the cheap, useful delegation case. Rejected as the *first*
     step; it is Phase 2.

3. **External MCP server per harness.**
   - Pros: reuses MCP plumbing.
   - Cons: MCP models tools, not whole agents driving their own loop; the
     loop-ownership problem remains. Rejected as the primary model (may still be
     a transport for some adapters).

## Unresolved Questions

- **Q1 — plug-in point:** Ship (A) provider-seam black-box first, or invest
  directly in (B) executor strategy? (Proposed: A first behind a flag, then B.)
- **Q2 — governance in streamed-step:** Enforce vs. observe-only for surfaced
  inner tool calls when the harness can't pause for approval? (Proposed:
  observe-only audit unless an interactive approval protocol exists; per-adapter.)
- **Q3 — contracts promotion:** When does `ExternalAgentExecutor` become a public
  `victor-contracts` type vs. staying framework-internal? (Proposed: only when an
  external package ships an adapter.)
- **Q4 — credential reuse:** Reuse provider credential plumbing or a separate
  external-harness credential store? (Proposed: reuse, scoped per harness.)

## Implementation Plan

### Phase 1: Black-box adapter (~2–3 days)
- [ ] `victor/framework/executors/` package: `ExternalAgentExecutor` protocol +
      `ExternalAgentEvent` type.
- [ ] One black-box adapter (Claude Code via subprocess `stream-json`, collapsed
      to final output) behind `USE_EXTERNAL_EXECUTORS` + `external_executors.*`.
- [ ] Provider-seam wiring (`execute_turn` returns one `CompletionResponse`).
- [ ] `acknowledge_ungoverned` gate + one-time ungoverned warning.

**Deliverable**: delegate a task to Claude Code and get the result inside Victor,
opt-in, with an explicit ungoverned acknowledgement.

### Phase 2: Streamed-step integration (~1–2 weeks)
- [ ] Executor-strategy selection at the `ChatService`/`TurnExecutor` seam that
      bypasses `AgenticLoop` for external executors.
- [ ] Stream-json → `ExternalAgentEvent` mapping; render via the streaming chunk
      path; record every event on the observability bus.
- [ ] Codex + OpenAI-Agents adapters.

**Deliverable**: live, observable external-agent runs surfaced step-by-step.

### Phase 3: Governance & sandbox over surfaced steps
- [ ] Route surfaced `TOOL_CALL` events through the FEP-0005 policy engine
      (enforce where an approval protocol exists; observe-only otherwise).
- [ ] Apply sandbox policy to adapter-spawned subprocesses where applicable.

**Deliverable**: governed external execution for harnesses that support it.

### Testing Strategy
- Unit: event normalization, black-box collapse, fail-open fallback, config/flag
  gating, ungoverned acknowledgement.
- Integration: a fake/stub harness emitting a canned event stream (no real
  binary) exercising both modes; one opt-in real-harness smoke (env-gated, like
  the sandbox e2e).
- Backward compat: external executors off ⇒ zero behavior change (guarded).

### Rollout Plan
- Feature flag `USE_EXTERNAL_EXECUTORS` + `external_executors.enabled`, both off.
- Docs: a "driving external agents" guide; explicit governance-gap callout.

## Migration Path

No migration — purely additive and opt-in. Existing providers and the agentic
loop are unchanged when external executors are disabled.

## Compatibility

### Backward Compatibility
- **Breaking change**: No.
- **Migration required**: No.
- **Deprecation period**: N/A.

### Version Compatibility
- Minimum Python: project baseline (3.10).
- `victor-contracts`: unchanged in Phase 1.

### Vertical Compatibility
- Built-in verticals: unaffected.
- External verticals: may opt to expose external executors once a public
  contract is promoted (deferred).

## References

- [FEP-0005: Policy Engine](fep-0005-policy-engine.md) — governance the black-box
  path escapes and streamed-step mode aims to restore.
- Sandbox isolation (`victor/tools/sandbox/`) — subprocess isolation reused by
  adapters that shell out.
- `victor/providers/base.py:898` (`BaseProvider.chat`), `:317` (`BaseProvider`),
  `victor/agent/coordinators/protocols.py:574` (`ExecutionProvider`),
  `victor/agent/services/turn_execution_runtime.py:165` (`TurnExecutor`),
  `victor/providers/factory.py:525` (`ManagedProviderFactory.create`),
  `victor/providers/openai_compat.py` (OpenAI-compatible patterns).
- Omnigent meta-harness (cross-learning source) — external-agent orchestration.

## Review Process

### Submission
Draft submitted 2026-06-15. 14-day review period per FEP-0001.

### Review Checklist

#### Technical Review
- [ ] Plug-in point decision (Q1) resolved.
- [ ] Governance posture in streamed-step (Q2) resolved.
- [ ] No new parallel abstraction inside `victor/agent`.
- [ ] Fail-open/opt-in/off-by-default honored.

#### Community Review
- [ ] Use cases validated against real harnesses.
- [ ] Governance-gap disclosure is sufficient.

### Decisions
_None yet (Draft)._

### Revision History
- 2026-06-15: Initial draft.

## Acceptance Criteria

- [ ] Q1–Q4 resolved by reviewers.
- [ ] Phase 1 lands behind `USE_EXTERNAL_EXECUTORS`, off by default, with the
      ungoverned acknowledgement gate.
- [ ] Zero behavior change when disabled (guard test).
- [ ] Docs include the governance-gap callout.
