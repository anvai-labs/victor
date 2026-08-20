---
fep: "0005"
title: "Governance Policy Engine (ALLOW/DENY/ASK over tool execution)"
type: Standards Track
status: Draft
created: 2026-06-14
modified: 2026-06-14
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/TBD
---

# FEP-0005: Governance Policy Engine

## Summary

Add a declarative **governance policy engine** to Victor that intercepts agent
tool execution at defined lifecycle phases and returns one of three verdicts —
**ALLOW / DENY / ASK** — composing an ordered list of policies (session →
agent → server, strictest-first). The engine is assembled entirely from
existing framework primitives (the `MiddlewareChain` tool seam, the turn
boundary, the `HITLController` approval flow, and live token/cost accounting),
exposed to the tool pipeline as a single `CRITICAL`-priority middleware and to
the turn path as a `MessagePolicyGate`. The engine spans both tool phases
(`TOOL_CALL` / `TOOL_RESULT`) and message phases (`REQUEST` gates/redacts the
user message before the LLM call; `RESPONSE` gates/redacts the assistant
output), on both the non-streaming and streaming paths. It ships with tool
builtins (cost budget, ask/deny/allow tool gates, per-session call cap) and
message builtins (regex redaction, content block). Everything is disabled by
default behind the `USE_POLICY_ENGINE` feature flag plus `governance.enabled`,
is fail-safe (an absent approval handler resolves ASK to DENY), and adds no new
parallel abstraction layer.

This is a cross-learning from **Omnigent** (a meta-harness over Claude Code,
Codex, and Pi), whose standout, directly-transplantable feature is exactly such
a three-level policy engine.

## Motivation

### Problem Statement

Victor has no unified, user-vetoable governance surface. Safety today is
scattered across per-vertical middleware (`GitSafetyMiddleware`,
`SecretMaskingMiddleware`, …), and there is:

- no spend cap or cost-aware gating,
- no per-session tool-call cap as a runaway/loop guard,
- no "ask me before this risky action" approval flow tied to tool execution,
- no single place to declare governance that stacks across trust levels.

Omnigent demonstrates that this control plane is high-value and can be modeled
cleanly. Victor already owns every primitive needed; they were simply never
assembled into one declarative layer.

### Goals

- A small, composable policy abstraction with ALLOW/DENY/ASK verdicts.
- Phase-based interception at `TOOL_CALL` (gate/modify args) and `TOOL_RESULT`
  (redact/transform output), plus message phases `REQUEST` (gate/redact the user
  message before the LLM call) and `RESPONSE` (gate/redact the assistant output).
- Three-level ordering (session → agent → server) with strict-first
  short-circuit on DENY.
- ASK resolved through the existing `HITLController`, with a fail-safe fallback
  when no approval handler is wired.
- Builtins: `CostBudgetPolicy`, `AskOnToolsPolicy`, `MaxToolCallsPolicy`,
  `DenyToolsPolicy`, `AllowToolsPolicy` (tool phases); `RedactContentPolicy`,
  `BlockPatternPolicy` (message phases).
- Zero behavior change when disabled; opt-in via feature flag + setting.
- No new parallel abstraction layer; reuse `MiddlewareChain`, `HITLController`,
  and `ConversationStore` cost accounting.

### Non-Goals

- Server-wide admin policy distribution/persistence and a container-registered
  approval handler for non-TTY surfaces (API/TUI) — the approval handler is
  pluggable; default is fail-safe deny.
- Sandboxing and the external-harness Executor-seam refactor (separate FEPs).

## Proposed Change

### High-Level Design

```python
from victor.framework.policies import (
    PolicyEngine, PolicyEngineMiddleware,
    CostBudgetPolicy, AskOnToolsPolicy, MaxToolCallsPolicy,
)

engine = PolicyEngine([
    CostBudgetPolicy(max_cost_usd=5.0, ask_thresholds_usd=[3.0]),
    AskOnToolsPolicy(["run_command"]),
    MaxToolCallsPolicy(limit=50),
])
chain.add(PolicyEngineMiddleware(engine, context_provider, approval_handler=handler))
```

A `Policy` evaluates a `PolicyEvent` (phase + tool name + arguments + a
`PolicyContext` snapshot of cost/model/labels) and returns a `PolicyVerdict`.
The `PolicyEngine` composes verdicts: first DENY short-circuits; any ASK (no
DENY) collapses to a single approval; otherwise ALLOW. Argument and result
modifications thread between policies.

### Detailed Specification

#### Verdict composition (`engine.py`)
- Policies run in list order (caller pre-orders strictest/most-local first).
- First `DENY` returns immediately (later policies do not run).
- `ASK` verdicts accumulate; if any ASK and no DENY, return a single ASK with
  combined reason and the first asking policy's name.
- A policy that raises is skipped (fail-open for that one policy) and logged —
  a misbehaving policy never crashes the tool pipeline.
- DENY/ASK are emitted to an optional `(topic, payload)` observability sink.

#### Middleware bridge (`middleware.py`)
- Implements the existing `MiddlewareProtocol`; priority `CRITICAL`, applies to
  all tools (per-policy scoping happens inside the engine).
- `before_tool_call` → `TOOL_CALL` phase. DENY → `MiddlewareResult(proceed=False)`;
  ASK → await `HITLController.process_approval` and collapse to proceed True/False;
  ALLOW → `proceed=True` with any modified arguments.
- `after_tool_call` → `TOOL_RESULT` phase; returns a transformed (redacted)
  result or `None` (unchanged) per the chain contract.
- Because `MiddlewareResult` is boolean-only, ASK is resolved *synchronously
  inside* the async `before_tool_call`. If no approval handler is configured,
  the ASK resolves via `ask_fallback` (`"deny"` default = fail safe).

#### Message-phase gate (`gate.py`)
- The middleware chain is tool-only, so message phases use a thin
  `MessagePolicyGate` adapter wired at the turn boundary — both the
  non-streaming `TurnExecutor.execute_agentic_loop` and the streaming
  `StreamingChatExecutor` — not via `MiddlewareChain`. The gate is built once at
  component assembly (`orchestrator._message_policy_gate`) and shared by both
  paths. (Tool phases already share the non-streaming `ToolPipeline`, so they
  govern streaming with no extra wiring.)
- `gate_request(text)` runs the `REQUEST` phase on the user message *before* it
  enters history or reaches the LLM; `gate_response(text)` runs the `RESPONSE`
  phase on the assistant's final output. Both return a `GateResult`
  (`allowed`, `content`, `reason`, `blocked_by`).
- A DENY short-circuits the turn: REQUEST-DENY returns a refusal
  `CompletionResponse` with no LLM call and nothing stored; RESPONSE-DENY
  replaces the output content and clears tool calls.
- Redaction substitutes the modified text (REQUEST: the redacted message is what
  gets stored/sent; RESPONSE: the redacted output is returned). ASK reuses the
  shared `resolve_policy_ask` helper (same HITL lifecycle and fail-safe as the
  tool path). `PolicyEvent` gained a `content` field and `PolicyVerdict` a
  `modified_content` field, both threaded by the engine like `arguments`/`result`.
- Built lazily by `build_message_policy_gate(settings, container, model)` and
  injected into `TurnExecutor` — `None` (zero behavior change) unless the flag +
  `governance.enabled` are set AND at least one message-phase policy is
  configured (`redact_patterns`, `block_request_patterns`,
  `block_response_patterns`).

### API Changes

New public package `victor.framework.policies` exporting `PolicyEngine`,
`PolicyEngineMiddleware`, `MessagePolicyGate`, `GateResult`, `Policy`, `Phase`,
`PolicyAction`, `PolicyContext`, `PolicyEvent`, `PolicyVerdict`, the builtins,
and `resolve_policy_ask`. New public method
`ConversationController.get_session_cost_usd()` (read-only, returns live
session cost; 0.0 when unavailable). `TurnExecutor.__init__` gains an optional
`message_policy_gate` parameter (default `None`).

### Configuration Changes

New nested settings group `settings.governance` (`GovernanceSettings`):

```yaml
governance:
  enabled: false                       # master switch (also needs the flag)
  cost_budget_usd: 0.0                  # hard cap (0 = off)
  cost_ask_thresholds_usd: []           # soft checkpoints -> one-time ASK
  expensive_models: ["opus", "fable", "gpt-5.5"]
  ask_on_tools: []                      # tools requiring approval
  deny_tools: []                        # tools hard-blocked (DENY)
  allow_tools: []                       # allowlist (non-empty -> others DENY)
  max_tool_calls_per_session: 0         # 0 = off
  ask_fallback: "deny"                  # "deny" (fail safe) | "allow"
  interactive_approval: false           # wire console ASK handler (TTY only)
  # Message phases (REQUEST/RESPONSE):
  redact_patterns: []                   # regex redacted from message + output
  redact_placeholder: "[REDACTED]"
  block_request_patterns: []            # regex -> DENY the user message
  block_response_patterns: []           # regex -> DENY the assistant output
```

New feature flag `USE_POLICY_ENGINE` (`VICTOR_USE_POLICY_ENGINE=true`), opt-in,
off by default. Wiring in `CoordinationBuilders.create_middleware_chain()` adds
the middleware only when the flag and `governance.enabled` are both set.

### Dependencies

None new. Reuses `victor-contracts` middleware protocol, `victor.framework.hitl`,
and `victor.agent.conversation` cost accounting.

## Benefits

### For Framework Users
Spend caps, "ask before risky tools," runaway-call guards, secret/PII redaction
from prompts and outputs, and content blocks — all declaratively, without
writing middleware. Governance applies uniformly across the non-streaming and
streaming chat paths, so behavior does not depend on which surface a user runs.

### For Vertical Developers
A first-class extension point: ship domain policies as small `Policy` subclasses
(tool or message phase) instead of bespoke middleware, and supply a
container-registered approval handler to drive ASK on non-TTY surfaces (HTTP
API, TUI, websocket) without reimplementing the elicitation lifecycle.

### For the Ecosystem
Closes a documented competitive gap (governance / production maturity) and gives
a foundation for future phases (per-step governance of external-harness
executors, FEP-0006) and server-wide admin policy distribution.

## Drawbacks and Alternatives

### Drawbacks
- ASK blocks the tool call synchronously; a slow/absent approver delays the turn
  (bounded by `ask_timeout_seconds`, mitigated by fail-safe fallback).
- Per-session cost relies on the `ConversationController` being resolvable; when
  absent, the cost gate fails open (other policies are unaffected).

### Alternatives Considered
- **A new parallel interception layer** — rejected; violates the "smallest layer
  that fits / no new parallel abstraction" mandate. The existing `MiddlewareChain`
  already provides the exact hook points.
- **Extending `MiddlewareResult` with a native ASK action** — rejected for the
  MVP to avoid a contracts (SDK) change; ASK is collapsed in the middleware.
  Can be revisited as a follow-up.

## Unresolved Questions

- Should `ask_on_tools` support glob/category matching in addition to exact
  names?
- Where should the interactive CLI/TUI approval handler live, and how does it
  surface elicitation to the user?

## Implementation Plan

### Phase 1: Foundation (done)
`victor/framework/policies/` (types, base, engine, middleware, builtins) + unit
tests.

### Phase 2: Integration (done)
`GovernanceSettings`, `USE_POLICY_ENGINE` flag, wiring in `CoordinationBuilders`,
`ConversationController.get_session_cost_usd()`.

### Phase 3: Rollout (in progress)
Done: observability emit of DENY/ASK to the event bus; reusable console
approval handler (`console_approval_handler` / `make_console_approval_handler`)
wired via the opt-in `governance.interactive_approval` setting (TTY-gated);
`DenyToolsPolicy` / `AllowToolsPolicy` tool builtins; **REQUEST/RESPONSE message
phases** (`MessagePolicyGate` at the turn boundary, `RedactContentPolicy` /
`BlockPatternPolicy` builtins, `build_message_policy_gate` wiring); **streaming
governance** — tool phases already share the non-streaming `ToolPipeline`, and
the message gate is now wired into `StreamingChatExecutor` (REQUEST gated before
the stream; RESPONSE gated on the final assistant output — the persisted copy
and authoritative final chunk; live-streamed tokens can't be un-sent).
Remaining: container-registered approval handler for non-TTY surfaces (API/TUI);
team-streaming terminal output; admin policy distribution.

### Testing Strategy
Unit tests covering engine composition (incl. `content` threading), middleware
DENY/ASK/fallback, result/content redaction, all tool + message builtins, the
message gate (allow/redact/deny/ask), and TurnExecutor gating integration.
Integration smoke test through the real
`MiddlewareChain` confirms flag/setting gating and tool-call blocking.

### Rollout Plan
Opt-in flag, off by default; no behavior change until explicitly enabled.

## Migration Path

No migration is required: the policy engine is purely additive and opt-in. With
`USE_POLICY_ENGINE` unset (the default) or `governance.enabled = false`, no
middleware or message gate is wired and behavior is byte-for-byte unchanged —
existing agents, tools, and chat surfaces are unaffected.

### Adopting governance

To enable, set the feature flag and configure at least one policy:

```bash
export VICTOR_USE_POLICY_ENGINE=true
```

```yaml
governance:
  enabled: true
  cost_budget_usd: 5.0
  ask_on_tools: ["run_command"]
  redact_patterns: ["sk-[A-Za-z0-9]+"]   # message-phase redaction
```

No code changes are needed for the builtins. Custom policies are added by
subclassing `Policy`; non-TTY surfaces opt into interactive ASK by registering a
`PolicyApprovalHandler` in the DI container. There is no deprecation: this FEP
introduces new capability and removes nothing.

### Deprecation Timeline
None — no existing API is changed or removed.

## Compatibility

### Backward Compatibility
Fully backward compatible. New optional settings group (auto-defaults), new opt-in
flag, new package. No existing API changed; one additive method on
`ConversationController`.

### Version Compatibility
No `victor-contracts` change required.

### Vertical Compatibility
Verticals may ship `Policy` subclasses; existing middleware is unaffected.

## References

- Omnigent policy engine: `../omnigent/omnigent/policies/`, `docs/POLICIES.md`.
- Victor middleware seam: `victor/agent/middleware_chain.py`,
  `victor-contracts/.../promoted.py`.
- HITL: `victor/framework/hitl.py`.
- Token/cost accounting: `victor/agent/conversation/store.py`.

## Review Process

### Submission
Submitted alongside the reference implementation under
`victor/framework/policies/`.

### Review Timeline
Standard 14-day review per FEP-0001.

### Review Checklist

#### Technical Review
- [x] Reuses existing primitives (no new parallel layer)
- [x] Off by default; opt-in flag + setting
- [x] Boundary-guard clean (no new container/global-state/singleton usage)
- [x] Unit + integration tests passing

#### Community Review
- [ ] API naming feedback
- [ ] Builtin coverage feedback

### Decisions
Pending.

### Revision History
- 2026-06-14: Initial draft + reference implementation.

## Acceptance Criteria

- [x] `victor.framework.policies` package with engine, middleware, builtins.
- [x] `GovernanceSettings` + `USE_POLICY_ENGINE` flag (off by default).
- [x] Wiring in `CoordinationBuilders`; no-op when disabled.
- [x] Tests green; regression and guard suites unaffected.
