---
fep: "0037"
title: "Versioned /v1/classify surface on victor serve"
type: Standards Track
status: Draft
created: 2026-09-26
modified: 2026-09-26
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
---

# FEP-0037: Versioned `/v1/classify` surface on victor serve

## Summary

Add `POST /v1/classify` to `victor serve`: one structured LLM completion with a
caller-supplied JSON-schema output contract and per-request provider/model
override — no agent loop, no tool supply, no sessions, no streaming. This is
the first **versioned** surface of `victor serve` and the template for
"agent-as-a-service": contract frozen by a CI conformance guard and generated
SDKs, not by documentation.

Motivation is measured, not speculative. The message-hub pilot (first
non-coding consumer) triages every inbound message through a full embedded
agent turn (`VictorClient.chat`) although the workload is one completion;
prompt prefill on a busy local GPU made that turn cost up to minutes per
message, and per-client conversation accumulation multiplied it until the
consumer added manual resets. A bare provider call through the same stack does
the job at a fraction of the tokens and latency. The pilot's measurement table
(A-fixed vs bare-completion vs classify-endpoint p50/p95) is the acceptance
artifact for this FEP.

## Scope

- Route: `POST /v1/classify`, auth-gated like every other HTTP route.
- Request: `{input, schema | preset, sender?, source?, provider?, model?,
  max_tokens?, timeout_ms?}`. `schema` is a JSON Schema for the output object;
  `preset` names a server-side schema (first preset: `triage.v1`). Provider
  override reuses `ProviderOverrideConfig` semantics.
- Response: `{result, usage, model, latency_ms}` on all outcomes including
  422/504 — consumers must never have to guess whether a failure consumed
  tokens.
- Errors: 401 (auth), 422 (contract violation, with the parse details),
  504 (timeout). `latency_ms` always present.
- Decoding is JSON-schema-constrained where the serving stack supports it
  (llama.cpp grammars); otherwise strict-parse with one retry, then a typed
  failure — never a coerced guess.

## Non-goals

- No sessions, conversation memory, tools, teams, HITL pauses, or streaming on
  `/v1` in this FEP. A future agent-tier surface is a separate proposal.
- The existing unversioned routes (chat, completions, git, terminal, mcp, rl,
  HITL) are explicitly **non-contractual**; they remain internal surfaces.
- No model-quality claims: the endpoint transports a completion; verdict
  quality is the consumer's evaluation concern.

## Versioning policy

- Additive-only within `/v1` (new optional fields; no removals or semantic
  changes). Breaking change ⇒ `/v2` with a minimum three-release overlap and a
  documented deprecation window.
- Any contract change requires a FEP update plus a consumer-decision note (the
  TD-0008 rule already establishes this pattern with Sandhi).
- The version exists for consumers, so it is enforced for them too: a
  conformance guard test asserts the frozen request/response schema and is a
  required check in the develop gate (same class as
  `test_architectural_boundaries.py`).

## Enforcement and SDK

- The keyed server disables the docs endpoints today; SDK generation therefore
  emits `app.openapi()` **offline in CI** (unauthenticated app instance) and
  generates the TypeScript SDK from the emitted schema. Drift between the
  frozen contract and the emitted schema fails the build.
- SDK publication follows the existing artifact contracts added in #1194.

## Authentication and metering

- Posture preserved: loopback bind by default, opt-in bearer `api_keys` mapped
  to client ids. Non-loopback binding requires keys, keeps docs off, and
  tightens CORS.
- Attribution is **mandatory** on this route: resolve the client id exactly as
  `/chat` does (the current `/completions` handler discards identity — that gap
  is fixed here, not repeated).
- `rate_limit_rpm` is wired from the constructor or removed from it; a dead
  knob must not survive into a versioned surface.
- Sandhi virtual keys + budgets + usage ledger are the documented
  multi-consumer/billing tier; not required for loopback single-consumer use.

## Resource bounds

- `max_tokens` capped server-side; input size capped (4xx with `latency_ms`).
- Documented behavior under provider saturation: the local inference queue
  (e.g. a 2-sequence llama.cpp server) is the throughput ceiling; the endpoint
  times out per `timeout_ms` rather than queueing unboundedly.

## Rollout

1. Land after the v0.10.0 promotion completes (develop churn during promotion
   reruns the full battery).
2. The message-hub pilot is the acceptance consumer: it moves triage to
   `/v1/classify` through the generated SDK only if measured p95 beats its
   embedded path; its keyword → fail-safe degradation ladder is unchanged.
3. Success criteria: the Phase 0 latency table plus one release cycle with
   zero contract drift.

## Compatibility

- Zero change to the embedded `VictorClient` path.
- `/completions` remains as-is (coding-assistant internal), with a noted
  follow-up to add the attribution join there.

## References

- Design review handoff: message-hub `docs/victor-codesign-api-vs-embedded-handoff.md`
  (verdict C — hybrid, phased; reviewer identified the accumulator-state defect
  and the unattributed `/completions` gap this FEP addresses).
- FEP-0020 (usage attribution), FEP-0009 (SDK tool contract),
  TD-0008 (Victor–Sandhi consumer-decision rule).
