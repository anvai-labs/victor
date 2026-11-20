# ADR-037: Untrusted-content containment at the context boundary

## Metadata

- **Status**: Proposed
- **Date**: 2026-11-16
- **Decision Makers**: Vijaykumar Singh
- **Related Documents**: ADR-035 (sandbox posture), ADR-036 (graduation policy), FEP-0013 (shell safety policy), `victor/security/source_credibility.py`, `victor/framework/rl/tool_reputation.py`, `victor/agent/services/context_lifecycle_service.py`

## Context

Victor's agent loop routinely ingests text Victor does not control: file contents, web pages
fetched by search/read tools, tool outputs from MCP servers and external plugins, issue
bodies, and code comments. All of it enters the model's context window as plain assistant-
visible content, with no structural marking of provenance and no defense against instructions
embedded in that content. Prompt-injection handling in the repo today is *analytical, not
defensive*: `victor/security/source_credibility.py` scores search-result sources,
`victor/framework/rl/tool_reputation.py` learns tool reliability signals, and
`prompt_pipeline.py` consumes both as RL features — but no mechanism marks untrusted spans,
detects injection patterns, or constrains what the model may do with embedded instructions.
The shell safety policy (FEP-0013) and permission modes gate *what tools may do*; nothing
gates *what untrusted text may ask*.

The 2025–2026 baseline moved against this posture: agentic products converged on treating
tool/web/file content as untrusted data — delimited, provenance-tagged, and passed through
an injection screen before context entry — because indirect injection via fetched content
became the dominant documented attack on coding agents. Victor's permission system limits
blast radius, but a model that follows an instruction smuggled inside a fetched web page can
still attempt destructive commands the user must then veto manually, and in
`danger-full-access` mode there is no veto at all.

## Decision

1. **Provenance marking is structural, not conventional.** Untrusted content (tool outputs,
   fetched web/file text) is wrapped in a delimited, provenance-tagged envelope at the point
   it enters context — one canonical wrapper owned by the context layer, encoding source
   class (file/web/mcp/plugin), origin identifier, and a per-session canary token. The
   system prompt states the containment rule: instructions inside envelopes are data.
2. **An injection screen runs on the tool-result path** (`context_lifecycle_service.py`'s
   `before_tool_output` hook): pattern classifiers for known injection families (instruction
   impersonation, tool-call forgery, canary exfiltration attempts) plus the existing
   credibility/reputation signals. Screen verdicts are **advisory annotations plus
   escalation**: flagged spans are additionally highlighted in-envelope; canary hits
   (envelope-internal tokens appearing in tool *arguments* or shell commands) hard-fail the
   action through the existing permission/`AuthorizationDecision` gate — fail-closed.
3. **Canary-based exfiltration detection is the non-negotiable floor**: every envelope
   carries unique canary strings; if a canary appears in any outbound tool call, the call is
   blocked and recorded. This is deterministic, model-independent, and catches the
   highest-severity outcome (content steering outbound calls) without depending on
   classifier quality.
4. **Screen verdicts feed the existing learning loops** (`tool_reputation.py`,
   `source_credibility.py`) so the defense improves from field data; verdicts and blocks are
   observable through the standard event stream.
5. **Graduation**: structural marking (1) and canary blocking (3) ship default-on — they are
   model-independent and cheap. Classifier escalation (2) ships default-on as annotation,
   with hard-fail behavior gated behind the ADR-036 evidence policy after a false-positive
   battery run.

## Rationale

The permission system answers "what may this tool do"; containment answers "what may this
text ask". The two compose: envelopes + canaries give deterministic guarantees that do not
degrade with model quality, classifiers add probabilistic depth, and the learning loop turns
field traffic into better signals — reusing infrastructure that already exists instead of
building a parallel defense stack. Canary blocking is elevated to the non-negotiable tier
because it is the only measure in the stack with no false-positive/negative ambiguity.

## Consequences

- **Positive**: indirect injection via fetched content gains a structural defense layer;
  exfiltration attempts become deterministically blockable; the existing credibility/reputation
  investments become defenses rather than analytics; incident forensics improve (envelopes
  record provenance per span).
- **Negative**: envelopes consume tokens (bounded: wrappers are small, but at scale the tax
  is real — the budget machinery absorbs it); classifiers will produce false positives that
  annotate legitimate content (why hard-fail is gated); system-prompt containment language
  can be overridden by a sufficiently steered model — which is precisely why the floor is
  deterministic canary blocking, not prompt language.
- **Neutral**: tool schemas, provider contracts, and the wire protocol are unchanged;
  envelopes exist only inside model context, not on the wire.

## Implementation

FOU Tier-S increments (tracked in `docs/architecture/victor-foundations-plan.md`):

1. **FOU-S3a**: canonical envelope + canary infrastructure in the context layer; RED: fixture
   corpus of injection payloads is wrapped, canaries present, envelope provenance correct.
2. **FOU-S3b**: canary blocking in the tool-call/permission path (fail-closed); negative
   tests pin that a canary in any outbound argument blocks the action and records the event.
3. **FOU-S3c**: classifier screen on `before_tool_output` with annotation-only default;
   adversarial fixture corpus (injection families) pinned in tests.
4. **FOU-S3d**: hard-fail graduation battery per ADR-036; verdict feeds into
   `tool_reputation.py`.

## Alternatives Considered

- **Prompt-only containment instructions**: rejected as the primary mechanism (evidence
  base: prompt-language defenses are bypassable); kept as the accompanying system-prompt rule.
- **Block all web/MCP content by default**: rejected; breaks the research and integration
  use cases the verticals exist for.
- **Sanitize/rewrite untrusted text**: rejected for v1; rewriting risks corrupting legitimate
  content (code, logs) that must reach the model verbatim; annotation preserves fidelity.
- **Outbound-only defense (canaries without envelopes)**: considered; envelopes retained
  because provenance marking also enables per-source classifier tuning and forensics.

## References

- `victor/security/source_credibility.py`, `victor/framework/rl/tool_reputation.py` —
  existing signals this ADR promotes into defenses
- `victor/agent/services/context_lifecycle_service.py` — `before_tool_output` seam
- `victor/security/permissions.py` — `AuthorizationDecision` gate the canary block composes with
- FEP-0013 — shell safety policy (the "what tools may do" half of the model)
- OWASP LLM Top-10 (LLM01: prompt injection) — threat-model reference
- `docs/architecture/victor-foundations-program.md` — FOU Tier-S ordering

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-11-16 | 1.0 | Initial ADR | Vijaykumar Singh |
| 2026-11-20 | 1.1 | Dispatch-enforcement hardening (Codex review fix wave 1): session-owned canary binding (owner state bound per turn), fail-closed envelope publication and dispatch gates on containment-infrastructure failure, full serialized-argument canary walk (keys + serialized leaves), flagged-family evidence moved from task-local ContextVar to bounded session-owner state with honest family-overlap semantics (overlap is necessary, not proof of echo or causation) | Vijaykumar Singh |
