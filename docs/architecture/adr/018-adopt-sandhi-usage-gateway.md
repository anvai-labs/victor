# ADR-018: Adopt the `sandhi` OSS usage gateway (own repo, not a Victor module)

## Metadata

- **Status**: Accepted
- **Date**: 2026-07-18
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: ADR-014/ADR-015 (shared `victor-codegraph` cross-repo pattern);
  AnvaiOps **ADR-0047** (authoritative open-core split); ProximaDB **ADR-067**
  (sibling consumer)
- **Related FEP**: FEP-0020 (public feature spec + framework seams)

## Context

Victor tracks token/USD cost per session (`victor/agent/session_cost_tracker.py`)
but has **no per-user/per-team attribution and no gateway** in front of a shared
provider key (see FEP-0020 for the full gap analysis). The capability is wanted
across three repos — Victor, ProximaDB (Rust), AnvaiOps (commercial) — each with
its own provider layer in a different language. A shared in-process library cannot
span them; the cross-language unifier is a reverse-proxy + a neutral usage-event
schema.

The homing question — own project, or folded into Victor / ProximaDB / AnvaiOps —
is decided authoritatively in **AnvaiOps ADR-0047**. This ADR records Victor's
**local consequence**: which repo owns the code, and which side of the open-core
line Victor stays on. (Per the cross-repo rules in this directory's README: one
repo is authoritative for a shared decision; the others hold thin consumer ADRs
that link to it. AnvaiOps ADR-0047 is authoritative; this is Victor's consumer ADR;
Victor is owner-of-record for the public feature spec, FEP-0020.)

## Decision

**Adopt a standalone Apache-2.0 OSS repo `sandhi` (`anvai-labs/sandhi`) as an external
dependency** — not a `victor/` internal package. Victor consumes it two ways: an
in-process metering middleware over `BaseProvider`, and an optional reverse-proxy
serve mode with virtual keys for shared-key teams (FEP-0020 § Proposed Change).

Victor stays entirely on the **OSS side** of the open-core line (AnvaiOps
ADR-0047 D4): virtual keys, per-user/per-team attribution, budgets/rate-limits,
local cost display, and a self-hosted dashboard are all OSS. Authoritative
multi-tenant billing, tier→$ policy, SSO/RBAC governance, and managed dashboards
at scale are AnvaiOps's commercial layer — Victor does not build them.

The dependency direction is one-way (`victor → sandhi`). The typed provider runtime is a
required Victor dependency; the separately configured attribution gateway/proxy remains
default-off (FEP-0020 § Migration). The current repository requirement is
`sandhi-gateway==0.5.0` in `pyproject.toml`. The earlier 0.1.1→0.1.2 rollout described
the July migration and is historical, not the current dependency train.

For provider transport, the ownership boundary is explicit: Victor owns agent/model
policy and normalized framework objects; Sandhi owns endpoint routing, headers, HTTP/SSE,
wire errors, resilience, and neutral usage extraction. The active transport is the sole
retry/timeout/circuit owner. Per TD-0004, Sandhi's catalog contains stable wire facts **and
curated model data** (id, context window, max output, wire capabilities; no pricing) exposed via
`provider_models_json`, while Victor retains model **policy** — which models to expose/select and
the discovery UX. (Revised 2026-07-23 by TD-0004; previously the catalog held wire facts only and
Victor retained all model metadata.) An admitted provider has one
Sandhi execution path with no replay, demotion, or native Python wire fallback.

For OpenAI-compatible Chat Completions, Victor owns normalized history and provider/model
capability policy while Sandhi owns the stable wire-role contract. Both preserve
`developer`, `system`, `user`, `assistant`, `tool`, and legacy `function`; Sandhi rejects
unknown roles and missing `tool_call_id`/legacy function `name` before HTTP. Neither layer
silently rewrites `developer` to `system`, because that is model-specific policy.

OpenAI Responses is a separate Sandhi endpoint family, not a Chat Completions alias. API-key
Responses uses the public `/v1/responses` profile. ChatGPT subscription OAuth explicitly selects
the Codex profile at `/backend-api/codex/responses`; Victor acquires and refreshes the credential
and optional workspace account id, while Sandhi enforces the profile's item-array input,
instructions, `store=false`, SSE-only upstream behavior, stream aggregation, and Responses token
metering. Token text and URL shape are never used to guess the protocol.

## Rationale

- **Why its own repo, not folded into Victor?** A Python-internal home cannot be
  consumed in-process by ProximaDB (Rust) or third parties; the language-agnostic
  proxy must live where every consumer is first-class. This mirrors why the
  code→CPG chunker became the standalone `victor-codegraph` rather than staying in
  `victor-coding`.
- **Why OSS (Apache-2.0)?** Matches Victor + ProximaDB; per-user attribution is
  the adoption lever (LiteLLM/Helicone ship it OSS). Gating it would put Victor
  *behind* the incumbents on the exact feature that drives adoption. The
  commercial moat is billing + governance + managed scale, which AnvaiOps holds.
- **Why a thin consumer ADR here?** The homing/business decision is AnvaiOps's
  (it owns the commercial split); Victor records only the local consequence
  (dependency + boundary posture), per this directory's cross-repo rule 3.
- **Pro:** closes the attribution gap; one neutral event feeds both Victor's
  dashboard and AnvaiOps billing; provider wire plumbing converges in Sandhi instead
  of multiplying in each consumer. **Con:** a new external dependency and a cross-repo wire
  contract to keep stable.

## Consequences

- **Positive**: Victor can answer "who spent what" across a shared key, budget
  per user, and rate-limit runaway subjects — matching the OSS incumbents while
  the code stays reusable by ProximaDB and AnvaiOps through one schema.
- **Negative**: a fourth repo in the ecosystem; the neutral usage-event schema is
  now a cross-repo API whose breaking changes must coordinate (same discipline as
  `victor-codegraph`). Building the proxy ahead of real shared-key demand would be
  premature — hence code is out of scope (docs only); the proxy ships when demand
  is real.
- **Neutral**: default-off; attribution fields default to `None`, so no migration
  and no change for users who don't opt in. The remaining rollout work is tracked
  as a TD in the Technical Debt Register, not in this ADR.

## Implementation

The framework seams and migration ledger live in FEP-0020 and Sandhi TD-0002. Attribution,
middleware, proxy serve mode, shared usage parsing, and the direct typed runtime have landed.
The admitted OpenAI-compatible cloud providers are thin Victor model/orchestration policies;
their endpoint routing, HTTP/SSE, errors, resilience, roles/tools, and usage parsing execute in
Sandhi. Groq payload budgeting and Cerebras inline-reasoning presentation remain in Victor because
they are agent/UX policy, not provider transport.

## Alternatives Considered

- **Fold the gateway into `victor/`.** Rejected: cannot serve ProximaDB/third
  parties over the proxy (AnvaiOps ADR-0047 D1).
- **Extend `SessionCostTracker` with attribution, no proxy.** Rejected: covers
  same-process Victor only, not shared-key/internal-network teams (FEP-0020 Alt 2).
- **Depend on an external gateway (LiteLLM/Portkey) directly instead of `sandhi`.**
  Rejected as the *primary* path: those are the reuse targets *inside* `sandhi`
  (AnvaiOps ADR-0020 D2), but Victor needs the neutral event + code-graph-aligned
  attribution the shared schema provides; `sandhi` is the thin owner of that seam.
  (`sandhi` may wrap a LiteLLM-shaped router internally.)

## References

- AnvaiOps **ADR-0047** — AI usage gateway open-core split (authoritative homing)
- ProximaDB **ADR-067** — egress-metering consumer (sibling)
- FEP-0020 — public feature spec + framework seams
- ADR-014 / ADR-015 — the `victor-codegraph` cross-repo package precedent
- `docs/architecture/adr/README.md` § "External ADR series (cross-repo)"

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-07-18 | 1.0 | Initial ADR | Vijaykumar Singh |
| 2026-07-21 | 1.1 | Accepted; pinned Victor/Sandhi provider ownership and migration gates | Vijaykumar Singh |
| 2026-07-22 | 1.2 | Added explicit public Responses and ChatGPT subscription/Codex profiles | Vijaykumar Singh |
| 2026-07-23 | 1.3 | Revised catalog boundary per TD-0004: Sandhi owns model catalog data; Victor owns policy | Vijaykumar Singh |
