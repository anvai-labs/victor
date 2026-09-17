# ADR-029: Provider Support Tiers

## Metadata

- **Status**: Accepted
- **Date**: 2026-08-02
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 006 (provider integration improvements), 022 (provider gateway feature layer)
- **Scope**: Support policy and documentation only. No adapter is removed, no runtime
  behavior changes, no registry gating is introduced. The only code artifact is the
  `TIER_1_PROVIDERS` constant in `victor/providers/registry.py`.

> Terminology: this ADR defines **support tiers** (maintainer commitment per provider
> adapter). This is unrelated to the *model tiers* in
> [provider-agnostic-tiers.md](../provider-agnostic-tiers.md), which classify model
> capability within a provider. Always say "support tier" when referring to this ADR.

## Context

Victor ships 25 provider adapters (`victor/providers/*_provider.py`, derived
automatically by `scripts/ci/check_docs_drift.py`). All of them are unit-tested, but
the implied support promise has been uniform and therefore dishonest: a single
maintainer cannot give every adapter the same depth of integration testing, issue
triage, and API-drift tracking. Provider APIs churn constantly (new tool-call formats,
streaming changes, auth schemes), and the cost of keeping an adapter *verified* — not
merely compiling — is paid per provider, per release.

In practice a small set of adapters carries nearly all real usage: the major hosted
APIs (Anthropic, OpenAI, Google Gemini, AWS Bedrock) and the local/self-hosted
OpenAI-compatible backends (Ollama, vLLM). The remaining adapters see occasional use
and community-reported fixes. Documentation should state that difference explicitly
instead of letting users infer equal support everywhere.

## Decision

**Adopt two support tiers for provider adapters. Documentation and one constant only —
no adapter is deleted, disabled, or deprioritized in the registry.**

### Tier 1 — six providers

`anthropic`, `openai`, `google` (Gemini), `ollama`, `vllm`, `bedrock`
(primary registry keys as registered in `victor/providers/registry.py`; aliases such
as `aws` → `bedrock` inherit their primary's tier).

Commitment:

- Integration-tested against real or recorded provider behavior; regressions here
  block release.
- Issues are triaged and fixed by the maintainer.
- Tracked proactively for upstream API drift.

### Community tier — all other adapters

Every adapter not listed above (18 today): `azure`, `cerebras`, `deepseek`,
`fireworks`, `groq`, `huggingface`, `llamacpp`, `lmstudio`, `mistral`, `mlx`,
`moonshot`, `openrouter`, `qwen`, `replicate`, `together`, `vertex`, `xai`, `zai`.

Commitment:

- Unit-tested; kept importable and registered.
- Best-effort support: issues are welcome and fixes are reviewed, but response time is
  not guaranteed. PRs are actively welcomed and prioritized for review.

### No-deletion stance

Membership in the Community tier is **not** a deprecation signal. No adapter is
removed, hidden, or feature-gated by this decision, and none of the tier machinery may
be used for behavioral gating. `TIER_1_PROVIDERS` is informational metadata for docs,
tests, and release checklists only.

### Promotion / demotion criteria

Tier membership is revisited when evidence changes, on these signals:

- **Usage**: sustained real-world usage (issue volume, telemetry where available,
  community reports) argues for promotion; a Tier-1 provider with negligible usage may
  be demoted to Community.
- **Maintainer bandwidth**: Tier 1 size is capped by what can genuinely be
  integration-tested per release; the tier must never promise more than is delivered.
- **CI cost**: a provider can only be Tier 1 if its verification is affordable in CI
  or via a documented pre-release procedure (see vLLM note below).

Any tier change is recorded as a revision to this ADR and mirrored in `SUPPORT.md`.

### vLLM verification honesty

Running live vLLM inference per-PR is impractical on hosted CI runners (GPU-dependent,
heavyweight model loading). vLLM's Tier-1 status is therefore verified through:

1. the OpenAI-compatible surface tests (vLLM's adapter speaks the OpenAI-compatible
   protocol, which is exercised by mocked/recorded unit and integration tests), and
2. a manual pre-release checklist item that runs a smoke conversation against a live
   vLLM server before tagging a release.

This is a weaker guarantee than live per-PR inference and is stated here deliberately
rather than implied away.

## Consequences

- **Positive.** The support promise is honest and matches maintainer capacity. Users
  choosing a provider know what they are getting. Release effort concentrates where
  usage is. Community contributors get an explicit, welcoming contract for the other
  adapters.
- **Neutral.** Zero behavioral change: the registry, routing, and every adapter work
  exactly as before. `TIER_1_PROVIDERS` is a frozen constant with no runtime readers.
- **Cost.** Two documentation surfaces (`SUPPORT.md`, this ADR) plus the constant must
  stay in sync; a unit test pins `TIER_1_PROVIDERS` to real registry keys so the
  constant cannot silently drift from the registry.

## Alternatives Considered

- **Delete or deprecate low-usage adapters.** Rejected outright: adapters are cheap to
  keep (lazy-loaded, unit-tested) and deletions would break existing users for no
  maintenance win. This ADR exists partly to put the no-deletion stance on record.
- **Three or more support tiers** (e.g. Tier 1 / Tier 2 / experimental). Rejected:
  finer gradations imply precision the support reality does not have; two tiers map
  exactly to "integration-tested + triaged" vs "unit-tested + best-effort".
- **Behavioral gating** (warnings or restrictions on Community-tier providers at
  runtime). Rejected: support tier is a documentation fact, not a product limitation;
  gating would punish working setups.
- **Live vLLM CI job.** Rejected for now: hosted runners lack GPUs and a
  representative model; a self-hosted GPU runner is not justified for a
  single-maintainer project. Revisit if infrastructure changes.

## Validation

- `tests/unit/providers/test_providers_registry.py` asserts every name in
  `TIER_1_PROVIDERS` is a registered primary provider key, so the constant and the
  registry cannot drift apart.
- `scripts/ci/check_docs_drift.py` continues to derive the total adapter count (24)
  from the file tree; this ADR's prose uses the derived count.

## Revision History

| Rev | Date | Change |
|-----|------|--------|
| 1.0 | 2026-08-02 | Initial decision: two support tiers, six Tier-1 members, no-deletion stance, vLLM verification note. |
