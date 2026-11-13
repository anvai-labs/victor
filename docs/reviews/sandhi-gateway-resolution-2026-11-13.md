# Sandhi gateway configuration: early auth and endpoint boundary

Prepared consumer baseline: `5aee2643387afe3c7e196139aa510d1a0f4f1cb9`.
Cumulative develop baseline: `8912e69c7ac3df7d856909dc744f2900a6386dbe`.

The ordinary Settings/ProviderConfigRegistry path previously resolved upstream
accounts and then ran direct strategies even with a gateway configured. A closed,
synthetic reproduction observed an LMStudio model-list probe and an OpenAI local
key resolver. A missing upstream account could stop gateway construction. Corrected
async factories alone did not cover this ordinary configuration entrypoint.

Gateway configuration now merges provider/profile/environment policy and validates
its endpoint/auth metadata before any upstream AccountManager, key strategy or
discovery. Account resolution is lazy and direct-mode only. Named gateway OIDC and
custom TokenCredential objects retain their authentication semantics without reading
the token during metadata validation. Malformed gateway configuration fails before
upstream work; a bad provider-file gateway cannot silently become a direct call.
Direct-mode account/strategy precedence is retained. Model, timeout and cache policy
stay independent of source credentials and client wire dialect.

Actual baseline regressions had 70 failures and one direct-path control pass.
After the correction, the final integrated candidate environment passes **2,563
provider/configuration and consumer checks**, with five existing skips and no TLS
or OIDC name deselections. Full collection completes: **40,362 tests**. The real
Agent/tool-service loop passes all eight protocol/delivery cases against the
qualified Sandhi gateway binary `b43a641becc32535fcece302c94a0a8016bc68a1c5a9e14b950f3632810561fd`.
Black/Ruff pass; configured typing and strict documentation checks pass on the
byte-matched author handoff. Source gateway bindings are pinned to 0.17.0; this
qualification uses its locally built candidate wheel.

The author environment's broader test attempts stalled in unchanged native/GPU/
OIDC fixture paths, including controls on the original baseline. Those runs remain
incomplete, with no cause attributed to this fix and no fabricated full-pass claim.
The final integrated run above completed independently in the candidate environment.

Physical calls remain retry-free; gateway count conventions remain server-owned;
cache intent has one server compiler; unknown gateway pricing remains unavailable.
No API keys, real provider account operations or live provider probes were used by
these configuration regressions. Final exact-commit independent review, published
Sandhi dependency installation and required public CI remain separate consumer
push/release gates. No Victor publication or deployment is included.
