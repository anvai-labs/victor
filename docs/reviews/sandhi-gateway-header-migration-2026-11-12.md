# Gateway header preparation for canonical accounting

Victor's gateway mode no longer generates `Idempotency-Key` or forwards
case-insensitive caller variants through static or per-call headers. Sandhi's
tracked admission rejects that header. Run/session/step attribution and local
logical-call correlation remain separate from inference replay and settlement.
Direct-provider custom header behavior remains independent. Existing gateway
retry/fallback suppression and retry-free physical handle construction remain.

The focused header/construction/resilience selection passed **213 tests**. The
broader providers and selected agent/core resilience selection passed **1,916
tests**, with five skips, against the Sandhi 0.17.0 candidate wheel. New cases
exercise buffered and streaming boundaries, custom header casing, one captured
invocation, step attribution and direct custom-header preservation. These boundary
fixtures use the fake binding; they do not prove actual two-hop durable execution.
Scoped Black/Ruff and configured mypy for the two changed source modules passed.
Configured typing success is not unsuppressed repository-wide type compliance.

This is a preparation checkpoint, not completed cutover. Explicit gateway peer
dialect/counts metadata, server-only cache compilation, streaming policy, mandatory
durable accounting and migration acceptance remain separate work. Sandhi's retained
native-exit, shutdown and accounting reliability failures remain open. Victor 0.29.0
and its exact `sandhi-gateway==0.17.0` dependency are prepared source versions;
publication and supported native packaging are unqualified. Keep this consumer
branch local/unmerged until the dependency publication gate is satisfied.
