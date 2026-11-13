# Final local identity co-design acceptance

Date: 2026-11-13. This follow-up joins the official published Sandhi **0.17.1
Python SDK**, the combined **unreleased Sandhi 0.18 server source**, the
remote-aligned local IDM candidate and Victor **0.29.0 source candidate**.
It does not release or deploy those source candidates.

The actual local HTTPS/PostgreSQL fixture uses supported owner registration,
passkey approval, login, consent and PKCE. Victor's originating OAuth client is
distinct from Sandhi's confidential introspection consumer and the explicit
resource audience. The actual private token-file credential and provider path
remain unchanged.

All **three integration/oracle tests passed in 84.47 seconds**. A real Victor
Agent executed a read tool against a synthetic loopback model. Two distinct
actual wire credential hashes and returned Sandhi request IDs joined the IDM
issuance receipts and original durable execution/usage records. The transcript
required the matching successful read result with the actual file contents.

Each of two physical model attempts settled **14 neutral tokens**: two fresh
input, six cache-read, two cache-creation and four output tokens. This split is
already included in each total; no cache tokens are counted twice. These are
synthetic source observations, not evidence of vendor cache savings or prices.

Six refusal controls passed: UserInfo-purpose token, ID token, wrong resource,
private token-file expiry, genuinely elapsed signed-token expiry, and logout.
Denied calls did not dispatch to the model backend. The final source also passed
24 gateway credential tests and complete test collection (40,166 tests).

The installed SDK native-module hash was matched to the clean official-wheel
qualification rather than inferred from its version string. The
[artifact record](artifacts/sandhi-idm-final-source-acceptance-2026-11-13.json)
binds the tested source and binary/module hashes. The earlier
[published-package consumer checkpoint](sandhi-published-0.17.1-consumer-2026-11-13.md)
and older SDK 0.17.0 co-design evidence remain historical snapshots.

The live development IDM still lacks the required introspection discovery
metadata. No live registration, credential or deployment was changed. Reviewed
PRs, CI, merges and separately authorized live acceptance remain distinct gates.
