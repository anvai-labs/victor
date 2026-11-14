# Published Sandhi 0.18.0 consumer qualification

Date: 2026-11-13. Victor 0.29.0 remains unreleased. The candidate is based on
Victor develop `12fc116dca58eb111c295c5ec252153faf3bbe54`; the final reviewed PR
head is authoritative for merge. All four dependency declarations pin
`sandhi-gateway==0.18.0`: the project manifest and the root, API and
CPU-embeddings requirements.

The official Linux ABI3 wheel was downloaded from PyPI, checked against the
0.18.0 JSON metadata and installed without substitutions. Its SHA-256 is
`df7851eec4e40e0b19de3ab0d10877f378a52e8702062f7ad8bbb62fd1256479`; the imported
native module SHA-256 is
`5a327110d6a5eea82fc4b4ed4efdbbc1e6f9b76bfecfcf910ef188ddd62c7d5e`. The fresh
Python 3.12 process reports wire contract v1, chat contract v1 minor 14, and
`pip check` reports no broken requirements.

The official Linux gateway archive was downloaded from the Sandhi GitHub release
and matched to the published digest. The extracted `sandhi-proxy` reports 0.18.0
and hashes to
`fb877221f9b89b9ee375712a591ac66a279dae50f1324755ac93c7290a84b3c6`.

| Qualification | Result |
| --- | --- |
| Static gates | Black, Ruff, full mypy and version synchronization pass |
| Provider/configuration and policy units | 2,313 passed, six existing skips |
| Public Agent/tool loops | 18 cases pass across four client dialects, buffered/streaming and token/starter modes |
| Local OIDC Agent session | Three integration tests pass, including two joined 14-token settlements and six refusal controls |
| Private-CA transport | Without a scoped CA bundle the SDK fails in TLS transport; with the bundle it reaches a deliberate HTTP 401 |
| Session-bound refresh | Real PKCE, rotation, concurrent reader availability, expiry, lost delivery, replay revocation and logout controls pass |
| Collection | Complete `tests/` discovery passes with the count recorded in the artifact record |

This is local Linux/Python 3.12 consumer qualification. It does not publish
Victor, qualify every live identity provider or vendor endpoint, or claim ongoing
unattended service operation. Historical 0.17.x records remain useful provenance
but do not substitute for this 0.18.0 acceptance. See the machine-readable
[artifact record](artifacts/sandhi-published-0.18.0-consumer-2026-11-13.json).
