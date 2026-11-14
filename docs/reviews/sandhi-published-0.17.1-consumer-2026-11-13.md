# Published Sandhi 0.17.1 consumer qualification

Date: 2026-11-13. Victor baseline: `956bf2aafc0c6620f50f0c559a90a9c9d2478aec`.
Victor 0.29.0 remains unreleased. All four mandatory dependency declarations now
pin `sandhi-gateway==0.17.1`: the project manifest and root, API and CPU-embeddings
requirements. Historical 0.17.0 candidate reviews remain unchanged.

The official PyPI Linux ABI3 wheel was downloaded from the exact 0.17.1 release,
verified against PyPI JSON metadata, and installed into a clean Python 3.12 venv.
Fresh processes imported that venv's module and native library, with Victor loaded
from the isolated candidate source. No local build substituted for the wheel.
`pip check` reports no broken requirements. The wheel SHA-256 is
`58e9efca8a4de62166a51a0c998393f9454cd5e3f22a420a2a4d3ce5e70c0b8d`.

The complete provider/configuration suites plus adjacent Sandhi observability and
agent consumers pass **2,593 tests with five existing skips**, without TLS/OIDC name
exclusions. The skips are the existing HuggingFace keyring fixture and four MLX
model-download integration cases. A clean environment initially lacked optional
CodeGraph/LanceDB, Google-auth and ProximaDB fixture packages; after installing
those prerequisites, the complete suite passed. Initial failed/interrupted runs
remain in the external author evidence rather than being represented as passes.

The mandatory complete `tests/` collection preflight also exits successfully with
**40,162 tests** in this clean environment. Optional-package-dependent discovery
differs from the older system-site candidate environment; no name exclusions were
used to make collection pass.

All **eight actual public Agent/tool-service loop cases** pass across four client
dialects and buffered/streaming delivery, using the official GitHub 0.17.1 Linux
gateway binary and synthetic loopback upstream. The archive matches GitHub's
published SHA-256 digest; the extracted proxy reports `sandhi-proxy 0.17.1` and hashes
to `a62eb88e5510064c5265db37ac6dd4b34232131274da73b781dc10f9ddcc156f`.

This is local published-artifact consumer evidence, not Victor publication, a live
identity deployment, or vendor qualification. The separate Victor/IDM/Sandhi
co-design demonstration still records SDK 0.17.0 and local source prototypes; it was
not rerun or relabeled here. The resource-audience candidate is separate from the
published corrective release. See the [artifact record](artifacts/sandhi-published-0.17.1-consumer-2026-11-13.json).
