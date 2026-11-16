# Published Sandhi 0.20.0 consumer qualification

Date: 2026-11-16. Victor develop `7c573abd92848b67bd5ad50f8cfef957ba5306c5`
(post-0.29.2 logical checkpoint) is the exact source baseline. Victor `VERSION`
remains `0.29.2`; this qualification does not publish Victor or claim a new
Victor release. All four dependency declarations are moved to
`sandhi-gateway==0.20.0`.

The official Linux ABI3 wheel was downloaded from PyPI metadata and its SHA-256
matches `2149caad6467a0cd81cf2ea751ef9acee88f1bc049137123e3ef5746eaad51e4`.
A clean Python 3.12 environment reports package version 0.20.0, wire contract
v1, chat contract v1 minor 14, and `pip check` finds no broken requirements.
The imported native module SHA-256 is
`c90d52e4dffb4049d5cdb58a2ba17fb8b7440943107d7b1cfac24cd8fa04b130`.

The immutable GitHub release tag `v0.20.0` points to Sandhi commit
`184215c7b265f7214246b5db50455ff3317cbb04`. Its Linux proxy archive matches
the release digest; the extracted `sandhi-proxy` reports 0.20.0 and hashes to
`6f0152c37d677fc2640f7dbafda6e891b98d15224cbd339f6802e38183a9d896`.

| Qualification | Result |
| --- | --- |
| Official wheel import and clean dependency check | Passed |
| Consumed contract and real binding construction | 48 tests passed |
| Public Agent/tool loops | 16 tests passed across four client dialects, buffered/streaming and token/starter modes |
| Released proxy version and digest | Verified |

This differential qualification covers the Sandhi 0.20.0 consumer pin and its
hierarchical rolling-window release. It does not make a live provider call,
publish Victor, upgrade a managed service, or claim unattended operation. The
0.19.0 and earlier consumer records remain historical provenance and do not
substitute for this acceptance. The machine-readable record is
[sandhi-published-0.20.0-consumer-2026-11-16.json](artifacts/sandhi-published-0.20.0-consumer-2026-11-16.json).
