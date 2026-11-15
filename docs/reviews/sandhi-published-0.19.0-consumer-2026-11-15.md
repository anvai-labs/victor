# Published Sandhi 0.19.0 consumer qualification

Date: 2026-11-15. Victor **0.29.2** is the source candidate prepared from develop
`e96dc578a89deca500605f475934bd8cd44817e9`; the final reviewed release head is
authoritative. All four dependency declarations pin `sandhi-gateway==0.19.0`.

The official Linux ABI3 wheel was downloaded from PyPI metadata and its SHA-256
matches `a0fbbe20584a04440aee9e7a891e9287ee2549ac1373beaf8f38a3c15fe95116`. A fresh
Python 3.12 environment reports package version 0.19.0, wire contract v1, chat
contract v1 minor 14, and a clean dependency check. The imported native module
SHA-256 is `b4eb692e1adedbd30d67b6903299fc689045f162ff7ada9307d56153f4e7103f`.

The immutable GitHub Linux archive was downloaded from `v0.19.0` and matched the
published digest. Its `sandhi-proxy` reports 0.19.0 and hashes to
`60a6b3da9a168a0f80b981228e2c4e7bce16847d62a8da4365bf2c76cfcf4705`.

| Qualification | Result |
| --- | --- |
| Official wheel import and dependency check | Passed |
| Consumed contract and real binding construction | 48 tests passed |
| Public Agent/tool loops | 16 tests passed across four client dialects, buffered/streaming and token/starter modes |
| Sandhi public artifact verifier | GitHub, PyPI, crates.io and npm targets verified after registry propagation retry |

This differential qualification covers the Sandhi 0.19.0 binding transition. It
does not make a live provider call, publish Victor, upgrade a managed service, or
claim unattended operation. Historical 0.18.x and 0.17.x records remain useful
provenance but do not substitute for this acceptance. The machine-readable record
is [sandhi-published-0.19.0-consumer-2026-11-15.json](artifacts/sandhi-published-0.19.0-consumer-2026-11-15.json).
