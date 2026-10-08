# Repository Agent Instructions

Follow the repository workflow and architecture rules in `CLAUDE.md`. The following performance
policy is mandatory for new code and refactoring:

- Choose architecture from measured end-to-end latency, throughput, CPU, memory, and reliability on representative workloads. Profile first; do not infer a hot path from file size or rewrite Python merely because compiled code may be faster.
- Simplify the algorithm, data flow, allocation rate, cache behavior, concurrency, and dependency surface before changing languages. Keep orchestration and I/O in typed async Python.
- For stable CPU-bound batch work that still misses a measured target, extend the existing Rust workspace through PyO3. Batch inputs, minimize FFI crossings and copies, release the interpreter for measured multi-millisecond Rust-only work, and use portable release targets with runtime feature detection.
- Do not add Cython, Numba, a direct CPython C/C++ extension, or another native toolchain unless a benchmarked case cannot be served cleanly by Python, a vectorized dependency already in that deployment shape, or the existing Rust/PyO3 path. Record the exception and its build, wheel, debugging, and maintenance cost in the design review.
- Every native path must preserve a typed Python reference implementation or an explicit installation requirement, differential parity tests, production-size benchmarks, bounded failure semantics, and observability of native versus fallback dispatch. Never replay a side effect after an FFI failure.
- Treat packaging as part of correctness: validate supported Python versions, operating systems, architectures, baseline CPU features, wheel installation, and fallback behavior before merge. A local `target-cpu=native` result is not release evidence.
- A performance refactor is complete only when it improves the measured user-level target without weakening correctness, cancellation, security, portability, or maintainability. Remove experiments that fail that gate.

Use `docs/architecture/native-acceleration-strategy.md` for the decision matrix and current audit
priorities.

## Shared agent-service work: persistent tracker

For Victor API, web UI, VS Code and related multiagent continuation work, read
[the canonical agent-service tracker](docs/architecture/victor-agent-service-plan.md)
and [FEP-0039](feps/fep-0039-unified-agent-service-api.md) before selecting work.
At session start or after reboot, fetch origin and reconcile the tracker's dated
checkpoint with actual PR/CI/merge state; do not restart completed tasks or assume
a local main checkout is current. Claim a ready task ID and record its owner,
branch, baseline and next action before implementation.

Use TDD in the existing test owner and the tracker's smoke gates. Update the
canonical ledger with evidence at RED/GREEN, PR, merge, release and handoff
boundaries. Keep local passes, merged code, released artifacts and live acceptance
separate. Before stopping, preserve pending work outside temporary directories and
record an executable next step. The tracker is the only task-status authority;
roadmap, audit and session handoffs link to it instead of duplicating its statuses.
These instructions apply to this workstream, not unrelated repository tasks.


For module/submodule and documentation discovery, use
[the repository map](docs/development/repository-map.md). Completed/superseded
interim records are condensed in [completed work](docs/development/completed-work.md)
with immutable history links. Keep current status in its canonical ledger;
transfer unresolved tasks before deleting obsolete handoffs, and update the map
when documentation or source-package ownership changes.
