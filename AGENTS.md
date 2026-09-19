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
