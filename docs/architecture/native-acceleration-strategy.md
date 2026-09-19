# Native Acceleration Strategy

**Status:** Adopted engineering guidance; implementation investigations remain incremental

**Last verified:** 2026-09-18

Victor should keep Python as the orchestration, integration, and I/O language and use the existing
Rust/PyO3 workspace as its default compiled acceleration path. Adding Cython, Numba, or a direct
CPython C/C++ extension would create another compiler, wheel, debugging, and parity surface. Such a
tool is an exception that requires measurements showing the existing choices cannot meet the need.

## Current codebase reality

- `rust/crates/python-bindings` builds `victor_native` with PyO3 0.29 and maturin. The workspace
  already covers tokenization, context fitting, similarity, embeddings, graph algorithms, parsing,
  filtering, indexing, and security scans.
- Python adapters under `victor/processing/native/` retain reference or fallback behavior. The
  required `native-parity` CI job builds a release wheel and checks native/fallback agreement for
  the tokenizer and context fitter; equivalent differential coverage is not yet universal.
- Rust release builds use optimization level 3, fat LTO, one codegen unit, and panic unwinding so a
  Rust panic does not abort the Python process.
- Release automation builds Linux x86-64 and ARM64 wheels for Python 3.11–3.13, plus the configured
  macOS and Windows targets. The matrix is not uniform: both macOS architectures and Windows x64
  currently build only for the active Python 3.12, Windows ARM64 is absent, and the package declares
  Python 3.10 support without a corresponding Linux wheel in the current matrix.
- The FFI is already batch-oriented in several places, but the current source scan finds
  `Python::detach` concentrated in similarity operations. Several embedding APIs still accept
  nested `Vec<Vec<f32>>`, which copies Python-owned data at the boundary. These are higher-value
  investigations than adding another native language.

## Required decision sequence

1. **Define the user-level target.** Record the deployment shape and a latency, throughput, CPU, or
   memory target. Include realistic input distributions and concurrency.
2. **Measure the whole path.** Profile Python and native variants, including import/startup cost,
   serialization, allocation, FFI conversion, locking, and fallback selection.
3. **Fix structure first.** Remove repeated work, use a better algorithm, batch calls, bound caches,
   reduce object churn, and avoid blocking the event loop. Re-measure.
4. **Qualify a native candidate.** The operation must be deterministic or have an explicit state
   contract, CPU-bound, batchable, and responsible for at least 10% of end-to-end time or a missed
   service target. A prototype must improve its production-size operation by at least 2x and the
   user-level path by at least 10%, after conversion costs.
5. **Choose the narrowest implementation.** Use the table below. Preserve one authoritative
   contract and avoid parallel execution engines.
6. **Prove delivery.** Add parity and failure tests, benchmarks at break-even and production sizes,
   wheel/install checks for supported platforms, and native/fallback dispatch telemetry.

These thresholds are admission gates, not promises that a rewrite will merge. Correctness,
maintainability, portability, and operational failure behavior can still outweigh a speedup.

## Implementation choice

| Choice | Use when | Constraints |
|---|---|---|
| Typed async Python | Control flow, provider/tool I/O, orchestration, rapidly changing policy | Default. Offload blocking work and keep dependencies lazy. |
| Existing vectorized library | A supported deployment shape already carries NumPy or another optimized kernel and the operation maps naturally to it | Do not make a heavy optional dependency mandatory for core execution. Measure data conversion. |
| Rust + PyO3 | Stable CPU-bound parsing, scanning, graph, token, or numeric batch work; shared logic with the Rust edge runtime | Preferred compiled path. One crossing per batch, owned/contiguous data where practical, `Python::detach` for multi-millisecond Rust-only work, no Python callbacks while detached. |
| Rust subprocess/sidecar | Crash or memory isolation is required, or the same engine must run independently of Python | Budget IPC, startup, cancellation, version negotiation, and process cleanup. |
| Cython | A hot typed loop is tightly coupled to an existing C API or Python object model and a Rust implementation would add materially more risk | Exception only. Requires comparative prototype, cross-platform wheels, GIL/free-threading plan, generated-code debugging plan, and explicit owner. |
| Direct C/C++ extension or CFFI | An unavoidable upstream C ABI has no safe maintained binding | Keep the boundary narrow. Prefer wrapping it behind the existing Rust/native facade; document memory ownership and sanitizer coverage. |
| GPU kernel/runtime | Large, batched tensor work dominates the selected deployment shape | Optional extra only; include transfer/setup cost and a CPU fallback. Never move agent control flow to the GPU. |

Numba or runtime compilation is not a core-runtime default: cold compilation, cache behavior,
platform support, and deployment dependencies conflict with predictable CLI and library startup.

## FFI and platform rules

- Pass batches or persistent native handles, not one Python object per call. Establish and benchmark
  the break-even size; keep smaller inputs on Python when the crossing costs more than the work.
- Prefer contiguous buffers or the Python buffer protocol for numeric arrays. Avoid repeated nested
  list-to-vector copies and return compact results.
- Release interpreter attachment for Rust-only computation expected to take milliseconds, and never
  wait on a Rust lock or async operation while attached when that can deadlock Python progress.
- Convert panics and typed native errors at the boundary. A fallback may handle a pure computation
  that failed before effects; it must not silently retry network, filesystem, tool, or billing
  effects.
- Publish conservative CPU baselines and use runtime dispatch for optional SIMD. Test x86-64 and
  ARM64 explicitly; never publish artifacts compiled for the build host's CPU.
- Consider PyO3 `abi3` only after an install/parity benchmark proves its API restrictions fit the
  extension. Stable ABI can reduce the wheel matrix, but it does not remove OS/architecture builds
  and must be evaluated separately from free-threaded `abi3t` support.

## Investigation backlog

1. Build a benchmark and parity inventory for every exported native operation: reference coverage,
   Python/native wall time, break-even input size, peak memory, conversion share, interpreter
   attachment, and call volume in representative agent sessions.
2. Audit CPU-heavy bindings outside `similarity.rs` for safe `Python::detach` regions. Start with
   tokenizer batches, embedding transforms, graph algorithms, and trace scanning.
3. Prototype contiguous-buffer inputs and persistent native indexes for embedding/similarity paths;
   compare them with current nested-vector conversion and the existing NumPy fallback.
4. Make the wheel support table explicit and tested. Resolve the Python 3.10 declaration versus
   wheel matrix, then evaluate an `abi3` proof build before expanding per-version artifacts.
5. Add advisory performance reports first. Ratchet only stable production-size benchmarks with
   noise controls; never make microbenchmarks the sole merge gate.
6. Revisit Cython only if this work identifies a qualifying hotspot for which Rust/PyO3 and an
   already-installed vector library both fail the decision gates.

## Primary references

- [PyO3 performance guidance](https://pyo3.rs/main/performance.html)
- [PyO3 parallelism and interpreter detachment](https://pyo3.rs/main/parallelism)
- [maturin distribution and cross-compilation](https://www.maturin.rs/distribution)
- [maturin platform support](https://www.maturin.rs/platform_support)
- [CPython Limited API and Stable ABI](https://docs.python.org/3/c-api/stable.html)
- [Cython and the GIL](https://docs.cython.org/en/latest/src/userguide/nogil.html)
- [Cython compilation model](https://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html)
