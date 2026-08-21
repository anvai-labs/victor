# Victor Rust Workspace

Rust libraries, the standalone edge runtime, and high-performance Python extensions for Victor.

## Packages

| Workspace member | Distribution | Purpose |
|---|---|---|
| `crates/protocol` | crates.io: `victor-protocol` | Portable message and tool-call contracts |
| `crates/state` | crates.io: `victor-state` | Conversation and scoped state management |
| `crates/tools` | crates.io: `victor-tools` | Tool schema registry and validation |
| `crates/edge-runtime` | crates.io: `victor-edge` | Standalone edge agent runtime and binary |
| `crates/python-bindings` | PyPI: `victor_native` | Python native extensions; never published to crates.io |

`victor-edge` depends on the other three Rust crates, so crates.io publication is intentionally
ordered as `protocol → state → tools → edge`.

## Features

- **Deduplication** (`dedup`): Rolling hash-based content deduplication using xxHash3
- **Similarity** (`similarity`): SIMD-optimized cosine similarity for embeddings
- **JSON Repair** (`json_repair`): Fast JSON repair with streaming parser
- **Hashing** (`hashing`): High-performance signature hashing for loop detection

## Performance

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Block deduplication | 1.0ms | 0.02ms | ~50x |
| Cosine similarity (384d, 100 vectors) | 5.0ms | 1.0ms | ~5x |
| JSON repair | 0.5ms | 0.05ms | ~10x |
| Signature hashing | 0.2ms | 0.02ms | ~10x |

## Building

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Python 3.10+
- maturin (`pip install maturin`)

### Development Build

```bash
cd rust

# Build and install in development mode
maturin develop

# Build with optimizations
maturin develop --release

# Build wheel for distribution
maturin build --release
```

### Install Pre-built Wheel

```bash
# From the rust directory after building
pip install target/wheels/victor_native-*.whl
```

## Usage

The native extensions are automatically used when available:

```python
from victor.processing.native import (
    rolling_hash_blocks,
    batch_cosine_similarity,
    repair_json,
    compute_signature,
    is_native_available,
)

# Check if native extensions are loaded
print(f"Native available: {is_native_available()}")

# Deduplication
blocks = rolling_hash_blocks(content, min_block_length=50)
for hash_str, block, is_duplicate in blocks:
    if is_duplicate:
        print(f"Duplicate block: {block[:50]}...")

# Similarity
query = [0.1, 0.2, ...]  # 384-dim embedding
corpus = [[...], [...], ...]  # List of embeddings
similarities = batch_cosine_similarity(query, corpus)

# JSON repair
fixed = repair_json("{'key': 'value', 'active': True}")
# '{"key": "value", "active": true}'

# Signature hashing
sig = compute_signature("read_file", {"path": "/test.py"})
```

## Fallback Behavior

If the native extension is not available (not installed or incompatible platform), the `victor.processing.native` module automatically falls back to pure Python implementations with equivalent functionality.

```python
from victor.processing.native import is_native_available

if is_native_available():
    print("Using Rust implementation (fast)")
else:
    print("Using Python fallback (compatible)")
```

## Architecture

```
rust/
├── Cargo.toml          # Workspace configuration and shared version
├── pyproject.toml      # maturin build configuration
├── README.md           # This file
└── crates/
    ├── protocol/
    ├── state/
    ├── tools/
    ├── edge-runtime/
    └── python-bindings/
```

## Releasing crates.io packages

Rust workspace releases use their own `rust-vX.Y.Z` tags. Before tagging, update the workspace
version and every internal dependency version in the crate manifests, then run:

```bash
cd rust
cargo fmt --all --check
cargo test --workspace --locked
cargo package -p victor-protocol --locked --no-verify
cargo package -p victor-state --list
cargo package -p victor-tools --list
cargo package -p victor-edge --list
```

For the first release, Cargo cannot fully prepare a dependent archive until its preceding Victor
crate is visible in the crates.io index. The workflow performs that stronger check naturally as it
publishes and verifies each crate in order.

Push an annotated `rust-vX.Y.Z` tag only after the release commit reaches `main`. The
`Publish Rust crates` workflow validates that the tag and workspace versions match, then publishes
the dependency chain. It requires the repository Actions secret `CARGO_REGISTRY_TOKEN`. Re-running
the workflow is safe: crate versions already present on crates.io are skipped and verified.

## Testing

```bash
# Run Rust tests
cargo test

# Run Python integration tests
pytest ../tests/unit/test_native.py -v
```

## License

Apache License 2.0
