# Native Performance Extensions

This directory contains Rust-based performance extensions for victor-coding.

## Overview

The native extensions provide high-performance implementations of:
- **Code chunking**: Zero-allocation sliding window with semantic awareness
- **Regex processing**: Fast pattern matching using Rust's regex crate

## What's NOT Included

**Tree-sitter symbol extraction** is intentionally **NOT** included because:
- Python's `tree-sitter` package already provides efficient bindings to the C library
- Adding Rust (Python → Rust → C) would introduce FFI overhead without benefit
- The existing Python bindings are well-optimized for this use case

For tree-sitter operations, use Python's `tree-sitter` package directly.

## Performance Improvements

Expected speedups compared to pure Python:
- Code chunking: **2-3x faster**
- Regex processing: **2-4x faster**

## Build and installation

This directory builds the separate `victor-coding-native` distribution, whose
Python extension is `_native`. Install `victor-coding` as well; the native wheel
does not replace that package. It is distinct from the framework's `victor_native`
extension in the root `rust/` directory.

Use Python 3.11 or newer, a current stable Rust toolchain and `maturin>=1.10,<2`.
From the monorepo root:

```bash
python -m pip install 'maturin>=1.10,<2'
cd verticals/victor-coding/native
maturin build --release --locked --out dist
python -m pip install dist/victor_coding_native-*.whl
```

For an activated development environment, run `maturin develop --release` from
this directory. The native `pyproject.toml` prevents maturin from inheriting the
parent package identity. The historical empty `victor-coding[native]` extra does
not build or install this extension.

## Usage

The native extensions are automatically used when available. No code changes required.

### Check availability
```python
from victor_coding.performance import BackendFactory

if BackendFactory.has_native_indexer():
    print("Native indexer available!")

if BackendFactory.has_native_chunker():
    print("Native chunker available!")
```

### Explicit usage
```python
from victor_coding.performance import BackendFactory

# Create indexer (uses native if available)
indexer = BackendFactory.create_indexer(config)

# Check if native backend is being used
if indexer.is_native:
    print("Using native backend for 3-5x speedup!")
```

### Disable native backends
```python
import os
os.environ["VICTOR_USE_NATIVE_BACKENDS"] = "0"

# Or in settings
settings.use_native_backends = False
```

## Development

### Project structure
```
native/
├── Cargo.toml           # Rust project configuration
├── src/
│   ├── lib.rs          # Python module definition
│   ├── extractor.rs    # Tree-sitter symbol extraction
│   ├── chunker.rs      # Code chunking
│   ├── regex_utils.rs  # Regex processing
│   └── python.rs       # Re-exports for Python
└── README.md           # This file
```

### Building specific targets
```bash
# Debug build (faster compilation)
maturin develop

# Release build (optimized)
maturin develop --release

# For specific Python version
maturin develop --python3.11

# Strip symbols for smaller binary
maturin develop --release --strip
```

### Running tests
```bash
# Python tests
pytest tests/performance/

# Rust tests
cargo test

# Benchmarking
pytest tests/performance/ --benchmark
```

## Configuration

### Environment variables
- `VICTOR_USE_NATIVE_BACKENDS=1` - Enable native backends (default: 1)
- `VICTOR_NATIVE_BACKEND_PRIORITY=80` - Priority for native backends (0-100)
- `VICTOR_MAX_PARALLEL_WORKERS=4` - Max parallel workers (default: auto-detect)
- `VICTOR_LOG_PERFORMANCE=0` - Enable performance logging (default: 0)

### Settings file
```toml
[performance]
use_native_backends = true
prefer_native_indexer = true
prefer_native_chunker = true
prefer_native_regex = true
native_backend_priority = 80
```

## Troubleshooting

### Build failures
```bash
# Ensure Rust toolchain is up to date
rustup update

# Clean and rebuild
cargo clean
maturin develop --release
```

### Import errors
```bash
# Check if extension was built
python -c "from victor_coding.native import NATIVE_AVAILABLE; print(NATIVE_AVAILABLE)"

# Force rebuild
maturin develop --release --strip
```

### Performance not improving
```bash
# Check which backend is being used
python -c "from victor_coding.performance import BackendFactory; print(BackendFactory.has_native_indexer())"

# Enable logging
export VICTOR_LOG_PERFORMANCE=1
```

## License

Apache-2.0

## Contributing

When modifying the Rust code:
1. Add tests in `tests/performance/`
2. Benchmark before/after changes
3. Ensure pure Python fallback still works
4. Update this README if API changes
