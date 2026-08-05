// Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Native performance extensions for victor-coding.
//!
//! This crate provides high-performance Rust implementations of:
//! - Code chunking with overlap
//! - Regex pattern matching
//!
//! Note: Tree-sitter extraction is NOT included here because Python's
//! tree-sitter bindings already provide efficient access to the C library.
//! Adding Rust would introduce FFI overhead (Python → Rust → C) without benefit.
//!
//! # Architecture
//!
//! The crate is organized into modules:
//! - `chunker`: Code chunking with semantic awareness
//! - `regex_utils`: Fast regex processing
//! - `python`: PyO3 bindings for Python integration
//!
//! # Usage
//!
//! The crate is compiled as a Python extension module via PyO3.
//! Install with: `pip install victor-ai[native]`

use pyo3::prelude::*;

mod chunker;
mod regex_utils;
mod python;

// Re-export Python classes and functions
pub use chunker::detect_language;
pub use python::{FastChunker, FastRegexProcessor};

/// Version information for the native extension.
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Check if native extensions are available.
#[pyfunction]
fn is_native() -> bool {
    true
}

/// Get native extension capabilities.
#[pyfunction]
fn get_capabilities() -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let caps = pyo3::types::PyDict::new(py);
        caps.set_item("zero_copy_chunking", true)?;
        caps.set_item("fast_regex", true)?;
        caps.set_item("query_caching", true)?;
        Ok(caps.to_object(py))
    })
}

/// Python module definition
#[pymodule]
fn _native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(is_native, m)?)?;
    m.add_function(wrap_pyfunction!(get_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(detect_language, m)?)?;
    m.add_class::<FastChunker>()?;
    m.add_class::<FastRegexProcessor>()?;
    Ok(())
}
