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

//! Fast regex processing using Rust's regex crate.
//!
//! The Rust regex crate is significantly faster than Python's re module
//! for complex patterns and large texts.

use pyo3::prelude::*;
use regex::Regex;

/// Fast regex processor using Rust's regex crate.
#[pyclass]
pub struct FastRegexProcessor {
    priority: i32,
    cache: std::sync::Arc<std::sync::RwLock<std::collections::HashMap<String, Regex>>>,
}

#[pymethods]
impl FastRegexProcessor {
    /// Create a new FastRegexProcessor.
    #[new]
    #[pyo3(signature = (config=None, priority=None))]
    fn new(config: Option<PyObject>, priority: Option<i32>) -> PyResult<Self> {
        // Extract priority from config if provided
        let final_priority = if let Some(cfg) = config {
            if let Some(p) = priority {
                p
            } else {
                Python::with_gil(|py| {
                    if let Ok(obj) = cfg.getattr(py, "priority") {
                        obj.extract::<i32>(py).unwrap_or(80)
                    } else {
                        80
                    }
                })
            }
        } else {
            priority.unwrap_or(80)
        };

        Ok(Self {
            priority: final_priority,
            cache: std::sync::Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Get the priority of this processor.
    #[getter]
    fn get_priority(&self) -> i32 {
        self.priority
    }

    /// Find all matches of a pattern in text.
    ///
    /// Args:
    ///     pattern: Regex pattern string
    ///     text: Text to search
    ///
    /// Returns:
    ///     List of matched strings
    #[pyo3(signature = (pattern, text))]
    fn findall(&self, pattern: &str, text: &str) -> PyResult<PyObject> {
        let regex = self.get_or_compile(pattern)?;
        let matches: Vec<String> = regex.find_iter(text).map(|m| m.as_str().to_string()).collect();

        Python::with_gil(|py| {
            let list = pyo3::types::PyList::new(py, matches);
            Ok(list.to_object(py))
        })
    }

    /// Find all matches as an iterator.
    ///
    /// Args:
    ///     pattern: Regex pattern string
    ///     text: Text to search
    ///
    /// Returns:
    ///     List of match objects (as dicts with start, end, text)
    #[pyo3(signature = (pattern, text))]
    fn finditer(&self, pattern: &str, text: &str) -> PyResult<PyObject> {
        let regex = self.get_or_compile(pattern)?;
        let matches: Vec<(usize, usize, String)> = regex
            .find_iter(text)
            .map(|m| (m.start(), m.end(), m.as_str().to_string()))
            .collect();

        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty(py);
            for (start, end, matched) in matches {
                let match_dict = pyo3::types::PyDict::new(py);
                match_dict.set_item("start", start)?;
                match_dict.set_item("end", end)?;
                match_dict.set_item("text", matched)?;
                list.append(match_dict)?;
            }
            Ok(list.to_object(py))
        })
    }

    /// Match pattern at the start of text.
    ///
    /// Args:
    ///     pattern: Regex pattern string
    ///     text: Text to match
    ///
    /// Returns:
    ///     Match object or None
    #[pyo3(signature = (pattern, text))]
    fn match_pattern(&self, pattern: &str, text: &str) -> PyResult<PyObject> {
        let regex = self.get_or_compile(pattern)?;

        Python::with_gil(|py| {
            if let Some(m) = regex.find(text) {
                let match_dict = pyo3::types::PyDict::new(py);
                match_dict.set_item("start", m.start())?;
                match_dict.set_item("end", m.end())?;
                match_dict.set_item("text", m.as_str())?;
                Ok(Some(match_dict.to_object(py)).to_object(py))
            } else {
                Ok(py.None())
            }
        })
    }

    /// Match pattern at the start of text (alias for match_pattern).
    ///
    /// This provides compatibility with Python's re.match() naming.
    ///
    /// Args:
    ///     pattern: Regex pattern string
    ///     text: Text to match
    ///
    /// Returns:
    ///     Match object or None
    #[pyo3(signature = (pattern, text))]
    fn r#match(&self, pattern: &str, text: &str) -> PyResult<PyObject> {
        self.match_pattern(pattern, text)
    }

    /// Substitute pattern with replacement.
    ///
    /// Args:
    ///     pattern: Regex pattern string
    ///     replacement: Replacement string
    ///     text: Text to process
    ///
    /// Returns:
    ///     Text with replacements applied
    #[pyo3(signature = (pattern, replacement, text))]
    fn sub(&self, pattern: &str, replacement: &str, text: &str) -> PyResult<String> {
        let regex = self.get_or_compile(pattern)?;
        Ok(regex.replace_all(text, replacement).to_string())
    }

    /// Check if this is a native implementation.
    fn is_native(&self) -> bool {
        true
    }
}

impl FastRegexProcessor {
    /// Get a compiled regex from cache or compile and cache it.
    fn get_or_compile(&self, pattern: &str) -> PyResult<Regex> {
        // Try to get from cache
        {
            let cache = self.cache.read().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Cache lock error: {}", e))
            })?;

            if let Some(regex) = cache.get(pattern) {
                return Ok(regex.clone());
            }
        }

        // Compile and cache
        let regex = Regex::new(pattern).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid regex: {}", e))
        })?;

        {
            let mut cache = self.cache.write().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Cache lock error: {}", e))
            })?;
            cache.insert(pattern.to_string(), regex.clone());
        }

        Ok(regex)
    }
}
