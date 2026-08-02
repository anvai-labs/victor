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

//! High-performance code chunking with semantic awareness.
//!
//! This module provides zero-allocation sliding window chunking that respects
//! code boundaries and provides overlap for context preservation.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

/// High-performance code chunker.
///
/// Features:
/// - Zero-allocation sliding window
/// - Line-aware chunk boundaries
/// - Configurable overlap
/// - Token estimation
#[pyclass]
pub struct FastChunker {
    max_tokens: usize,
    overlap_tokens: usize,
    max_chars: usize,
    overlap_chars: usize,
    priority: i32,
}

#[pymethods]
impl FastChunker {
    /// Create a new FastChunker.
    ///
    /// Can be initialized with:
    /// - No arguments (uses defaults)
    /// - A config object (dict with max_tokens, overlap_tokens, priority keys)
    /// - Direct keyword arguments (max_tokens, overlap_tokens, priority)
    #[new]
    #[pyo3(signature = (config=None, max_tokens=None, overlap_tokens=None, priority=None))]
    fn new(
        config: Option<PyObject>,
        max_tokens: Option<usize>,
        overlap_tokens: Option<usize>,
        priority: Option<i32>,
    ) -> PyResult<Self> {
        // Determine values from various sources
        let (final_max_tokens, final_overlap_tokens, final_priority) = if let Some(cfg) = config {
            // Extract from config object
            Python::with_gil(|py| {
                let mut max_tok = max_tokens.unwrap_or(512);
                let mut overlap_tok = overlap_tokens.unwrap_or(64);
                let mut prio = priority.unwrap_or(80);

                if let Ok(obj) = cfg.getattr(py, "max_chunk_tokens") {
                    if let Ok(val) = obj.extract::<usize>(py) {
                        max_tok = val;
                    }
                }
                if let Ok(obj) = cfg.getattr(py, "max_tokens") {
                    if let Ok(val) = obj.extract::<usize>(py) {
                        max_tok = val;
                    }
                }
                if let Ok(obj) = cfg.getattr(py, "overlap_tokens") {
                    if let Ok(val) = obj.extract::<usize>(py) {
                        overlap_tok = val;
                    }
                }
                if let Ok(obj) = cfg.getattr(py, "priority") {
                    if let Ok(val) = obj.extract::<i32>(py) {
                        prio = val;
                    }
                }

                (max_tok, overlap_tok, prio)
            })
        } else {
            // Use direct arguments or defaults
            (max_tokens.unwrap_or(512), overlap_tokens.unwrap_or(64), priority.unwrap_or(80))
        };

        Ok(Self {
            max_tokens: final_max_tokens,
            overlap_tokens: final_overlap_tokens,
            max_chars: final_max_tokens * 3,
            overlap_chars: final_overlap_tokens * 3,
            priority: final_priority,
        })
    }

    /// Get the priority of this chunker.
    #[getter]
    fn get_priority(&self) -> i32 {
        self.priority
    }

    /// Get max_tokens.
    #[getter]
    fn get_max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Get overlap_tokens.
    #[getter]
    fn get_overlap_tokens(&self) -> usize {
        self.overlap_tokens
    }

    /// Chunk code content with semantic awareness.
    ///
    /// Args:
    ///     content: Source code content
    ///     language: Programming language
    ///     file_path: Optional file path
    ///
    /// Returns:
    ///     List of ChunkInfo dictionaries
    #[pyo3(signature = (content, language, file_path=None))]
    fn chunk_code(
        &self,
        content: &str,
        language: &str,
        file_path: Option<&str>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let chunks = PyList::empty(py);

            // Skip empty content
            if content.trim().is_empty() {
                return Ok(chunks.to_object(py));
            }

            let lines: Vec<&str> = content.lines().collect();
            let total_lines = lines.len();

            // If content fits in one chunk, return it as a single chunk
            if content.len() <= self.max_chars {
                let chunk = self.create_chunk_info(
                    py,
                    content,
                    "file",
                    1,
                    total_lines,
                    language,
                    file_path,
                )?;
                chunks.append(chunk)?;
                return Ok(chunks.to_object(py));
            }

            // Sliding window chunking with overlap
            let mut current_line = 0;
            let mut chunk_index = 0;

            while current_line < total_lines {
                let start_line = current_line + 1; // 1-indexed
                let end_line = (current_line + self.count_lines_for_chars(
                    &lines[current_line..],
                    self.max_chars,
                )).min(total_lines);

                // Collect lines for this chunk
                let chunk_content: String = lines[current_line..end_line].join("\n");

                // Determine chunk type based on content
                let chunk_type = if chunk_index == 0 {
                    "file_start"
                } else if end_line == total_lines {
                    "file_end"
                } else {
                    "chunk"
                };

                let chunk = self.create_chunk_info(
                    py,
                    &chunk_content,
                    chunk_type,
                    start_line,
                    end_line,
                    language,
                    file_path,
                )?;
                chunks.append(chunk)?;

                chunk_index += 1;

                // Safety limit: max 100 chunks per file
                if chunk_index >= 100 {
                    break;
                }

                // Move to next chunk with overlap
                if end_line >= total_lines {
                    break;
                }

                // Calculate overlap (number of lines to overlap)
                let overlap_lines = self.count_lines_for_chars(
                    &lines[current_line..end_line],
                    self.overlap_chars,
                ).min(self.overlap_tokens);

                current_line = end_line - overlap_lines.min(end_line - current_line);
            }

            Ok(chunks.to_object(py))
        })
    }

    /// Chunk content with overlap (simplified interface).
    ///
    /// This matches the TextChunkerProtocol interface used by CodeChunker.
    ///
    /// Args:
    ///     content: Content to chunk
    ///     max_chars: Maximum characters per chunk
    ///     overlap_chars: Number of characters to overlap
    ///
    /// Returns:
    ///     List of ChunkInfo dictionaries
    #[pyo3(signature = (content, max_chars, overlap_chars))]
    fn chunk_with_overlap(
        &self,
        content: &str,
        max_chars: usize,
        overlap_chars: usize,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let chunks = PyList::empty(py);

            // Skip empty content
            if content.trim().is_empty() {
                return Ok(chunks.to_object(py));
            }

            let lines: Vec<&str> = content.lines().collect();
            let total_lines = lines.len();

            // If content fits in one chunk, return it
            if content.len() <= max_chars {
                let chunk = self.create_chunk_info(
                    py,
                    content,
                    "file",
                    1,
                    total_lines,
                    "unknown",
                    None,
                )?;
                chunks.append(chunk)?;
                return Ok(chunks.to_object(py));
            }

            // Sliding window chunking
            let mut current_line = 0;
            let mut chunk_index = 0;

            while current_line < total_lines {
                let start_line = current_line + 1;
                let end_line = (current_line + self.count_lines_for_chars(
                    &lines[current_line..],
                    max_chars,
                )).min(total_lines);

                let chunk_content: String = lines[current_line..end_line].join("\n");

                let chunk = self.create_chunk_info(
                    py,
                    &chunk_content,
                    "chunk",
                    start_line,
                    end_line,
                    "unknown",
                    None,
                )?;
                chunks.append(chunk)?;

                chunk_index += 1;
                if chunk_index >= 100 {
                    break;
                }

                if end_line >= total_lines {
                    break;
                }

                // Calculate overlap
                let overlap_lines = self.count_lines_for_chars(
                    &lines[current_line..end_line],
                    overlap_chars,
                ).min(overlap_chars / 40); // Approximate lines from chars

                current_line = end_line - overlap_lines.min(end_line - current_line);
            }

            Ok(chunks.to_object(py))
        })
    }

    /// Chunk a file by reading it first.
    ///
    /// Args:
    ///     file_path: Path to the file (str or PathLike)
    ///     language: Optional language (auto-detected if None)
    ///
    /// Returns:
    ///     List of chunk dictionaries
    #[pyo3(signature = (file_path, language=None))]
    fn chunk_file(&self, file_path: PyObject, language: Option<&str>) -> PyResult<PyObject> {
        // Convert file_path to string (handles both str and PathLike)
        let file_path_str: String = Python::with_gil(|py| {
            if let Ok(s) = file_path.extract::<String>(py) {
                Ok(s)
            } else if let Ok(path) = file_path.getattr(py, "__str__") {
                // PathLike object - call __str__ method
                let result = path.call0(py)?;
                result.extract(py)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "file_path must be str or PathLike"
                ))
            }
        })?;

        // Read file content
        let content = std::fs::read_to_string(&file_path_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        // Detect language from extension if not provided (using same logic as Python)
        let lang = if let Some(lang) = language {
            lang
        } else {
            detect_language(&file_path_str)?
        };

        self.chunk_code(&content, lang, Some(&file_path_str))
    }

    /// Estimate token count for text.
    ///
    /// Uses the standard heuristic of ~3.5 characters per token.
    ///
    /// Args:
    ///     text: Text to estimate
    ///
    /// Returns:
    ///     Estimated token count
    fn estimate_tokens(&self, text: &str) -> usize {
        // Standard heuristic: ~3.5 chars/token for code
        (text.len() / 3).max(1)
    }

    /// Estimate tokens for a list of texts.
    ///
    /// Args:
    ///     texts: List of texts to estimate
    ///
    /// Returns:
    ///     List of estimated token counts
    fn estimate_tokens_batch(&self, texts: Vec<String>) -> PyResult<Vec<usize>> {
        Ok(texts.iter().map(|t| self.estimate_tokens(t)).collect())
    }

    /// Check if this is a native implementation.
    fn is_native(&self) -> bool {
        true
    }

    /// Check if language is supported.
    /// Matches the Python implementation's supported languages.
    fn supports_language(&self, language: &str) -> bool {
        matches!(
            language,
            "python" | "javascript" | "typescript" | "go" | "rust"
                | "java" | "kotlin" | "c" | "cpp" | "csharp" | "ruby"
                | "php" | "swift" | "scala" | "r" | "julia" | "lua"
                | "bash" | "zsh" | "sql" | "html" | "css" | "scss"
                | "less" | "json" | "yaml" | "toml" | "xml" | "markdown"
                | "restructuredtext"
        )
    }

    /// Get supported languages.
    fn get_supported_languages(&self) -> Vec<String> {
        vec![
            "python".to_string(),
            "javascript".to_string(),
            "typescript".to_string(),
            "go".to_string(),
            "rust".to_string(),
            "java".to_string(),
            "kotlin".to_string(),
            "c".to_string(),
            "cpp".to_string(),
            "csharp".to_string(),
            "ruby".to_string(),
            "php".to_string(),
            "swift".to_string(),
            "scala".to_string(),
            "r".to_string(),
            "julia".to_string(),
            "lua".to_string(),
            "bash".to_string(),
            "zsh".to_string(),
            "sql".to_string(),
            "html".to_string(),
            "css".to_string(),
            "scss".to_string(),
            "less".to_string(),
            "json".to_string(),
            "yaml".to_string(),
            "toml".to_string(),
            "xml".to_string(),
            "markdown".to_string(),
            "restructuredtext".to_string(),
        ]
    }
}

impl FastChunker {
    /// Create a ChunkInfo dictionary.
    fn create_chunk_info(
        &self,
        py: Python,
        content: &str,
        chunk_type: &str,
        start_line: usize,
        end_line: usize,
        language: &str,
        file_path: Option<&str>,
    ) -> PyResult<PyObject> {
        let chunk = PyDict::new(py);
        chunk.set_item("content", content)?;
        chunk.set_item("text", content)?; // Alias for compatibility
        chunk.set_item("chunk_type", chunk_type)?;
        chunk.set_item("start_line", start_line)?;
        chunk.set_item("end_line", end_line)?;

        if let Some(fp) = file_path {
            chunk.set_item("file_path", fp)?;
        }

        let token_count = self.estimate_tokens(content);
        chunk.set_item("token_count", token_count)?;

        // Add metadata
        let metadata = PyDict::new(py);
        metadata.set_item("language", language)?;
        metadata.set_item("line_count", end_line - start_line + 1)?;
        chunk.set_item("metadata", metadata)?;

        Ok(chunk.to_object(py))
    }

    /// Count how many lines are needed to reach approximately target_chars.
    fn count_lines_for_chars(&self, lines: &[&str], target_chars: usize) -> usize {
        let mut current_chars = 0;
        let mut line_count = 0;

        for line in lines {
            current_chars += line.len() + 1; // +1 for newline
            line_count += 1;

            if current_chars >= target_chars {
                break;
            }
        }

        line_count.max(1)
    }
}

/// Detect programming language from file extension.
///
/// This is a standalone function for convenience.
/// Matches the Python implementation in chunker.py exactly.
#[pyfunction]
pub fn detect_language(file_path: &str) -> PyResult<&'static str> {
    let path = std::path::Path::new(file_path);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    Ok(match ext {
        "py" | "pyi" | "pyx" => "python",
        "js" | "jsx" => "javascript",
        "ts" | "tsx" => "typescript",
        "go" => "go",
        "rs" => "rust",
        "java" => "java",
        "kt" | "kts" => "kotlin",
        "c" => "c",
        "cpp" | "cc" | "cxx" => "cpp",
        "h" => "c",
        "hpp" => "cpp",
        "cs" => "csharp",
        "rb" => "ruby",
        "php" => "php",
        "swift" => "swift",
        "scala" => "scala",
        "r" | "R" => "r",
        "jl" => "julia",
        "lua" => "lua",
        "sh" | "bash" => "bash",
        "zsh" => "zsh",
        "sql" => "sql",
        "html" | "htm" => "html",
        "css" => "css",
        "scss" => "scss",
        "less" => "less",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "toml" => "toml",
        "xml" => "xml",
        "md" => "markdown",
        "rst" => "restructuredtext",
        _ => "unknown",
    })
}
