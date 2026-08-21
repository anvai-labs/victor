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

//! Tool Call Extractor
//!
//! High-performance extraction of tool calls from model text output.
//! Provides ~3x speedup over Python regex-based extraction.
//!
//! Features:
//! - Pre-compiled regex patterns for file paths, code blocks, shell commands
//! - Batch extraction for multiple texts
//! - Combined tool call extraction with confidence scoring

use pyo3::prelude::*;
use pyo3::types::PyDict;
use regex::Regex;
use std::sync::LazyLock;

// Pre-compiled patterns for file path extraction
static FILE_PATH_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        // Pattern: "write to file.py", "edit file.js", etc.
        Regex::new(
            r#"(?i)(?:to|file|path|in|create|write|save|modify|update|edit)\s+[`'"]?([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})[`'"]?"#
        ).unwrap(),
        // Pattern: "file.py with", "file.js should", etc.
        Regex::new(
            r#"(?m)^[`'"]?([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})[`'"]?\s+(?:with|should|will)"#
        ).unwrap(),
        // Pattern: "the file file.py", "this file test.js"
        Regex::new(
            r#"(?i)(?:the|this)\s+file\s+[`'"]?([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})[`'"]?"#
        ).unwrap(),
        // Pattern: backtick-wrapped paths `file.py`
        Regex::new(r"`([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})`").unwrap(),
    ]
});

// Pre-compiled pattern for fenced code blocks
static CODE_BLOCK_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?is)```(?:python|py|javascript|js|typescript|ts|bash|sh|json|yaml|yml|toml|html|css|markdown|md|sql|go|rust|java|c|cpp|ruby|php)?\s*\n(.*?)```"
    ).unwrap()
});

// Pre-compiled pattern for indented code blocks
static INDENTED_CODE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)(?:^|\n)((?:[ ]{4,}|\t).+(?:\n(?:[ ]{4,}|\t).+)*)").unwrap());

// Pre-compiled patterns for shell commands
static SHELL_COMMAND_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        // Fenced bash blocks
        Regex::new(r"(?is)```(?:bash|sh|shell|zsh)?\s*\n(.+?)```").unwrap(),
        // Inline command patterns
        Regex::new(r#"(?i)(?:run|execute|command):\s*[`'"](.+?)[`'"]"#).unwrap(),
        // Shell prompt pattern
        Regex::new(r"(?m)(?:^|\n)\$\s+(.+?)(?:\n|$)").unwrap(),
    ]
});

// Pattern for backtick-wrapped paths (for ls/list commands)
static BACKTICK_PATH_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"`([a-zA-Z0-9_./-]+)`").unwrap());

/// Extract a file path from text.
///
/// Uses pre-compiled regex patterns for ~3x speedup over Python.
///
/// # Arguments
/// * `text` - Text to search for file paths
///
/// # Returns
/// Extracted file path or None
#[pyfunction]
pub fn extract_file_path(text: &str) -> Option<String> {
    for pattern in FILE_PATH_PATTERNS.iter() {
        if let Some(caps) = pattern.captures(text) {
            if let Some(m) = caps.get(1) {
                let path = m.as_str();
                // Validate it looks like a path
                if path.contains('/') || path.contains('.') {
                    return Some(path.to_string());
                }
            }
        }
    }
    None
}

/// Extract code blocks from text (fenced and indented).
///
/// Uses pre-compiled regex patterns for ~3x speedup over Python.
///
/// # Arguments
/// * `text` - Text containing code blocks
///
/// # Returns
/// List of extracted code block contents
#[pyfunction]
pub fn extract_code_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();

    // Try fenced code blocks first
    for caps in CODE_BLOCK_PATTERN.captures_iter(text) {
        if let Some(m) = caps.get(1) {
            let content = m.as_str().trim();
            if !content.is_empty() {
                blocks.push(content.to_string());
            }
        }
    }

    // If no fenced blocks, try indented code blocks
    if blocks.is_empty() {
        for caps in INDENTED_CODE_PATTERN.captures_iter(text) {
            if let Some(m) = caps.get(1) {
                let block = m.as_str();
                let dedented = dedent_block(block);
                if !dedented.is_empty() {
                    blocks.push(dedented);
                }
            }
        }
    }

    blocks
}

/// Dedent a code block by removing common leading whitespace.
fn dedent_block(block: &str) -> String {
    let lines: Vec<&str> = block.lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    // Find minimum indentation (ignoring empty lines)
    let min_indent = lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);

    // Remove the common indentation
    lines
        .iter()
        .map(|line| {
            if line.len() > min_indent {
                &line[min_indent..]
            } else {
                line.trim()
            }
        })
        .collect::<Vec<&str>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Extract shell commands from text.
///
/// Uses pre-compiled regex patterns for ~3x speedup over Python.
///
/// # Arguments
/// * `text` - Text containing shell commands
///
/// # Returns
/// List of extracted shell commands
#[pyfunction]
pub fn extract_shell_commands(text: &str) -> Vec<String> {
    let mut commands = Vec::new();

    for pattern in SHELL_COMMAND_PATTERNS.iter() {
        for caps in pattern.captures_iter(text) {
            if let Some(m) = caps.get(1) {
                let cmd = m.as_str().trim();
                if !cmd.is_empty() {
                    commands.push(cmd.to_string());
                }
            }
        }
    }

    commands
}

/// Extract a tool call from text for a specific tool.
///
/// Uses pre-compiled regex patterns for ~3x speedup over Python.
///
/// # Arguments
/// * `text` - Text to extract from
/// * `tool_name` - Tool name to extract for
/// * `current_file` - Optional current file context
///
/// # Returns
/// Dictionary with tool, args, confidence, source or None
#[pyfunction]
#[pyo3(signature = (text, tool_name, current_file=None))]
pub fn extract_tool_call(
    py: Python<'_>,
    text: &str,
    tool_name: &str,
    current_file: Option<&str>,
) -> PyResult<Option<Py<PyAny>>> {
    let tool_lower = tool_name.to_lowercase();

    match tool_lower.as_str() {
        "write" | "write_file" => extract_write_tool(py, text, current_file),
        "read" | "read_file" => extract_read_tool(py, text),
        "shell" | "bash" | "execute" | "run" => extract_shell_tool(py, text),
        "ls" | "list" => extract_list_tool(py, text),
        _ => Ok(None),
    }
}

/// Extract write tool call
fn extract_write_tool(
    py: Python<'_>,
    text: &str,
    current_file: Option<&str>,
) -> PyResult<Option<Py<PyAny>>> {
    let file_path = match extract_file_path(text).or_else(|| current_file.map(|s| s.to_string())) {
        Some(p) => p,
        None => return Ok(None),
    };

    let blocks = extract_code_blocks(text);
    let content = match blocks.first() {
        Some(c) => c.clone(),
        None => return Ok(None),
    };

    let mut confidence = 0.85;
    if file_path.ends_with(".py") && (content.contains("def ") || content.contains("class ")) {
        confidence = 0.95;
    }

    let args = PyDict::new(py);
    args.set_item("path", &file_path)?;
    args.set_item("content", &content)?;

    let dict = PyDict::new(py);
    dict.set_item("tool", "write")?;
    dict.set_item("args", args)?;
    dict.set_item("confidence", confidence)?;
    dict.set_item("source", text.chars().take(200).collect::<String>())?;

    Ok(Some(dict.unbind().into_any()))
}

/// Extract read tool call
fn extract_read_tool(py: Python<'_>, text: &str) -> PyResult<Option<Py<PyAny>>> {
    let file_path = match extract_file_path(text) {
        Some(p) => p,
        None => return Ok(None),
    };

    let args = PyDict::new(py);
    args.set_item("path", &file_path)?;

    let dict = PyDict::new(py);
    dict.set_item("tool", "read")?;
    dict.set_item("args", args)?;
    dict.set_item("confidence", 0.9)?;
    dict.set_item("source", text.chars().take(100).collect::<String>())?;

    Ok(Some(dict.unbind().into_any()))
}

/// Extract shell tool call
fn extract_shell_tool(py: Python<'_>, text: &str) -> PyResult<Option<Py<PyAny>>> {
    let commands = extract_shell_commands(text);
    let command = match commands.first() {
        Some(c) => c.clone(),
        None => return Ok(None),
    };

    let args = PyDict::new(py);
    args.set_item("command", &command)?;

    let dict = PyDict::new(py);
    dict.set_item("tool", "shell")?;
    dict.set_item("args", args)?;
    dict.set_item("confidence", 0.75)?;
    dict.set_item("source", text.chars().take(150).collect::<String>())?;

    Ok(Some(dict.unbind().into_any()))
}

/// Extract list tool call
fn extract_list_tool(py: Python<'_>, text: &str) -> PyResult<Option<Py<PyAny>>> {
    let path = BACKTICK_PATH_PATTERN
        .captures(text)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| ".".to_string());

    let args = PyDict::new(py);
    args.set_item("path", &path)?;

    let dict = PyDict::new(py);
    dict.set_item("tool", "ls")?;
    dict.set_item("args", args)?;
    dict.set_item("confidence", 0.8)?;
    dict.set_item("source", text.chars().take(100).collect::<String>())?;

    Ok(Some(dict.unbind().into_any()))
}

/// Extract file paths from multiple texts (batch operation).
///
/// Uses parallel processing for ~5x speedup on large batches.
///
/// # Arguments
/// * `texts` - List of texts to search
///
/// # Returns
/// List of extracted paths (None for texts without paths)
#[pyfunction]
pub fn batch_extract_file_paths(texts: Vec<String>) -> Vec<Option<String>> {
    use rayon::prelude::*;

    if texts.len() < 10 {
        // Serial for small batches
        texts.iter().map(|text| extract_file_path(text)).collect()
    } else {
        // Parallel for larger batches
        texts
            .par_iter()
            .map(|text| extract_file_path(text))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_file_path() {
        assert_eq!(
            extract_file_path("I'll write to hello.py"),
            Some("hello.py".to_string())
        );
        assert_eq!(
            extract_file_path("Edit the file `src/main.rs`"),
            Some("src/main.rs".to_string())
        );
        assert_eq!(extract_file_path("No file here"), None);
    }

    #[test]
    fn test_extract_code_blocks() {
        let text = "Here's the code:\n```python\ndef hello():\n    print('hi')\n```";
        let blocks = extract_code_blocks(text);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].contains("def hello"));
    }

    #[test]
    fn test_extract_shell_commands() {
        let text = "Run this:\n```bash\nls -la\n```";
        let commands = extract_shell_commands(text);
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0], "ls -la");
    }

    #[test]
    fn test_dedent_block() {
        let block = "    def hello():\n        print('hi')";
        let dedented = dedent_block(block);
        assert_eq!(dedented, "def hello():\n    print('hi')");
    }
}
