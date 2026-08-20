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

//! Response Sanitizer
//!
//! High-performance sanitization of model responses.
//! Provides ~3x speedup over Python regex-based sanitization.
//!
//! Features:
//! - Pre-compiled regex patterns for garbage detection
//! - Training data leakage pattern detection
//! - Markup stripping
//! - Tool name validation

use pyo3::prelude::*;
use regex::Regex;
use std::sync::LazyLock;

// Pre-compiled leakage patterns
static LEAKAGE_PATTERNS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    vec![
        (
            Regex::new(r"(?i)Do not invent any new (?:or additional )?parameters.*").unwrap(),
            "no_new_params",
        ),
        (
            Regex::new(r"(?i)The parameter value should be passed as a string.*").unwrap(),
            "string_params",
        ),
        (
            Regex::new(r"(?i)If you want to call multiple functions.*").unwrap(),
            "multiple_funcs",
        ),
        (
            Regex::new(r"(?i)Do NOT surround the function call.*").unwrap(),
            "no_surround",
        ),
        (
            Regex::new(r"(?i)All parameters are required unless.*").unwrap(),
            "required_params",
        ),
        (
            Regex::new(r"(?i)The agent is not allowed to directly access.*").unwrap(),
            "no_direct_access",
        ),
        (
            Regex::new(r"(?i)Begin by calling list_directory.*").unwrap(),
            "begin_list_dir",
        ),
    ]
});

// Pre-compiled garbage detection patterns
static GARBAGE_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"FUNCTION_CALL\s*\{").unwrap(),
        Regex::new(r"</function>\s*</function>").unwrap(),
        Regex::new(r"<parameter[^>]*>").unwrap(),
        Regex::new(r#"(?m)^\s*\{\s*"name":\s*"[^"]+",\s*"arguments":"#).unwrap(),
        Regex::new(r"(?m)^\s*<IMPORTANT>").unwrap(),
        Regex::new(r"(?m)^\s*Do NOT").unwrap(),
        Regex::new(r"(?m)^\s*NEVER\s+").unwrap(),
        Regex::new(r"\[TOOL_REQUEST\]").unwrap(),
        Regex::new(r"\[END_TOOL_REQUEST\]").unwrap(),
    ]
});

// Pre-compiled cleanup patterns (pattern, replacement)
static CLEANUP_PATTERNS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    vec![
        // Repeated closing tags
        (Regex::new(r"(</\w+>\s*){3,}").unwrap(), ""),
        // Function tags
        (Regex::new(r"</?function[^>]*>").unwrap(), ""),
        // Parameter tags
        (Regex::new(r"</?parameter[^>]*>").unwrap(), ""),
        // Tool tags
        (Regex::new(r"</?tool[^>]*>").unwrap(), ""),
        // Important tags
        (Regex::new(r"</?IMPORTANT[^>]*>").unwrap(), ""),
        // JSON tool calls
        (
            Regex::new(r#"\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\}"#).unwrap(),
            "",
        ),
        // Excessive newlines
        (Regex::new(r"\n{4,}").unwrap(), "\n\n\n"),
    ]
});

// Thinking token patterns (from streaming_filter)
static THINKING_TOKENS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    vec![
        "<｜begin▁of▁thinking｜>",
        "<｜end▁of▁thinking｜>",
        "<|begin_of_thinking|>",
        "<|end_of_thinking|>",
        "<think>",
        "</think>",
    ]
});

// Markup pattern
static MARKUP_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").unwrap());

// Invalid tool name patterns
static INVALID_TOOL_PREFIXES: &[&str] = &[
    "example_",
    "func_",
    "function_",
    "tool_name",
    "my_",
    "test_tool",
    "sample_",
];

// Valid tool name pattern
static VALID_TOOL_NAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-zA-Z][a-zA-Z0-9_]*$").unwrap());

/// Sanitize model response by removing malformed patterns.
///
/// Uses pre-compiled regex patterns for ~3x speedup over Python.
///
/// # Arguments
/// * `text` - Raw response text from the model
///
/// # Returns
/// Cleaned text suitable for display
#[pyfunction]
pub fn sanitize_response(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    let mut result = text.to_string();

    // Apply cleanup patterns
    for (pattern, replacement) in CLEANUP_PATTERNS.iter() {
        result = pattern.replace_all(&result, *replacement).to_string();
    }

    // Strip thinking tokens
    for token in THINKING_TOKENS.iter() {
        result = result.replace(token, "");
    }

    // Remove leakage patterns
    for (pattern, _) in LEAKAGE_PATTERNS.iter() {
        result = pattern.replace_all(&result, "").to_string();
    }

    // Remove lines that are just tool call syntax
    let lines: Vec<&str> = result
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.starts_with("{\"name\":") && !trimmed.starts_with("</")
        })
        .collect();
    result = lines.join("\n");

    // Remove parameter= lines
    let lines: Vec<&str> = result
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.starts_with("parameter=") && !trimmed.starts_with("<parameter")
        })
        .collect();
    result = lines.join("\n");

    result.trim().to_string()
}

/// Detect if content is garbage/malformed output.
///
/// Uses pre-compiled regex patterns for ~3x speedup over Python.
///
/// # Arguments
/// * `content` - Content to check
///
/// # Returns
/// True if content appears to be garbage
#[pyfunction]
pub fn is_garbage_content(content: &str) -> bool {
    if content.is_empty() {
        return false;
    }

    for pattern in GARBAGE_PATTERNS.iter() {
        if pattern.is_match(content) {
            return true;
        }
    }
    false
}

/// Detect training data leakage patterns in text.
///
/// Uses pre-compiled regex patterns for ~3x speedup over Python.
///
/// # Arguments
/// * `text` - Text to check for leakage
///
/// # Returns
/// List of (start, end, pattern_name) tuples for matches
#[pyfunction]
pub fn detect_leakage_patterns(text: &str) -> Vec<(usize, usize, String)> {
    let mut matches = Vec::new();

    for (pattern, name) in LEAKAGE_PATTERNS.iter() {
        for m in pattern.find_iter(text) {
            matches.push((m.start(), m.end(), (*name).to_string()));
        }
    }

    // Sort by start position
    matches.sort_by_key(|(start, _, _)| *start);
    matches
}

/// Remove XML/HTML-like tags to salvage plain text.
///
/// Uses pre-compiled regex for ~3x speedup over Python.
///
/// # Arguments
/// * `text` - Text potentially containing markup
///
/// # Returns
/// Plain text with markup removed
#[pyfunction]
pub fn strip_markup(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    let cleaned = MARKUP_PATTERN.replace_all(text, " ");
    // Collapse whitespace
    cleaned.split_whitespace().collect::<Vec<&str>>().join(" ")
}

/// Validate a tool name is not a hallucination.
///
/// Uses pre-compiled patterns for ~3x speedup over Python.
///
/// # Arguments
/// * `name` - Tool name to validate
///
/// # Returns
/// Tuple of (is_valid, rejection_reason)
#[pyfunction]
pub fn validate_tool_name(name: &str) -> (bool, Option<String>) {
    if name.is_empty() {
        return (false, Some("empty_or_invalid_type".to_string()));
    }

    // Check invalid prefixes
    for prefix in INVALID_TOOL_PREFIXES.iter() {
        if name.starts_with(prefix) {
            return (false, Some(format!("invalid_prefix:{}", prefix)));
        }
    }

    // Check invalid suffixes
    if name.ends_with('/') || name.ends_with('>') {
        return (false, Some("invalid_suffix".to_string()));
    }

    // Check starts with tag
    if name.starts_with('<') {
        return (false, Some("starts_with_tag".to_string()));
    }

    // Check for whitespace
    if name.contains(' ') || name.contains('\t') {
        return (false, Some("contains_whitespace".to_string()));
    }

    // Check starts with number
    if name
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
    {
        return (false, Some("starts_with_number".to_string()));
    }

    // Check valid format
    if !VALID_TOOL_NAME.is_match(name) {
        return (false, Some("invalid_characters".to_string()));
    }

    (true, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_response() {
        let text = "<function>test</function> Hello";
        let result = sanitize_response(text);
        assert_eq!(result, "test Hello");
    }

    #[test]
    fn test_is_garbage_content() {
        assert!(is_garbage_content("FUNCTION_CALL { something }"));
        assert!(is_garbage_content("[TOOL_REQUEST]"));
        assert!(is_garbage_content("[END_TOOL_REQUEST]"));
        assert!(!is_garbage_content("Normal content"));
    }

    #[test]
    fn test_strip_markup() {
        let text = "<div>Hello</div> <span>World</span>";
        let result = strip_markup(text);
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_validate_tool_name() {
        assert_eq!(validate_tool_name("read_file"), (true, None));
        assert!(matches!(
            validate_tool_name("example_tool"),
            (false, Some(_))
        ));
        assert!(matches!(validate_tool_name("123tool"), (false, Some(_))));
        assert!(matches!(validate_tool_name("tool name"), (false, Some(_))));
    }

    #[test]
    fn test_detect_leakage_patterns() {
        let text = "Do not invent any new parameters here.";
        let matches = detect_leakage_patterns(text);
        assert!(!matches.is_empty());
    }
}
