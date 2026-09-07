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

//! High-performance streaming content filter for thinking tokens.
//!
//! This module provides fast detection and filtering of LLM thinking tokens
//! during streaming responses. Uses Aho-Corasick for O(n) multi-pattern
//! matching instead of O(n * patterns) sequential regex searches.
//!
//! Supported thinking patterns:
//! - DeepSeek: `<｜begin▁of▁thinking｜>` / `<｜end▁of▁thinking｜>`
//! - DeepSeek ASCII: `<|begin_of_thinking|>` / `<|end_of_thinking|>`
//! - Qwen3: `<think>` / `</think>`

use aho_corasick::AhoCorasick;
use pyo3::prelude::*;
use std::sync::OnceLock;

/// Pre-compiled pattern matchers (singleton, thread-safe)
static START_MATCHER: OnceLock<AhoCorasick> = OnceLock::new();
static END_MATCHER: OnceLock<AhoCorasick> = OnceLock::new();
static INLINE_MATCHER: OnceLock<AhoCorasick> = OnceLock::new();

/// Start patterns for thinking blocks
const START_PATTERNS: &[&str] = &[
    "<｜begin▁of▁thinking｜>", // DeepSeek Unicode
    "<|begin_of_thinking|>",   // DeepSeek ASCII
    "<think>",                 // Qwen3
];

/// End patterns for thinking blocks
const END_PATTERNS: &[&str] = &[
    "<｜end▁of▁thinking｜>", // DeepSeek Unicode
    "<|end_of_thinking|>",   // DeepSeek ASCII
    "</think>",              // Qwen3
];

/// All patterns for inline stripping
const ALL_PATTERNS: &[&str] = &[
    "<｜begin▁of▁thinking｜>",
    "<｜end▁of▁thinking｜>",
    "<|begin_of_thinking|>",
    "<|end_of_thinking|>",
    "<think>",
    "</think>",
];

/// Get or initialize the start pattern matcher
fn get_start_matcher() -> &'static AhoCorasick {
    START_MATCHER
        .get_or_init(|| AhoCorasick::new(START_PATTERNS).expect("Failed to compile start patterns"))
}

/// Get or initialize the end pattern matcher
fn get_end_matcher() -> &'static AhoCorasick {
    END_MATCHER
        .get_or_init(|| AhoCorasick::new(END_PATTERNS).expect("Failed to compile end patterns"))
}

/// Get or initialize the inline pattern matcher
fn get_inline_matcher() -> &'static AhoCorasick {
    INLINE_MATCHER
        .get_or_init(|| AhoCorasick::new(ALL_PATTERNS).expect("Failed to compile inline patterns"))
}

/// Thinking state during streaming
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingState {
    Normal,
    InThinking,
}

/// Result of processing a streaming chunk
#[pyclass]
#[derive(Debug, Clone)]
pub struct StreamingChunkResult {
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub is_thinking: bool,
    #[pyo3(get)]
    pub state_changed: bool,
    #[pyo3(get)]
    pub entering_thinking: bool,
    #[pyo3(get)]
    pub exiting_thinking: bool,
}

#[pymethods]
impl StreamingChunkResult {
    fn __repr__(&self) -> String {
        format!(
            "StreamingChunkResult(content={:?}, is_thinking={}, entering={}, exiting={})",
            if self.content.len() > 50 {
                format!("{}...", &self.content[..50])
            } else {
                self.content.clone()
            },
            self.is_thinking,
            self.entering_thinking,
            self.exiting_thinking
        )
    }
}

/// High-performance streaming content filter.
///
/// Processes thinking tokens with O(n) complexity using Aho-Corasick
/// multi-pattern matching, compared to O(n * patterns) for sequential regex.
#[pyclass]
pub struct StreamingFilter {
    state: ThinkingState,
    buffer: String,
    thinking_content_length: usize,
    total_thinking_length: usize,
    should_abort: bool,
    abort_reason: Option<String>,
    suppress_thinking: bool,
    max_thinking_content: usize,
}

#[pymethods]
impl StreamingFilter {
    /// Create a new streaming filter.
    ///
    /// # Arguments
    /// * `suppress_thinking` - If true, completely suppress thinking content
    /// * `max_thinking_content` - Maximum chars before aborting (default 50000)
    #[new]
    #[pyo3(signature = (suppress_thinking = false, max_thinking_content = 50000))]
    pub fn new(suppress_thinking: bool, max_thinking_content: usize) -> Self {
        Self {
            state: ThinkingState::Normal,
            buffer: String::with_capacity(256),
            thinking_content_length: 0,
            total_thinking_length: 0,
            should_abort: false,
            abort_reason: None,
            suppress_thinking,
            max_thinking_content,
        }
    }

    /// Reset filter state for a new response.
    pub fn reset(&mut self) {
        self.state = ThinkingState::Normal;
        self.buffer.clear();
        self.thinking_content_length = 0;
        self.total_thinking_length = 0;
        self.should_abort = false;
        self.abort_reason = None;
    }

    /// Check if streaming should be aborted.
    pub fn should_abort(&self) -> bool {
        self.should_abort
    }

    /// Get abort reason if any.
    pub fn get_abort_reason(&self) -> Option<String> {
        self.abort_reason.clone()
    }

    /// Get current thinking state as string.
    pub fn get_state(&self) -> &'static str {
        match self.state {
            ThinkingState::Normal => "normal",
            ThinkingState::InThinking => "in_thinking",
        }
    }

    /// Get total thinking content length processed.
    pub fn get_thinking_length(&self) -> usize {
        self.total_thinking_length
    }

    /// Flush the residual buffer at end-of-stream.
    ///
    /// `process_chunk` buffers a potential partial pattern at the end of a
    /// chunk; if the stream ends before that pattern completes, the buffered
    /// bytes belong to the output. Previously they were silently dropped on
    /// the native path while the Python fallback's flush emitted them
    /// (co-design review U8-F7) — a response ending in a partial `<thi`
    /// lost those bytes only when the Rust wheel was installed.
    pub fn flush(&mut self) -> StreamingChunkResult {
        let content = std::mem::take(&mut self.buffer);
        let is_thinking = self.state == ThinkingState::InThinking;
        if is_thinking {
            self.thinking_content_length += content.len();
            self.total_thinking_length += content.len();
        }
        StreamingChunkResult {
            content: if is_thinking && self.suppress_thinking {
                String::new()
            } else {
                content
            },
            is_thinking,
            state_changed: false,
            entering_thinking: false,
            exiting_thinking: false,
        }
    }

    /// Process a content chunk, detecting thinking state transitions.
    ///
    /// This is the hot path - called for every streaming chunk.
    /// Uses Aho-Corasick for O(n) pattern detection.
    pub fn process_chunk(&mut self, chunk: &str) -> StreamingChunkResult {
        // Combine buffer with new chunk
        let combined = if self.buffer.is_empty() {
            chunk.to_string()
        } else {
            let mut s = std::mem::take(&mut self.buffer);
            s.push_str(chunk);
            s
        };

        // Fast path: check if any patterns might be present
        let start_matcher = get_start_matcher();
        let end_matcher = get_end_matcher();

        let mut result_content = String::with_capacity(combined.len());
        let mut entering_thinking = false;
        let mut exiting_thinking = false;
        let mut pos = 0;

        // Process content looking for state transitions
        while pos < combined.len() {
            match self.state {
                ThinkingState::Normal => {
                    // Look for start pattern
                    if let Some(mat) = start_matcher.find(&combined[pos..]) {
                        if mat.start() == 0 {
                            // Found start pattern at current position
                            self.state = ThinkingState::InThinking;
                            entering_thinking = true;
                            pos += mat.end();
                            continue;
                        } else {
                            // Content before the pattern
                            result_content.push_str(&combined[pos..pos + mat.start()]);
                            self.state = ThinkingState::InThinking;
                            entering_thinking = true;
                            pos += mat.start() + mat.len();
                            continue;
                        }
                    }
                    // No start pattern found - check if we might have partial pattern at end
                    let remaining = &combined[pos..];
                    if let Some(partial_len) = check_partial_pattern(remaining, START_PATTERNS) {
                        // Buffer the potential partial pattern
                        self.buffer = remaining[remaining.len() - partial_len..].to_string();
                        result_content.push_str(&remaining[..remaining.len() - partial_len]);
                        break;
                    }
                    // No patterns - return all content
                    result_content.push_str(remaining);
                    break;
                }
                ThinkingState::InThinking => {
                    // Look for end pattern
                    if let Some(mat) = end_matcher.find(&combined[pos..]) {
                        // Found end pattern
                        let thinking_content = &combined[pos..pos + mat.start()];
                        self.thinking_content_length += thinking_content.len();
                        self.total_thinking_length += thinking_content.len();

                        // Check for runaway thinking
                        if self.total_thinking_length > self.max_thinking_content {
                            self.should_abort = true;
                            self.abort_reason = Some(format!(
                                "Thinking content exceeded {} chars",
                                self.max_thinking_content
                            ));
                        }

                        if !self.suppress_thinking {
                            result_content.push_str(thinking_content);
                        }

                        self.state = ThinkingState::Normal;
                        exiting_thinking = true;
                        pos += mat.end();
                        continue;
                    }
                    // No end pattern found - check for partial
                    let remaining = &combined[pos..];
                    if let Some(partial_len) = check_partial_pattern(remaining, END_PATTERNS) {
                        self.buffer = remaining[remaining.len() - partial_len..].to_string();
                        let thinking_content = &remaining[..remaining.len() - partial_len];
                        self.thinking_content_length += thinking_content.len();
                        self.total_thinking_length += thinking_content.len();
                        if !self.suppress_thinking {
                            result_content.push_str(thinking_content);
                        }
                        break;
                    }
                    // No patterns - all is thinking content
                    self.thinking_content_length += remaining.len();
                    self.total_thinking_length += remaining.len();
                    if !self.suppress_thinking {
                        result_content.push_str(remaining);
                    }
                    break;
                }
            }
        }

        StreamingChunkResult {
            content: result_content,
            is_thinking: self.state == ThinkingState::InThinking,
            state_changed: entering_thinking || exiting_thinking,
            entering_thinking,
            exiting_thinking,
        }
    }
}

/// Check if text ends with a partial match of any pattern.
/// Returns the byte length of the partial match if found.
fn check_partial_pattern(text: &str, patterns: &[&str]) -> Option<usize> {
    if text.is_empty() {
        return None;
    }

    // Check last N characters where N is max pattern length
    // Use character counting for multi-byte UTF-8 safety
    let max_pattern_len = patterns
        .iter()
        .map(|p| p.chars().count())
        .max()
        .unwrap_or(0);

    // Get character positions for safe UTF-8 indexing
    let char_indices: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    let char_count = char_indices.len();

    let check_len = char_count.min(max_pattern_len - 1);

    if check_len == 0 {
        return None;
    }

    // Check if any pattern starts with text suffix
    for suffix_chars in (1..=check_len).rev() {
        // Get the byte index at the character position (safe for UTF-8)
        let start_byte_idx = char_indices[char_count - suffix_chars];
        let suffix = &text[start_byte_idx..];

        for pattern in patterns {
            // Use character counting for comparison
            if pattern.starts_with(suffix) && suffix.chars().count() < pattern.chars().count() {
                return Some(suffix.len());
            }
        }
    }

    None
}

/// Strip all thinking tokens from content (batch processing).
///
/// More efficient than streaming for complete content.
#[pyfunction]
pub fn strip_thinking_tokens(content: &str) -> String {
    let matcher = get_inline_matcher();

    // Fast path: no patterns found
    if matcher.find(content).is_none() {
        return content.to_string();
    }

    // Remove all matches
    let mut result = String::with_capacity(content.len());
    let mut last_end = 0;

    for mat in matcher.find_iter(content) {
        result.push_str(&content[last_end..mat.start()]);
        last_end = mat.end();
    }
    result.push_str(&content[last_end..]);

    result
}

/// Check if content contains any thinking tokens.
#[pyfunction]
pub fn contains_thinking_tokens(content: &str) -> bool {
    get_inline_matcher().find(content).is_some()
}

/// Find all thinking token positions in content.
///
/// Returns list of (start, end, pattern_index) tuples.
#[pyfunction]
pub fn find_thinking_tokens(content: &str) -> Vec<(usize, usize, usize)> {
    get_inline_matcher()
        .find_iter(content)
        .map(|m| (m.start(), m.end(), m.pattern().as_usize()))
        .collect()
}

/// Extract thinking content from a complete response.
///
/// Returns (main_content, thinking_content) tuple.
#[pyfunction]
pub fn extract_thinking_content(content: &str) -> (String, String) {
    let start_matcher = get_start_matcher();
    let end_matcher = get_end_matcher();

    let mut main_content = String::with_capacity(content.len());
    let mut thinking_content = String::with_capacity(content.len() / 4);
    let mut in_thinking = false;
    let mut pos = 0;

    while pos < content.len() {
        if !in_thinking {
            if let Some(mat) = start_matcher.find(&content[pos..]) {
                main_content.push_str(&content[pos..pos + mat.start()]);
                in_thinking = true;
                pos += mat.start() + mat.len();
            } else {
                main_content.push_str(&content[pos..]);
                break;
            }
        } else if let Some(mat) = end_matcher.find(&content[pos..]) {
            thinking_content.push_str(&content[pos..pos + mat.start()]);
            in_thinking = false;
            pos += mat.start() + mat.len();
        } else {
            thinking_content.push_str(&content[pos..]);
            break;
        }
    }

    (main_content, thinking_content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_thinking_tokens() {
        let content = "Hello <think>internal thought</think> World";
        let result = strip_thinking_tokens(content);
        assert_eq!(result, "Hello internal thought World");
    }

    #[test]
    fn test_strip_deepseek_tokens() {
        let content = "Result <|begin_of_thinking|>thinking...<|end_of_thinking|> done";
        let result = strip_thinking_tokens(content);
        assert_eq!(result, "Result thinking... done");
    }

    #[test]
    fn test_contains_thinking_tokens() {
        assert!(contains_thinking_tokens("<think>test</think>"));
        assert!(contains_thinking_tokens("<|begin_of_thinking|>"));
        assert!(!contains_thinking_tokens("normal text"));
    }

    #[test]
    fn test_extract_thinking_content() {
        let content = "Start <think>thinking here</think> End";
        let (main, thinking) = extract_thinking_content(content);
        assert_eq!(main, "Start  End");
        assert_eq!(thinking, "thinking here");
    }

    #[test]
    fn test_streaming_filter_basic() {
        let mut filter = StreamingFilter::new(false, 50000);

        let result = filter.process_chunk("Hello ");
        assert_eq!(result.content, "Hello ");
        assert!(!result.is_thinking);

        let result = filter.process_chunk("<think>thinking");
        assert!(result.entering_thinking);
        assert!(result.is_thinking);

        let result = filter.process_chunk(" more</think> done");
        assert!(result.exiting_thinking);
        assert!(!result.is_thinking);
    }

    #[test]
    fn test_streaming_filter_suppress() {
        let mut filter = StreamingFilter::new(true, 50000);

        filter.process_chunk("Start ");
        let result = filter.process_chunk("<think>hidden</think>");
        // Content should be empty when suppressing
        assert!(!result.content.contains("hidden"));
    }

    #[test]
    fn test_partial_pattern_detection() {
        assert_eq!(check_partial_pattern("<thi", START_PATTERNS), Some(4));
        assert_eq!(check_partial_pattern("<think>", START_PATTERNS), None);
        assert_eq!(check_partial_pattern("hello", START_PATTERNS), None);
    }
}
