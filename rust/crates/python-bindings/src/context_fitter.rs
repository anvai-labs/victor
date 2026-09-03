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

//! Context fitting and message truncation for LLM context windows.
//!
//! This module provides strategies for fitting conversation messages into
//! a fixed token budget, plus a fast message truncation helper.  All token
//! counts use a lightweight heuristic (word-count * 1.3) so there is no
//! dependency on an external tokenizer.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Token estimation heuristic
// ---------------------------------------------------------------------------

/// Estimate the number of tokens in a text using a fast heuristic.
///
/// Uses `words * 13 / 10` (≈ 1.3 tokens per word) which is a reasonable
/// approximation for English text with typical LLM tokenizers.
#[inline]
fn count_tokens_fast(text: &str) -> usize {
    let words = text.split_whitespace().count();
    words * 13 / 10
}

// ---------------------------------------------------------------------------
// MessageSlot
// ---------------------------------------------------------------------------

/// A single message slot with metadata used by the fitting algorithm.
#[pyclass]
#[derive(Clone)]
pub struct MessageSlot {
    /// Original index in the conversation
    #[pyo3(get)]
    pub index: usize,
    /// Number of tokens in this message
    #[pyo3(get)]
    pub token_count: usize,
    /// Priority level (0 = lowest, 255 = highest)
    #[pyo3(get)]
    pub priority: u8,
    /// Message role (e.g. "system", "user", "assistant")
    #[pyo3(get)]
    pub role: String,
    /// Recency score (0.0 = oldest, 1.0 = newest)
    #[pyo3(get)]
    pub recency: f32,
}

#[pymethods]
impl MessageSlot {
    #[new]
    pub fn new(index: usize, token_count: usize, priority: u8, role: String, recency: f32) -> Self {
        Self {
            index,
            token_count,
            priority,
            role,
            recency,
        }
    }
}

// ---------------------------------------------------------------------------
// FitResult
// ---------------------------------------------------------------------------

/// Result of a context-fitting operation.
#[pyclass]
pub struct FitResult {
    /// Indices of messages that were kept, in original order
    #[pyo3(get)]
    pub kept_indices: Vec<usize>,
    /// Total token count of kept messages
    #[pyo3(get)]
    pub total_tokens: usize,
    /// Number of messages that were dropped
    #[pyo3(get)]
    pub dropped_count: usize,
    /// Number of tokens freed by dropping messages
    #[pyo3(get)]
    pub freed_tokens: usize,
}

// ---------------------------------------------------------------------------
// fit_context
// ---------------------------------------------------------------------------

/// Fit a list of messages into a token budget using the given strategy.
///
/// # Strategies
///
/// - `"fifo"` — Keep newest messages, drop oldest first.  System messages
///   are always preserved when `preserve_system` is true.
/// - `"priority"` — Score each message as `0.4 * (priority / 100) +
///   recency * 0.6`, keep the highest-scoring messages that fit.
/// - `"smart"` — Always keep system messages, the first user message, and
///   the last 2 messages.  Score the remaining messages by the same
///   formula and drop the lowest until the budget is met.
///
/// Unknown strategy names raise `ValueError` (no silent default).
///
/// # Arguments
/// * `messages` - List of `MessageSlot` instances
/// * `budget` - Maximum number of tokens to keep
/// * `strategy` - One of `"fifo"`, `"priority"`, `"smart"` (default `"smart"`)
/// * `preserve_system` - Whether system messages are always kept (default `true`)
///
/// # Returns
/// A `FitResult` with the kept indices, total tokens, dropped count, and freed tokens.
#[pyfunction]
#[pyo3(signature = (messages, budget, strategy = "smart", preserve_system = true))]
pub fn fit_context(
    messages: Vec<MessageSlot>,
    budget: usize,
    strategy: &str,
    preserve_system: bool,
) -> PyResult<FitResult> {
    if messages.is_empty() {
        return Ok(FitResult {
            kept_indices: Vec::new(),
            total_tokens: 0,
            dropped_count: 0,
            freed_tokens: 0,
        });
    }

    let total_all: usize = messages.iter().map(|m| m.token_count).sum();

    // If everything fits, keep all
    if total_all <= budget {
        return Ok(FitResult {
            kept_indices: messages.iter().map(|m| m.index).collect(),
            total_tokens: total_all,
            dropped_count: 0,
            freed_tokens: 0,
        });
    }

    // Unknown strategy names are an error, not a silent default — the
    // Python fallback previously mapped unknown names to "balanced" while
    // this side mapped them to smart (co-design review U8-F1).
    let kept_indices = match strategy {
        "fifo" => fit_fifo(&messages, budget, preserve_system),
        "priority" => fit_priority(&messages, budget, preserve_system),
        "smart" => fit_smart(&messages, budget),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown fit strategy '{other}' (expected 'fifo', 'priority', or 'smart')"
            )));
        }
    };

    let total_tokens: usize = kept_indices
        .iter()
        .filter_map(|&idx| messages.iter().find(|m| m.index == idx))
        .map(|m| m.token_count)
        .sum();
    let dropped_count = messages.len() - kept_indices.len();
    let freed_tokens = total_all - total_tokens;

    Ok(FitResult {
        kept_indices,
        total_tokens,
        dropped_count,
        freed_tokens,
    })
}

/// FIFO: keep newest, drop oldest.  System messages are pinned when requested.
fn fit_fifo(messages: &[MessageSlot], budget: usize, preserve_system: bool) -> Vec<usize> {
    // Separate system messages if preserving
    let (pinned, rest): (Vec<&MessageSlot>, Vec<&MessageSlot>) = if preserve_system {
        messages.iter().partition(|m| m.role == "system")
    } else {
        (Vec::new(), messages.iter().collect())
    };

    let pinned_cost: usize = pinned.iter().map(|m| m.token_count).sum();
    let remaining_budget = budget.saturating_sub(pinned_cost);

    // Walk rest in reverse (newest first) and keep until budget exhausted
    let mut kept: Vec<usize> = pinned.iter().map(|m| m.index).collect();
    let mut used = 0usize;
    for m in rest.iter().rev() {
        if used + m.token_count <= remaining_budget {
            kept.push(m.index);
            used += m.token_count;
        }
    }

    // Return in original order
    kept.sort_unstable();
    kept
}

/// Priority: score = 0.4 * (priority / 100) + recency * 0.6, keep highest.
///
/// Priority is a 0-255 u8 but is scored on the documented /100 scale (see
/// batch_score_messages) — the previous raw `priority * 0.4` made priority
/// dominate recency by ~40x (co-design review U8 scale-drift finding).
fn fit_priority(messages: &[MessageSlot], budget: usize, preserve_system: bool) -> Vec<usize> {
    let (pinned, rest): (Vec<&MessageSlot>, Vec<&MessageSlot>) = if preserve_system {
        messages.iter().partition(|m| m.role == "system")
    } else {
        (Vec::new(), messages.iter().collect())
    };

    let pinned_cost: usize = pinned.iter().map(|m| m.token_count).sum();
    let remaining_budget = budget.saturating_sub(pinned_cost);

    // Score and sort descending
    let mut scored: Vec<(&MessageSlot, f32)> = rest
        .iter()
        .map(|m| {
            let score = (m.priority as f32 / 100.0) * 0.4 + m.recency * 0.6;
            (*m, score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept: Vec<usize> = pinned.iter().map(|m| m.index).collect();
    let mut used = 0usize;
    for (m, _score) in &scored {
        if used + m.token_count <= remaining_budget {
            kept.push(m.index);
            used += m.token_count;
        }
    }

    kept.sort_unstable();
    kept
}

/// Smart: pin system + first user + last 2, score the rest, drop lowest.
fn fit_smart(messages: &[MessageSlot], budget: usize) -> Vec<usize> {
    if messages.is_empty() {
        return Vec::new();
    }

    // Identify pinned indices
    let mut pinned_set = std::collections::HashSet::new();

    // Always keep system messages
    for m in messages {
        if m.role == "system" {
            pinned_set.insert(m.index);
        }
    }

    // Keep the first user message
    for m in messages {
        if m.role == "user" {
            pinned_set.insert(m.index);
            break;
        }
    }

    // Keep the last 2 messages
    let n = messages.len();
    if n >= 1 {
        pinned_set.insert(messages[n - 1].index);
    }
    if n >= 2 {
        pinned_set.insert(messages[n - 2].index);
    }

    let pinned_cost: usize = messages
        .iter()
        .filter(|m| pinned_set.contains(&m.index))
        .map(|m| m.token_count)
        .sum();

    // If pinned alone exceed budget, just return pinned
    if pinned_cost >= budget {
        let mut kept: Vec<usize> = pinned_set.into_iter().collect();
        kept.sort_unstable();
        return kept;
    }

    let remaining_budget = budget - pinned_cost;

    // Score the rest (same normalized formula as fit_priority)
    let mut scored: Vec<(&MessageSlot, f32)> = messages
        .iter()
        .filter(|m| !pinned_set.contains(&m.index))
        .map(|m| {
            let score = (m.priority as f32 / 100.0) * 0.4 + m.recency * 0.6;
            (m, score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept: Vec<usize> = pinned_set.into_iter().collect();
    let mut used = 0usize;
    for (m, _score) in &scored {
        if used + m.token_count <= remaining_budget {
            kept.push(m.index);
            used += m.token_count;
        }
    }

    kept.sort_unstable();
    kept
}

// ---------------------------------------------------------------------------
// truncate_message
// ---------------------------------------------------------------------------

/// Truncate a message to fit within a token budget while preserving context.
///
/// Keeps the first `preserve_lines` and last `preserve_lines` lines, replacing
/// the middle with a `[...truncated N lines...]` marker.  If the message
/// already fits, it is returned unchanged.
///
/// # Arguments
/// * `content` - The message text to truncate
/// * `max_tokens` - Maximum token budget for the output
/// * `preserve_lines` - Number of lines to keep at the beginning and end (default 5)
///
/// # Returns
/// The (possibly truncated) message text
#[pyfunction]
#[pyo3(signature = (content, max_tokens, preserve_lines = 5))]
pub fn truncate_message(content: &str, max_tokens: usize, preserve_lines: usize) -> String {
    // If it already fits, return as-is
    if count_tokens_fast(content) <= max_tokens {
        return content.to_string();
    }

    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();

    // If there aren't enough lines to split meaningfully, just truncate words
    if total_lines <= preserve_lines * 2 + 1 {
        // Fall back to simple word truncation
        let words: Vec<&str> = content.split_whitespace().collect();
        let target_words = max_tokens * 10 / 13; // inverse of the 13/10 heuristic
        if target_words >= words.len() {
            return content.to_string();
        }
        let mut result: String = words[..target_words].join(" ");
        result.push_str("\n[...truncated...]");
        return result;
    }

    // Binary search for the right amount of middle lines to drop
    let head = &lines[..preserve_lines];
    let tail = &lines[total_lines - preserve_lines..];
    let dropped = total_lines - 2 * preserve_lines;

    let marker = format!("\n[...truncated {} lines...]\n", dropped);

    let mut result = head.join("\n");
    result.push_str(&marker);
    result.push_str(&tail.join("\n"));

    // If still too large after the initial truncation, keep reducing
    // by shrinking preserve_lines (but this path is uncommon)
    if count_tokens_fast(&result) > max_tokens && preserve_lines > 1 {
        return truncate_message(content, max_tokens, preserve_lines - 1);
    }

    result
}

// ---------------------------------------------------------------------------
// Batch scoring for ConversationMemory._score_messages()
// ---------------------------------------------------------------------------

/// Score messages by priority (40%) and recency (60%), return sorted indices.
///
/// Formula: score = 0.4 * (priority / 100) + 0.6 * (1 - age / max_age)
/// where age = max_timestamp - timestamp, max_age = max(ages) or 1.0
///
/// Returns Vec<(index, score)> sorted by score descending.
#[pyfunction]
#[pyo3(signature = (priorities, timestamps))]
pub fn batch_score_messages(priorities: Vec<u8>, timestamps: Vec<f64>) -> Vec<(usize, f32)> {
    let n = priorities.len().min(timestamps.len());
    if n == 0 {
        return Vec::new();
    }

    // Find max timestamp (most recent) and compute max_age
    let max_ts = timestamps[..n]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let max_age = timestamps[..n]
        .iter()
        .map(|&t| max_ts - t)
        .fold(0.0_f64, f64::max)
        .max(1e-9); // avoid division by zero

    // Score each message
    let mut scored: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let priority_score = priorities[i] as f32 / 100.0;
            let age = (max_ts - timestamps[i]) as f32;
            let recency_score = 1.0 - (age / max_age as f32);
            let score = priority_score * 0.4 + recency_score * 0.6;
            (i, score)
        })
        .collect();

    // Sort descending by score, stable sort preserves insertion order for ties
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- count_tokens_fast --------------------------------------------------

    #[test]
    fn test_count_tokens_fast_empty() {
        assert_eq!(count_tokens_fast(""), 0);
    }

    #[test]
    fn test_count_tokens_fast_single_word() {
        // 1 word * 13 / 10 = 1 (integer division)
        assert_eq!(count_tokens_fast("hello"), 1);
    }

    #[test]
    fn test_count_tokens_fast_multiple_words() {
        // 10 words * 13 / 10 = 13
        assert_eq!(count_tokens_fast("a b c d e f g h i j"), 13);
    }

    // -- MessageSlot --------------------------------------------------------

    #[test]
    fn test_message_slot_new() {
        let slot = MessageSlot::new(0, 100, 5, "user".to_string(), 0.8);
        assert_eq!(slot.index, 0);
        assert_eq!(slot.token_count, 100);
        assert_eq!(slot.priority, 5);
        assert_eq!(slot.role, "user");
        assert!((slot.recency - 0.8).abs() < 1e-6);
    }

    // -- fit_context: edge cases --------------------------------------------

    #[test]
    fn test_fit_context_empty() {
        let result = fit_context(vec![], 1000, "smart", true).unwrap();
        assert!(result.kept_indices.is_empty());
        assert_eq!(result.total_tokens, 0);
        assert_eq!(result.dropped_count, 0);
        assert_eq!(result.freed_tokens, 0);
    }

    #[test]
    fn test_fit_context_all_fit() {
        let messages = vec![
            MessageSlot::new(0, 50, 5, "system".to_string(), 0.0),
            MessageSlot::new(1, 50, 5, "user".to_string(), 1.0),
        ];
        let result = fit_context(messages, 200, "fifo", true).unwrap();
        assert_eq!(result.kept_indices, vec![0, 1]);
        assert_eq!(result.total_tokens, 100);
        assert_eq!(result.dropped_count, 0);
        assert_eq!(result.freed_tokens, 0);
    }

    // -- fit_context: fifo --------------------------------------------------

    #[test]
    fn test_fit_fifo_drops_oldest() {
        let messages = vec![
            MessageSlot::new(0, 100, 5, "user".to_string(), 0.0), // oldest
            MessageSlot::new(1, 100, 5, "assistant".to_string(), 0.5),
            MessageSlot::new(2, 100, 5, "user".to_string(), 1.0), // newest
        ];
        // Budget 200: can keep 2 of 3 (total 300)
        let result = fit_context(messages, 200, "fifo", false).unwrap();
        assert_eq!(result.kept_indices, vec![1, 2]); // newest two
        assert_eq!(result.total_tokens, 200);
        assert_eq!(result.dropped_count, 1);
        assert_eq!(result.freed_tokens, 100);
    }

    #[test]
    fn test_fit_fifo_preserves_system() {
        let messages = vec![
            MessageSlot::new(0, 100, 10, "system".to_string(), 0.0),
            MessageSlot::new(1, 100, 5, "user".to_string(), 0.3),
            MessageSlot::new(2, 100, 5, "assistant".to_string(), 0.6),
            MessageSlot::new(3, 100, 5, "user".to_string(), 1.0),
        ];
        // Budget 200: system (100) pinned + 1 more
        let result = fit_context(messages, 200, "fifo", true).unwrap();
        assert!(result.kept_indices.contains(&0)); // system preserved
        assert!(result.kept_indices.contains(&3)); // newest non-system
        assert_eq!(result.total_tokens, 200);
        assert_eq!(result.dropped_count, 2);
    }

    // -- fit_context: priority ----------------------------------------------

    #[test]
    fn test_fit_priority_keeps_high_priority() {
        let messages = vec![
            MessageSlot::new(0, 100, 10, "user".to_string(), 0.0), // score: 0.4*0.10 + 0.6*0.0 = 0.040
            MessageSlot::new(1, 100, 1, "user".to_string(), 0.5), // score: 0.4*0.01 + 0.6*0.5 = 0.304
            MessageSlot::new(2, 100, 8, "user".to_string(), 1.0), // score: 0.4*0.08 + 0.6*1.0 = 0.632
        ];
        // Budget 200: keep top 2 by score → indices 2 (0.632) and 1 (0.304).
        let result = fit_context(messages, 200, "priority", false).unwrap();
        assert_eq!(result.kept_indices, vec![1, 2]);
        assert_eq!(result.dropped_count, 1);
    }

    #[test]
    fn test_fit_priority_preserves_system() {
        let messages = vec![
            MessageSlot::new(0, 100, 1, "system".to_string(), 0.0),
            MessageSlot::new(1, 100, 10, "user".to_string(), 1.0),
            MessageSlot::new(2, 100, 1, "assistant".to_string(), 0.5),
        ];
        // Budget 200: system pinned (100), then best scored non-system
        let result = fit_context(messages, 200, "priority", true).unwrap();
        assert!(result.kept_indices.contains(&0)); // system
        assert!(result.kept_indices.contains(&1)); // highest score
        assert_eq!(result.total_tokens, 200);
    }

    // -- fit_context: smart -------------------------------------------------

    #[test]
    fn test_fit_smart_pins_required() {
        let messages = vec![
            MessageSlot::new(0, 50, 10, "system".to_string(), 0.0), // pinned (system)
            MessageSlot::new(1, 50, 5, "user".to_string(), 0.2),    // pinned (first user)
            MessageSlot::new(2, 50, 5, "assistant".to_string(), 0.4),
            MessageSlot::new(3, 50, 5, "user".to_string(), 0.6),
            MessageSlot::new(4, 50, 5, "assistant".to_string(), 0.8), // pinned (2nd to last)
            MessageSlot::new(5, 50, 5, "user".to_string(), 1.0),      // pinned (last)
        ];
        // Budget 200: 4 pinned messages = 200 tokens, no room for rest
        let result = fit_context(messages, 200, "smart", true).unwrap();
        assert!(result.kept_indices.contains(&0)); // system
        assert!(result.kept_indices.contains(&1)); // first user
        assert!(result.kept_indices.contains(&4)); // 2nd to last
        assert!(result.kept_indices.contains(&5)); // last
        assert_eq!(result.total_tokens, 200);
    }

    #[test]
    fn test_fit_smart_fills_remaining_by_score() {
        let messages = vec![
            MessageSlot::new(0, 30, 10, "system".to_string(), 0.0),
            MessageSlot::new(1, 30, 5, "user".to_string(), 0.2),
            MessageSlot::new(2, 30, 1, "assistant".to_string(), 0.4), // low score
            MessageSlot::new(3, 30, 9, "user".to_string(), 0.6),      // high score
            MessageSlot::new(4, 30, 5, "assistant".to_string(), 0.8),
            MessageSlot::new(5, 30, 5, "user".to_string(), 1.0),
        ];
        // Budget 150: pinned = {0,1,4,5} = 120, room for 1 more (30)
        // index 3 score = 0.4*0.09 + 0.6*0.6 = 0.396
        // index 2 score = 0.4*0.01 + 0.6*0.4 = 0.244
        // Should pick index 3
        let result = fit_context(messages, 150, "smart", true).unwrap();
        assert!(result.kept_indices.contains(&3));
        assert!(!result.kept_indices.contains(&2));
        assert_eq!(result.total_tokens, 150);
    }

    #[test]
    fn test_fit_context_unknown_strategy_errors() {
        let messages = vec![MessageSlot::new(0, 100, 5, "user".to_string(), 1.0)];
        let result = fit_context(messages, 50, "bogus", true);
        assert!(result.is_err());
    }

    // -- truncate_message ---------------------------------------------------

    #[test]
    fn test_truncate_message_already_fits() {
        let content = "short message";
        let result = truncate_message(content, 1000, 5);
        assert_eq!(result, content);
    }

    #[test]
    fn test_truncate_message_inserts_marker() {
        // Create a long message with many lines
        let lines: Vec<String> = (0..20).map(|i| format!("Line number {}", i)).collect();
        let content = lines.join("\n");

        // Use a small token budget to force truncation
        let result = truncate_message(&content, 30, 3);

        assert!(result.contains("[...truncated"));
        assert!(result.contains("lines...]"));

        // Should start with the first 3 lines
        assert!(result.starts_with("Line number 0\nLine number 1\nLine number 2"));
        // Should end with the last 3 lines
        assert!(result.ends_with("Line number 17\nLine number 18\nLine number 19"));
    }

    #[test]
    fn test_truncate_message_few_lines_fallback() {
        // Fewer lines than 2 * preserve_lines + 1 → word truncation
        let content = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10";
        let result = truncate_message(content, 2, 5);
        // token estimate for 10 words = 13, budget = 2
        // target_words = 2 * 10 / 13 = 1
        assert!(result.contains("[...truncated...]"));
    }

    #[test]
    fn test_truncate_message_preserve_lines_zero() {
        let lines: Vec<String> = (0..20).map(|i| format!("Line {}", i)).collect();
        let content = lines.join("\n");
        let result = truncate_message(&content, 2, 0);
        assert!(result.contains("[...truncated"));
    }
}
