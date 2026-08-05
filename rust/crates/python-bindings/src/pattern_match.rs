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

//! High-performance multi-pattern matching using Aho-Corasick.
//!
//! This module provides 10-100x faster multi-pattern matching compared to
//! Python re module for keyword detection, tool selection, and task classification.
//!
//! Use cases:
//! - Tool keyword matching (find all matching tools for a query)
//! - Intent detection (match multiple intent patterns simultaneously)
//! - Task classification (detect action/analysis/generation keywords)
//! - Entity extraction (find all entity mentions in text)

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Cache of compiled Aho-Corasick automata keyed by
/// `(case_insensitive, leftmost_longest, patterns joined with NUL)`.
///
/// The free `#[pyfunction]`s below are frequently called in loops over the SAME
/// small, static pattern sets (e.g. tool-selection keyword categories), so
/// rebuilding the automaton on every call is pure waste — compile once, reuse.
type AcKey = (bool, bool, String);
static AC_CACHE: OnceLock<Mutex<HashMap<AcKey, Arc<AhoCorasick>>>> = OnceLock::new();

/// Cap on retained automata so a caller passing per-request-varying patterns
/// cannot grow the cache without bound. Realistic callers use a handful of sets.
const AC_CACHE_MAX_ENTRIES: usize = 512;

/// Return a compiled automaton for `patterns`, building and caching it on first
/// use. `leftmost_longest` selects `MatchKind::LeftmostLongest` (vs the default
/// `Standard`) and is part of the cache key so the two never collide. Returns
/// `None` only if the patterns fail to compile.
fn cached_automaton(
    patterns: &[String],
    case_insensitive: bool,
    leftmost_longest: bool,
) -> Option<Arc<AhoCorasick>> {
    let key: AcKey = (case_insensitive, leftmost_longest, patterns.join("\u{0}"));
    let cache = AC_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache.lock().ok()?;
        if let Some(ac) = guard.get(&key) {
            return Some(Arc::clone(ac));
        }
    }
    let mut builder = AhoCorasickBuilder::new();
    builder.ascii_case_insensitive(case_insensitive);
    if leftmost_longest {
        builder.match_kind(MatchKind::LeftmostLongest);
    }
    let ac = Arc::new(builder.build(patterns).ok()?);
    if let Ok(mut guard) = cache.lock() {
        // Simple cap: once full, keep serving the hot first-seen sets.
        if guard.len() < AC_CACHE_MAX_ENTRIES {
            guard.insert(key, Arc::clone(&ac));
        }
    }
    Some(ac)
}

/// Owned version of PatternMatch for Python compatibility
/// (stores owned strings to avoid lifetime issues across Python FFI boundary)
#[pyclass]
#[derive(Clone)]
pub struct PatternMatch {
    /// The pattern index that matched
    #[pyo3(get)]
    pub pattern_idx: usize,
    /// The matched text (owned for Python FFI)
    #[pyo3(get)]
    pub matched_text: String,
    /// Start position in text
    #[pyo3(get)]
    pub start: usize,
    /// End position in text
    #[pyo3(get)]
    pub end: usize,
}

/// Zero-copy match result for internal Rust use
/// Uses borrowed data to avoid allocations in hot paths
#[derive(Clone, Copy)]
pub struct MatchResult<'a> {
    /// The pattern index that matched
    pub pattern_idx: usize,
    /// The matched text (borrowed from input)
    pub matched_text: &'a str,
    /// Start position in text
    pub start: usize,
    /// End position in text
    pub end: usize,
}

impl<'a> From<MatchResult<'a>> for PatternMatch {
    fn from(m: MatchResult<'a>) -> Self {
        Self {
            pattern_idx: m.pattern_idx,
            matched_text: m.matched_text.to_string(),
            start: m.start,
            end: m.end,
        }
    }
}

#[pymethods]
impl PatternMatch {
    fn __repr__(&self) -> String {
        format!(
            "PatternMatch(pattern={}, text='{}', span=({}, {}))",
            self.pattern_idx, self.matched_text, self.start, self.end
        )
    }
}

/// Fast multi-pattern matcher using Aho-Corasick algorithm
#[pyclass]
pub struct PatternMatcher {
    /// The compiled Aho-Corasick automaton
    automaton: AhoCorasick,
    /// Original patterns for reference
    patterns: Vec<String>,
    /// Whether matching is case-insensitive
    #[pyo3(get)]
    case_insensitive: bool,
}

#[pymethods]
impl PatternMatcher {
    /// Create a new pattern matcher from a list of patterns
    #[new]
    #[pyo3(signature = (patterns, case_insensitive=true))]
    pub fn new(patterns: Vec<String>, case_insensitive: bool) -> PyResult<Self> {
        let automaton = AhoCorasickBuilder::new()
            .ascii_case_insensitive(case_insensitive)
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            automaton,
            patterns,
            case_insensitive,
        })
    }

    /// Find all matches in text (Python FFI version with owned strings)
    pub fn find_all(&self, text: &str) -> Vec<PatternMatch> {
        self.automaton
            .find_iter(text)
            .map(|m| PatternMatch {
                pattern_idx: m.pattern().as_usize(),
                matched_text: text[m.start()..m.end()].to_string(),
                start: m.start(),
                end: m.end(),
            })
            .collect()
    }

    /// Check if text contains any pattern
    pub fn contains_any(&self, text: &str) -> bool {
        self.automaton.is_match(text)
    }

    /// Count total number of matches
    pub fn count_matches(&self, text: &str) -> usize {
        self.automaton.find_iter(text).count()
    }

    /// Get unique pattern indices that matched
    pub fn matched_patterns(&self, text: &str) -> Vec<usize> {
        let mut seen = vec![false; self.patterns.len()];
        let mut result = Vec::new();

        for m in self.automaton.find_iter(text) {
            let idx = m.pattern().as_usize();
            if !seen[idx] {
                seen[idx] = true;
                result.push(idx);
            }
        }

        result
    }

    /// Get pattern strings that matched
    pub fn matched_pattern_strings(&self, text: &str) -> Vec<String> {
        self.matched_patterns(text)
            .into_iter()
            .map(|i| self.patterns[i].clone())
            .collect()
    }

    /// Count matches per pattern
    pub fn count_by_pattern(&self, text: &str) -> HashMap<usize, usize> {
        let mut counts = HashMap::new();
        for m in self.automaton.find_iter(text) {
            *counts.entry(m.pattern().as_usize()).or_insert(0) += 1;
        }
        counts
    }

    /// Get pattern at index
    pub fn get_pattern(&self, idx: usize) -> Option<String> {
        self.patterns.get(idx).cloned()
    }

    /// Get number of patterns
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Replace all matches with a replacement string
    #[pyo3(signature = (text, replacement))]
    pub fn replace_all(&self, text: &str, replacement: &str) -> String {
        // Create a replacement for each pattern
        let replacements: Vec<&str> = vec![replacement; self.patterns.len()];
        self.automaton.replace_all(text, &replacements)
    }

    /// Replace matches with pattern-specific replacements
    #[pyo3(signature = (text, replacements))]
    pub fn replace_all_with(&self, text: &str, replacements: Vec<String>) -> PyResult<String> {
        if replacements.len() != self.patterns.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} replacements, got {}",
                self.patterns.len(),
                replacements.len()
            )));
        }
        let refs: Vec<&str> = replacements.iter().map(|s| s.as_str()).collect();
        Ok(self.automaton.replace_all(text, &refs))
    }
}

/// Internal implementation methods (not exposed to Python)
impl PatternMatcher {
    /// Find all matches in text (zero-copy internal version)
    #[allow(dead_code)]
    fn find_all_internal<'a>(&'a self, text: &'a str) -> Vec<MatchResult<'a>> {
        self.automaton
            .find_iter(text)
            .map(|m| MatchResult {
                pattern_idx: m.pattern().as_usize(),
                matched_text: &text[m.start()..m.end()],
                start: m.start(),
                end: m.end(),
            })
            .collect()
    }
}

/// Quick function to check if text contains any of the patterns
#[pyfunction]
#[pyo3(signature = (text, patterns, case_insensitive=true))]
pub fn contains_any_pattern(text: &str, patterns: Vec<String>, case_insensitive: bool) -> bool {
    match cached_automaton(&patterns, case_insensitive, false) {
        Some(ac) => ac.is_match(text),
        None => false,
    }
}

/// Quick function to find all pattern matches in text (zero-copy internal)
#[pyfunction]
#[pyo3(signature = (text, patterns, case_insensitive=true))]
pub fn find_all_patterns(
    text: &str,
    patterns: Vec<String>,
    case_insensitive: bool,
) -> Vec<PatternMatch> {
    match cached_automaton(&patterns, case_insensitive, true) {
        Some(ac) => ac
            .find_iter(text)
            .map(|m| PatternMatch {
                pattern_idx: m.pattern().as_usize(),
                matched_text: text[m.start()..m.end()].to_string(),
                start: m.start(),
                end: m.end(),
            })
            .collect(),
        None => Vec::new(),
    }
}

/// Zero-copy internal version of find_all_patterns for Rust-to-Rust calls
#[allow(dead_code)]
pub fn find_all_patterns_internal<'a>(
    text: &'a str,
    patterns: &[String],
    case_insensitive: bool,
) -> Vec<MatchResult<'a>> {
    match cached_automaton(patterns, case_insensitive, true) {
        Some(ac) => ac
            .find_iter(text)
            .map(|m| MatchResult {
                pattern_idx: m.pattern().as_usize(),
                matched_text: &text[m.start()..m.end()],
                start: m.start(),
                end: m.end(),
            })
            .collect(),
        None => Vec::new(),
    }
}

/// Quick function to count pattern matches
#[pyfunction]
#[pyo3(signature = (text, patterns, case_insensitive=true))]
pub fn count_pattern_matches(text: &str, patterns: Vec<String>, case_insensitive: bool) -> usize {
    match cached_automaton(&patterns, case_insensitive, false) {
        Some(ac) => ac.find_iter(text).count(),
        None => 0,
    }
}

/// Quick function to get indices of matched patterns
#[pyfunction]
#[pyo3(signature = (text, patterns, case_insensitive=true))]
pub fn get_matched_pattern_indices(
    text: &str,
    patterns: Vec<String>,
    case_insensitive: bool,
) -> Vec<usize> {
    match cached_automaton(&patterns, case_insensitive, false) {
        Some(ac) => {
            let mut seen = vec![false; patterns.len()];
            let mut result = Vec::new();

            for m in ac.find_iter(text) {
                let idx = m.pattern().as_usize();
                if !seen[idx] {
                    seen[idx] = true;
                    result.push(idx);
                }
            }
            result
        }
        None => Vec::new(),
    }
}

/// Batch check multiple texts against the same patterns
#[pyfunction]
#[pyo3(signature = (texts, patterns, case_insensitive=true))]
pub fn batch_contains_any(
    texts: Vec<String>,
    patterns: Vec<String>,
    case_insensitive: bool,
) -> Vec<bool> {
    match cached_automaton(&patterns, case_insensitive, false) {
        Some(ac) => texts.iter().map(|t| ac.is_match(t)).collect(),
        None => vec![false; texts.len()],
    }
}

/// Calculate a weighted score for pattern matches
/// Each pattern can have a weight, and the function returns the sum of weights
/// for all matched patterns (unique patterns only).
#[pyfunction]
#[pyo3(signature = (text, patterns, weights, case_insensitive=true))]
pub fn weighted_pattern_score(
    text: &str,
    patterns: Vec<String>,
    weights: Vec<f64>,
    case_insensitive: bool,
) -> PyResult<f64> {
    if patterns.len() != weights.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Number of patterns ({}) must match number of weights ({})",
            patterns.len(),
            weights.len()
        )));
    }

    let ac = cached_automaton(&patterns, case_insensitive, true).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("failed to build pattern automaton")
    })?;

    let mut seen = vec![false; patterns.len()];
    let mut score = 0.0;

    for m in ac.find_iter(text) {
        let idx = m.pattern().as_usize();
        if !seen[idx] {
            seen[idx] = true;
            score += weights[idx];
        }
    }

    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_matcher_creation() {
        let patterns = vec!["hello".to_string(), "world".to_string()];
        let matcher = PatternMatcher::new(patterns, true).unwrap();
        assert_eq!(matcher.pattern_count(), 2);
    }

    #[test]
    fn test_find_all() {
        let patterns = vec!["hello".to_string(), "world".to_string()];
        let matcher = PatternMatcher::new(patterns, true).unwrap();
        let matches = matcher.find_all("Hello World Hello");
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_contains_any() {
        let patterns = vec!["hello".to_string(), "world".to_string()];
        let matcher = PatternMatcher::new(patterns, true).unwrap();
        assert!(matcher.contains_any("Hello there"));
        assert!(!matcher.contains_any("Goodbye"));
    }

    #[test]
    fn test_case_insensitive() {
        let patterns = vec!["HELLO".to_string()];
        let matcher = PatternMatcher::new(patterns, true).unwrap();
        assert!(matcher.contains_any("hello"));

        let patterns2 = vec!["HELLO".to_string()];
        let matcher2 = PatternMatcher::new(patterns2, false).unwrap();
        assert!(!matcher2.contains_any("hello"));
    }

    #[test]
    fn test_weighted_score() {
        let patterns = vec![
            "read".to_string(),
            "write".to_string(),
            "analyze".to_string(),
        ];
        let weights = vec![1.0, 2.0, 3.0];
        let score = weighted_pattern_score("Please read and analyze this", patterns, weights, true)
            .unwrap();
        assert_eq!(score, 4.0); // read (1.0) + analyze (3.0)
    }

    #[test]
    fn test_cached_free_functions_consistent() {
        let patterns = vec!["read".to_string(), "write".to_string()];
        // Repeated calls hit the cache and must return identical results.
        let a = find_all_patterns("read and write and read", patterns.clone(), true);
        let b = find_all_patterns("read and write and read", patterns.clone(), true);
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), 3);
        assert!(contains_any_pattern("please read", patterns.clone(), true));
        assert!(!contains_any_pattern(
            "nothing here",
            patterns.clone(),
            true
        ));
        // Standard match_kind: "read" x2 + "write" x1 = 3.
        assert_eq!(
            count_pattern_matches("read read write", patterns.clone(), true),
            3
        );
        // Same patterns, different match_kind must not collide in the cache.
        assert_eq!(
            get_matched_pattern_indices("write only", patterns, true),
            vec![1]
        );
    }
}
