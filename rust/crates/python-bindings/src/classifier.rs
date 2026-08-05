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

//! High-performance task classification using Aho-Corasick pattern matching.
//!
//! This module provides fast keyword-based task classification with:
//! - O(n) multi-pattern matching via Aho-Corasick
//! - Weighted keyword scoring
//! - Negation detection to prevent false positives
//! - Position-aware scoring (early matches weighted higher)
//!
//! Task types:
//! - ACTION: Execute, run, deploy, build
//! - ANALYSIS: Explore, review, understand, analyze
//! - GENERATION: Create, write, generate, implement
//! - SEARCH: Find, locate, grep, search
//! - EDIT: Modify, refactor, fix, update
//! - DEFAULT: Ambiguous or conversational

use aho_corasick::AhoCorasick;
use pyo3::prelude::*;
use std::sync::OnceLock;

/// Static category strings using Arc for zero-copy sharing
static CATEGORY_ACTION: &str = "action";
static CATEGORY_ANALYSIS: &str = "analysis";
static CATEGORY_GENERATION: &str = "generation";
static CATEGORY_SEARCH: &str = "search";
static CATEGORY_EDIT: &str = "edit";

/// Task type enumeration
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    Action = 0,
    Analysis = 1,
    Generation = 2,
    Search = 3,
    Edit = 4,
    Default = 5,
}

#[pymethods]
impl TaskType {
    fn __str__(&self) -> &'static str {
        match self {
            TaskType::Action => "action",
            TaskType::Analysis => "analysis",
            TaskType::Generation => "generation",
            TaskType::Search => "search",
            TaskType::Edit => "edit",
            TaskType::Default => "default",
        }
    }

    fn __repr__(&self) -> String {
        format!("TaskType.{}", self.__str__().to_uppercase())
    }
}

/// Action keywords with weights
const ACTION_PATTERNS: &[(&str, f32)] = &[
    ("execute", 1.0),
    ("apply", 1.0),
    ("run", 0.9),
    ("deploy", 1.0),
    ("build", 0.8),
    ("install", 0.9),
    ("start", 0.7),
    ("stop", 0.7),
    ("restart", 0.8),
    ("test", 0.8),
    ("commit", 0.9),
    ("push", 0.8),
    ("pull", 0.7),
    ("merge", 0.8),
];

/// Analysis keywords with weights
const ANALYSIS_PATTERNS: &[(&str, f32)] = &[
    ("analyze", 1.0),
    ("explore", 1.0),
    ("review", 0.9),
    ("understand", 0.9),
    ("explain", 0.9),
    ("describe", 0.8),
    ("investigate", 0.9),
    ("examine", 0.9),
    ("study", 0.8),
    ("assess", 0.8),
    ("evaluate", 0.8),
    ("summarize", 0.8),
    ("overview", 0.7),
    ("what is", 0.8),
    ("what does", 0.8),
    ("how does", 0.8),
    ("why does", 0.7),
];

/// Generation keywords with weights
const GENERATION_PATTERNS: &[(&str, f32)] = &[
    ("create", 1.0),
    ("generate", 1.0),
    ("write", 0.9),
    ("implement", 1.0),
    ("add", 0.8),
    ("new", 0.7),
    ("scaffold", 0.9),
    ("initialize", 0.8),
    ("setup", 0.8),
    ("bootstrap", 0.9),
];

/// Search keywords with weights
const SEARCH_PATTERNS: &[(&str, f32)] = &[
    ("find", 1.0),
    ("search", 1.0),
    ("locate", 0.9),
    ("grep", 1.0),
    ("look for", 0.9),
    ("where is", 0.9),
    ("which file", 0.8),
    ("list", 0.7),
];

/// Edit keywords with weights
const EDIT_PATTERNS: &[(&str, f32)] = &[
    ("modify", 1.0),
    ("refactor", 1.0),
    ("fix", 1.0),
    ("update", 0.9),
    ("change", 0.8),
    ("edit", 0.9),
    ("rename", 0.9),
    ("move", 0.7),
    ("delete", 0.8),
    ("remove", 0.8),
    ("replace", 0.9),
    ("improve", 0.8),
    ("optimize", 0.9),
    ("clean up", 0.8),
    ("debug", 0.9),
];

/// Negation patterns that reduce confidence
const NEGATION_PATTERNS: &[&str] = &[
    "don't",
    "do not",
    "dont",
    "shouldn't",
    "should not",
    "wouldn't",
    "would not",
    "can't",
    "cannot",
    "not",
    "never",
    "without",
    "avoid",
    "skip",
    "ignore",
];

// Pre-compiled matchers
static ACTION_MATCHER: OnceLock<AhoCorasick> = OnceLock::new();
static ANALYSIS_MATCHER: OnceLock<AhoCorasick> = OnceLock::new();
static GENERATION_MATCHER: OnceLock<AhoCorasick> = OnceLock::new();
static SEARCH_MATCHER: OnceLock<AhoCorasick> = OnceLock::new();
static EDIT_MATCHER: OnceLock<AhoCorasick> = OnceLock::new();
static NEGATION_MATCHER: OnceLock<AhoCorasick> = OnceLock::new();

fn get_action_matcher() -> &'static AhoCorasick {
    ACTION_MATCHER.get_or_init(|| {
        let patterns: Vec<&str> = ACTION_PATTERNS.iter().map(|(p, _)| *p).collect();
        AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(&patterns)
            .expect("Failed to compile action patterns")
    })
}

fn get_analysis_matcher() -> &'static AhoCorasick {
    ANALYSIS_MATCHER.get_or_init(|| {
        let patterns: Vec<&str> = ANALYSIS_PATTERNS.iter().map(|(p, _)| *p).collect();
        AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(&patterns)
            .expect("Failed to compile analysis patterns")
    })
}

fn get_generation_matcher() -> &'static AhoCorasick {
    GENERATION_MATCHER.get_or_init(|| {
        let patterns: Vec<&str> = GENERATION_PATTERNS.iter().map(|(p, _)| *p).collect();
        AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(&patterns)
            .expect("Failed to compile generation patterns")
    })
}

fn get_search_matcher() -> &'static AhoCorasick {
    SEARCH_MATCHER.get_or_init(|| {
        let patterns: Vec<&str> = SEARCH_PATTERNS.iter().map(|(p, _)| *p).collect();
        AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(&patterns)
            .expect("Failed to compile search patterns")
    })
}

fn get_edit_matcher() -> &'static AhoCorasick {
    EDIT_MATCHER.get_or_init(|| {
        let patterns: Vec<&str> = EDIT_PATTERNS.iter().map(|(p, _)| *p).collect();
        AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(&patterns)
            .expect("Failed to compile edit patterns")
    })
}

fn get_negation_matcher() -> &'static AhoCorasick {
    NEGATION_MATCHER.get_or_init(|| {
        AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(NEGATION_PATTERNS)
            .expect("Failed to compile negation patterns")
    })
}

/// Classification result with confidence and metadata
#[pyclass]
#[derive(Clone)]
pub struct ClassificationResult {
    #[pyo3(get)]
    pub task_type: TaskType,
    #[pyo3(get)]
    pub confidence: f64,
    #[pyo3(get)]
    pub is_action_task: bool,
    #[pyo3(get)]
    pub is_analysis_task: bool,
    #[pyo3(get)]
    pub is_generation_task: bool,
    #[pyo3(get)]
    pub needs_execution: bool,
    #[pyo3(get)]
    pub matched_count: usize,
    #[pyo3(get)]
    pub negated_count: usize,
    #[pyo3(get)]
    pub recommended_tool_budget: usize,
}

#[pymethods]
impl ClassificationResult {
    fn __repr__(&self) -> String {
        format!(
            "ClassificationResult(type={:?}, confidence={:.2}, matches={}, negated={})",
            self.task_type, self.confidence, self.matched_count, self.negated_count
        )
    }

    /// Convert to legacy dictionary format for backward compatibility.
    fn to_legacy_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);

            let coarse_type = match self.task_type {
                TaskType::Action | TaskType::Generation | TaskType::Edit => "action",
                TaskType::Analysis | TaskType::Search => "analysis",
                TaskType::Default => "default",
            };

            dict.set_item("is_action_task", self.is_action_task)?;
            dict.set_item("is_analysis_task", self.is_analysis_task)?;
            dict.set_item("needs_execution", self.needs_execution)?;
            dict.set_item("coarse_task_type", coarse_type)?;
            dict.set_item("confidence", self.confidence)?;
            dict.set_item("source", "native")?;
            dict.set_item("task_type", self.task_type.__str__())?;

            Ok(dict.into())
        })
    }
}

/// Fast task classifier using Aho-Corasick multi-pattern matching.
#[pyclass]
pub struct TaskClassifier {
    position_weight: f64,
    negation_radius: usize,
}

#[pymethods]
impl TaskClassifier {
    /// Create a new task classifier.
    ///
    /// # Arguments
    /// * `position_weight` - Weight factor for early matches (default 0.1)
    /// * `negation_radius` - Character distance for negation detection (default 20)
    #[new]
    #[pyo3(signature = (position_weight = 0.1, negation_radius = 20))]
    pub fn new(position_weight: f64, negation_radius: usize) -> Self {
        Self {
            position_weight,
            negation_radius,
        }
    }

    /// Classify a task from user input text.
    ///
    /// Returns classification result with task type, confidence, and metadata.
    pub fn classify(&self, text: &str) -> ClassificationResult {
        // Find negation positions for context-aware matching
        let negation_positions: Vec<usize> = get_negation_matcher()
            .find_iter(text)
            .map(|m| m.start())
            .collect();

        // Score each category
        let action_score = self.score_category(
            text,
            get_action_matcher(),
            ACTION_PATTERNS,
            &negation_positions,
        );
        let analysis_score = self.score_category(
            text,
            get_analysis_matcher(),
            ANALYSIS_PATTERNS,
            &negation_positions,
        );
        let generation_score = self.score_category(
            text,
            get_generation_matcher(),
            GENERATION_PATTERNS,
            &negation_positions,
        );
        let search_score = self.score_category(
            text,
            get_search_matcher(),
            SEARCH_PATTERNS,
            &negation_positions,
        );
        let edit_score =
            self.score_category(text, get_edit_matcher(), EDIT_PATTERNS, &negation_positions);

        // Find winner
        let scores = [
            (TaskType::Action, action_score),
            (TaskType::Analysis, analysis_score),
            (TaskType::Generation, generation_score),
            (TaskType::Search, search_score),
            (TaskType::Edit, edit_score),
        ];

        let (task_type, score) = scores
            .iter()
            .max_by(|a, b| a.1.score.partial_cmp(&b.1.score).unwrap())
            .map(|(t, s)| (*t, s.clone()))
            .unwrap();

        // Compute confidence
        let total_score: f64 = scores.iter().map(|(_, s)| s.score).sum();
        let confidence = if total_score > 0.0 {
            score.score / total_score
        } else {
            0.0
        };

        // Determine final task type
        let final_type = if confidence < 0.3 {
            TaskType::Default
        } else {
            task_type
        };

        // Total matched and negated counts
        let matched_count: usize = scores.iter().map(|(_, s)| s.matched).sum();
        let negated_count: usize = scores.iter().map(|(_, s)| s.negated).sum();

        // Determine flags
        let is_action_task = matches!(
            final_type,
            TaskType::Action | TaskType::Generation | TaskType::Edit
        );
        let is_analysis_task = matches!(final_type, TaskType::Analysis | TaskType::Search);
        let needs_execution = matches!(final_type, TaskType::Action);

        // Recommended tool budget based on task type
        let recommended_tool_budget = match final_type {
            TaskType::Action => 15,
            TaskType::Analysis => 30,
            TaskType::Generation => 20,
            TaskType::Search => 10,
            TaskType::Edit => 15,
            TaskType::Default => 20,
        };

        ClassificationResult {
            task_type: final_type,
            confidence,
            is_action_task,
            is_analysis_task,
            is_generation_task: final_type == TaskType::Generation,
            needs_execution,
            matched_count,
            negated_count,
            recommended_tool_budget,
        }
    }
}

#[derive(Clone)]
struct CategoryScore {
    score: f64,
    matched: usize,
    negated: usize,
}

impl TaskClassifier {
    fn score_category(
        &self,
        text: &str,
        matcher: &AhoCorasick,
        patterns: &[(&str, f32)],
        negation_positions: &[usize],
    ) -> CategoryScore {
        let text_len = text.len() as f64;
        let mut score = 0.0;
        let mut matched = 0;
        let mut negated = 0;

        for mat in matcher.find_iter(text) {
            let pattern_idx = mat.pattern().as_usize();
            let (_, weight) = patterns[pattern_idx];

            // Check if negated
            let is_negated = negation_positions.iter().any(|&neg_pos| {
                let distance = if neg_pos < mat.start() {
                    mat.start() - neg_pos
                } else {
                    neg_pos - mat.start()
                };
                distance <= self.negation_radius && neg_pos < mat.start()
            });

            if is_negated {
                negated += 1;
                // Reduce score for negated matches
                score -= weight as f64 * 0.5;
            } else {
                matched += 1;
                // Position weighting: earlier matches score higher
                let position_bonus = if text_len > 0.0 {
                    1.0 + self.position_weight * (1.0 - mat.start() as f64 / text_len)
                } else {
                    1.0
                };
                score += weight as f64 * position_bonus;
            }
        }

        CategoryScore {
            score: score.max(0.0),
            matched,
            negated,
        }
    }
}

/// Classify a task using default settings (convenience function).
#[pyfunction]
pub fn classify_task(text: &str) -> ClassificationResult {
    let classifier = TaskClassifier::new(0.1, 20);
    classifier.classify(text)
}

/// Check if text contains action keywords.
#[pyfunction]
pub fn has_action_keywords(text: &str) -> bool {
    get_action_matcher().find(text).is_some()
}

/// Check if text contains analysis keywords.
#[pyfunction]
pub fn has_analysis_keywords(text: &str) -> bool {
    get_analysis_matcher().find(text).is_some()
}

/// Check if text contains generation keywords.
#[pyfunction]
pub fn has_generation_keywords(text: &str) -> bool {
    get_generation_matcher().find(text).is_some()
}

/// Check if text contains negation patterns.
#[pyfunction]
pub fn has_negation(text: &str) -> bool {
    get_negation_matcher().find(text).is_some()
}

/// Zero-copy internal keyword match result
#[derive(Clone, Copy)]
pub struct KeywordMatch<'a> {
    pub start: usize,
    pub end: usize,
    pub matched_text: &'a str,
    pub category: &'static str,
}

/// Find all keyword matches in text (zero-copy internal version).
/// This is more efficient for Rust-to-Rust calls as it avoids allocations.
pub fn find_all_keywords_internal(text: &str) -> Vec<KeywordMatch<'_>> {
    let mut results = Vec::new();

    for mat in get_action_matcher().find_iter(text) {
        results.push(KeywordMatch {
            start: mat.start(),
            end: mat.end(),
            matched_text: &text[mat.start()..mat.end()],
            category: CATEGORY_ACTION,
        });
    }

    for mat in get_analysis_matcher().find_iter(text) {
        results.push(KeywordMatch {
            start: mat.start(),
            end: mat.end(),
            matched_text: &text[mat.start()..mat.end()],
            category: CATEGORY_ANALYSIS,
        });
    }

    for mat in get_generation_matcher().find_iter(text) {
        results.push(KeywordMatch {
            start: mat.start(),
            end: mat.end(),
            matched_text: &text[mat.start()..mat.end()],
            category: CATEGORY_GENERATION,
        });
    }

    for mat in get_search_matcher().find_iter(text) {
        results.push(KeywordMatch {
            start: mat.start(),
            end: mat.end(),
            matched_text: &text[mat.start()..mat.end()],
            category: CATEGORY_SEARCH,
        });
    }

    for mat in get_edit_matcher().find_iter(text) {
        results.push(KeywordMatch {
            start: mat.start(),
            end: mat.end(),
            matched_text: &text[mat.start()..mat.end()],
            category: CATEGORY_EDIT,
        });
    }

    // Sort by position
    results.sort_by_key(|m| m.start);
    results
}

/// Find all keyword matches in text (Python FFI version with owned strings).
#[pyfunction]
pub fn find_all_keywords(text: &str) -> Vec<(usize, usize, String, String)> {
    find_all_keywords_internal(text)
        .into_iter()
        .map(|m| {
            (
                m.start,
                m.end,
                m.matched_text.to_string(),
                m.category.to_string(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_action() {
        let classifier = TaskClassifier::new(0.1, 20);
        let result = classifier.classify("Run the tests and deploy to production");
        assert_eq!(result.task_type, TaskType::Action);
        assert!(result.is_action_task);
    }

    #[test]
    fn test_classify_analysis() {
        let classifier = TaskClassifier::new(0.1, 20);
        let result = classifier.classify("Analyze the codebase for security issues");
        assert_eq!(result.task_type, TaskType::Analysis);
        assert!(result.is_analysis_task);
    }

    #[test]
    fn test_classify_generation() {
        let classifier = TaskClassifier::new(0.1, 20);
        let result = classifier.classify("Create a new authentication module");
        assert_eq!(result.task_type, TaskType::Generation);
        assert!(result.is_generation_task);
    }

    #[test]
    fn test_classify_search() {
        let classifier = TaskClassifier::new(0.1, 20);
        let result = classifier.classify("Find all usages of the deprecated API");
        assert_eq!(result.task_type, TaskType::Search);
    }

    #[test]
    fn test_classify_edit() {
        let classifier = TaskClassifier::new(0.1, 20);
        let result = classifier.classify("Refactor the database connection code");
        assert_eq!(result.task_type, TaskType::Edit);
    }

    #[test]
    fn test_negation_detection() {
        let classifier = TaskClassifier::new(0.1, 20);
        let result = classifier.classify("Don't run the tests yet");
        // Negation should reduce action score
        assert!(result.negated_count > 0);
    }

    #[test]
    fn test_has_keywords() {
        assert!(has_action_keywords("run the tests"));
        assert!(has_analysis_keywords("analyze this code"));
        assert!(has_generation_keywords("create a new file"));
        assert!(has_negation("don't do that"));
    }

    #[test]
    fn test_find_all_keywords() {
        let matches = find_all_keywords("Analyze the code and fix the bug");
        assert!(matches.len() >= 2);
        // Should find "analyze" and "fix"
    }

    #[test]
    fn test_low_confidence_default() {
        let classifier = TaskClassifier::new(0.1, 20);
        let result = classifier.classify("Hello, how are you?");
        // Ambiguous input should have low confidence
        assert!(result.confidence < 0.5 || result.task_type == TaskType::Default);
    }
}
