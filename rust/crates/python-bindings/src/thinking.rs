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

//! High-performance thinking pattern detection for breaking circular reasoning loops.
//!
//! This module provides fast detection of repetitive thinking patterns in agent
//! responses using:
//! - Aho-Corasick for O(n) multi-pattern circular phrase detection
//! - xxHash3 for fast content hashing (10x faster than MD5)
//! - Efficient Jaccard similarity with pre-computed keyword sets
//!
//! Features:
//! - Exact content hash detection
//! - Semantic similarity via keyword overlap
//! - Circular phrase pattern detection
//! - Stalling pattern detection (DeepSeek-specific)

use ahash::AHashSet;
use aho_corasick::AhoCorasick;
use lru::LruCache;
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::OnceLock;
use xxhash_rust::xxh3::xxh3_64;

/// Pre-compiled circular pattern matcher
static CIRCULAR_MATCHER: OnceLock<AhoCorasick> = OnceLock::new();

/// Circular thinking patterns
const CIRCULAR_PATTERNS: &[&str] = &[
    "let me read",
    "let me check",
    "let me look at",
    "let me examine",
    "let me see",
    "i need to read",
    "i need to check",
    "i need to look at",
    "i need to examine",
    "i need to see",
    "first let me",
    "now let me",
    "let me first",
    "let me start by",
    "i'll need to",
    "i will need to",
    "i'll have to",
    "i will have to",
    "let me actually use",
    "let me use the",
    "i'll actually read",
    "i'll read",
    "i will read",
    "i need to read",
    "now i'll",
    "now i will",
    "i should read",
    "i should examine",
    "i should check",
    "i should look",
    "let me continue",
    "let me proceed",
];

/// Stopwords to filter from keyword extraction
static STOPWORDS: OnceLock<AHashSet<&'static str>> = OnceLock::new();

const STOPWORD_LIST: &[&str] = &[
    "let", "me", "i", "the", "a", "an", "to", "and", "of", "in", "for", "is", "it", "this", "that",
    "with", "be", "on", "as", "at", "by", "from", "or", "but", "not", "are", "was", "were", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "must", "shall", "can", "need", "now", "just", "also", "very", "well", "here",
    "there", "when", "where", "what", "which", "who", "how", "why", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "nor", "only", "own", "same", "so",
    "than", "too", "about", "into", "through", "during", "before", "after", "above", "below",
    "between",
];

fn get_stopwords() -> &'static AHashSet<&'static str> {
    STOPWORDS.get_or_init(|| STOPWORD_LIST.iter().copied().collect())
}

fn get_circular_matcher() -> &'static AhoCorasick {
    CIRCULAR_MATCHER.get_or_init(|| {
        AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(CIRCULAR_PATTERNS)
            .expect("Failed to compile circular patterns")
    })
}

/// Stalling patterns (intent without action)
const STALLING_PATTERNS: &[&str] = &[
    "let me",
    "i'll",
    "i will",
    "i need to",
    "i should",
    "now",
    "first",
];

/// Stored thinking pattern for history tracking
/// Using Arc for keywords to enable cheap cloning
#[derive(Clone)]
struct ThinkingPattern {
    keywords: Arc<AHashSet<String>>,
}

/// Detection result
#[pyclass]
#[derive(Clone)]
pub struct PatternAnalysis {
    #[pyo3(get)]
    pub is_loop: bool,
    #[pyo3(get)]
    pub loop_type: String,
    #[pyo3(get)]
    pub similarity_score: f64,
    #[pyo3(get)]
    pub matching_patterns: usize,
    #[pyo3(get)]
    pub guidance: String,
    #[pyo3(get)]
    pub category: String,
}

/// High-performance thinking pattern detector.
///
/// Uses Aho-Corasick for O(n) pattern matching and xxHash3 for fast hashing.
#[pyclass]
pub struct ThinkingDetector {
    history: VecDeque<ThinkingPattern>,
    pattern_counts: ahash::AHashMap<u64, usize>,
    repetition_threshold: usize,
    similarity_threshold: f64,
    stalling_threshold: usize,
    window_size: usize,
    iteration: usize,
    consecutive_stalls: usize,
    // Statistics
    total_analyzed: usize,
    loops_detected: usize,
    exact_matches: usize,
    similar_matches: usize,
    stalling_detected: usize,
    // Keyword cache (using Arc for zero-copy retrieval)
    keyword_cache: LruCache<u64, Arc<AHashSet<String>>>,
}

#[pymethods]
impl ThinkingDetector {
    /// Create a new thinking detector.
    ///
    /// # Arguments
    /// * `repetition_threshold` - Count for exact repetition detection (default 3)
    /// * `similarity_threshold` - Jaccard similarity threshold (default 0.65)
    /// * `window_size` - Number of recent patterns to track (default 10)
    /// * `stalling_threshold` - Consecutive stalls before detection (default 2)
    #[new]
    #[pyo3(signature = (repetition_threshold = 3, similarity_threshold = 0.65, window_size = 10, stalling_threshold = 2))]
    pub fn new(
        repetition_threshold: usize,
        similarity_threshold: f64,
        window_size: usize,
        stalling_threshold: usize,
    ) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            pattern_counts: ahash::AHashMap::new(),
            repetition_threshold,
            similarity_threshold,
            stalling_threshold,
            window_size,
            iteration: 0,
            consecutive_stalls: 0,
            total_analyzed: 0,
            loops_detected: 0,
            exact_matches: 0,
            similar_matches: 0,
            stalling_detected: 0,
            keyword_cache: LruCache::new(NonZeroUsize::new(100).unwrap()),
        }
    }

    /// Record a thinking block and detect loops.
    ///
    /// Returns (is_loop, guidance_message) tuple.
    pub fn record_thinking(&mut self, content: &str) -> (bool, String) {
        self.total_analyzed += 1;
        self.iteration += 1;

        // Normalize whitespace and compute hash
        let normalized = normalize_content(content);
        let content_hash = xxh3_64(normalized.as_bytes());

        // Extract keywords (with caching)
        let keywords = self.extract_keywords_cached(content, content_hash);

        // Categorize the thinking
        let category = categorize_thinking(content);

        // Create pattern record (Arc clone is cheap ref-count bump)
        let pattern = ThinkingPattern {
            keywords: Arc::clone(&keywords),
        };

        // Check for stalling patterns first (DeepSeek-specific)
        if detect_stalling(content) {
            self.consecutive_stalls += 1;
            if self.consecutive_stalls >= self.stalling_threshold {
                self.loops_detected += 1;
                self.stalling_detected += 1;
                self.add_to_history(pattern);

                let guidance =
                    generate_guidance("stalling", self.consecutive_stalls, &category, 0.0);
                return (true, guidance);
            }
        } else {
            self.consecutive_stalls = 0;
        }

        // Check for exact repetition
        let count = {
            let entry = self.pattern_counts.entry(content_hash).or_insert(0);
            *entry += 1;
            *entry
        };

        if count >= self.repetition_threshold {
            self.loops_detected += 1;
            self.exact_matches += 1;
            self.add_to_history(pattern);

            let guidance = generate_guidance("exact_repetition", count, &category, 0.0);
            return (true, guidance);
        }

        // Check for semantic similarity with recent patterns
        let mut similar_count = 0;
        let mut max_similarity = 0.0f64;

        for prev in &self.history {
            let similarity = jaccard_similarity(&keywords, &prev.keywords);
            max_similarity = max_similarity.max(similarity);

            if similarity >= self.similarity_threshold {
                similar_count += 1;
            }
        }

        self.add_to_history(pattern);

        if similar_count >= self.repetition_threshold - 1 {
            self.loops_detected += 1;
            self.similar_matches += 1;

            let guidance = generate_guidance(
                "semantic_similarity",
                similar_count + 1,
                &category,
                max_similarity,
            );
            return (true, guidance);
        }

        (false, String::new())
    }

    /// Reset detector state for new task.
    pub fn reset(&mut self) {
        self.history.clear();
        self.pattern_counts.clear();
        self.iteration = 0;
        self.consecutive_stalls = 0;
    }

    /// Clear statistics but keep pattern history.
    pub fn clear_stats(&mut self) {
        self.total_analyzed = 0;
        self.loops_detected = 0;
        self.exact_matches = 0;
        self.similar_matches = 0;
        self.stalling_detected = 0;
    }

    /// Get detection statistics.
    pub fn get_stats(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("total_analyzed", self.total_analyzed)?;
            dict.set_item("loops_detected", self.loops_detected)?;
            dict.set_item("exact_matches", self.exact_matches)?;
            dict.set_item("similar_matches", self.similar_matches)?;
            dict.set_item("stalling_detected", self.stalling_detected)?;
            dict.set_item("consecutive_stalls", self.consecutive_stalls)?;
            dict.set_item(
                "detection_rate",
                if self.total_analyzed > 0 {
                    self.loops_detected as f64 / self.total_analyzed as f64
                } else {
                    0.0
                },
            )?;
            dict.set_item("history_size", self.history.len())?;
            dict.set_item("unique_patterns", self.pattern_counts.len())?;
            Ok(dict.unbind().into_any())
        })
    }
}

impl ThinkingDetector {
    fn add_to_history(&mut self, pattern: ThinkingPattern) {
        if self.history.len() >= self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(pattern);
    }

    fn extract_keywords_cached(&mut self, content: &str, hash: u64) -> Arc<AHashSet<String>> {
        // Check cache first (returns Arc, zero-copy on hit)
        if let Some(cached) = self.keyword_cache.get(&hash) {
            return Arc::clone(cached);
        }

        // Compute and cache with Arc for zero-copy retrieval
        let keywords = Arc::new(extract_keywords(content));
        self.keyword_cache.put(hash, Arc::clone(&keywords));
        keywords
    }
}

/// Normalize content for consistent hashing.
fn normalize_content(content: &str) -> String {
    let mut result = String::with_capacity(content.len());
    let mut prev_was_space = true;

    for c in content.chars() {
        if c.is_whitespace() {
            if !prev_was_space {
                result.push(' ');
                prev_was_space = true;
            }
        } else {
            result.push(c);
            prev_was_space = false;
        }
    }

    result.trim().to_string()
}

/// Extract significant keywords from text.
fn extract_keywords(text: &str) -> AHashSet<String> {
    let stopwords = get_stopwords();
    let mut keywords = AHashSet::new();

    // Simple word extraction
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        let lower = word.to_lowercase();
        if lower.len() >= 4 && !stopwords.contains(lower.as_str()) {
            keywords.insert(lower);
        }
    }

    keywords
}

/// Compute Jaccard similarity between two keyword sets.
/// Optimized to accept Arc references to avoid cloning.
fn jaccard_similarity(a: &Arc<AHashSet<String>>, b: &Arc<AHashSet<String>>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let intersection = a.intersection(b).count();
    let union = a.union(b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Detect circular phrases using Aho-Corasick.
#[pyfunction]
pub fn detect_circular_phrases(text: &str) -> bool {
    get_circular_matcher().find(text).is_some()
}

/// Detect stalling patterns (intent without action).
fn detect_stalling(text: &str) -> bool {
    // Check first line/sentence only
    let first_line = text.lines().next().unwrap_or("");
    let first_sentence = first_line.split('.').next().unwrap_or("").trim();

    if first_sentence.is_empty() {
        return false;
    }

    // Must START with a stalling pattern
    let lower = first_sentence.to_lowercase();
    STALLING_PATTERNS
        .iter()
        .any(|pattern| lower.starts_with(pattern))
}

/// Categorize thinking content.
fn categorize_thinking(text: &str) -> String {
    let lower = text.to_lowercase();

    if lower.contains("read") || lower.contains("file") || lower.contains("content") {
        "file_read".to_string()
    } else if lower.contains("search") || lower.contains("find") || lower.contains("look for") {
        "search".to_string()
    } else if lower.contains("understand") || lower.contains("analyze") || lower.contains("examine")
    {
        "analysis".to_string()
    } else if lower.contains("implement") || lower.contains("create") || lower.contains("write") {
        "implementation".to_string()
    } else {
        "general".to_string()
    }
}

/// Generate guidance message to break the loop.
fn generate_guidance(loop_type: &str, count: usize, category: &str, similarity: f64) -> String {
    let base_guidance = if loop_type == "stalling" {
        format!(
            "STALLING DETECTED: You've stated your intent {} times without taking action. ",
            count
        )
    } else if loop_type == "semantic_similarity" {
        format!(
            "LOOP DETECTED (semantic, {:.0}% similar): You've repeated this thought pattern {} times. ",
            similarity * 100.0, count
        )
    } else {
        format!(
            "LOOP DETECTED ({}): You've repeated this thought pattern {} times. ",
            loop_type, count
        )
    };

    let category_advice = if loop_type == "stalling" {
        match category {
            "file_read" => {
                "STOP saying 'let me read' - EXECUTE the read tool NOW. \
                 If you've already read the file, use that content."
            }
            "search" => {
                "STOP saying 'let me search' - EXECUTE the search tool NOW. \
                 State your query and run the search."
            }
            "analysis" => {
                "STOP saying 'let me analyze' - provide your analysis NOW. \
                 Use the information you have."
            }
            "implementation" => {
                "STOP planning - EXECUTE the edit/write tool NOW. \
                 Write the code based on what you know."
            }
            _ => {
                "STOP stating intent - TAKE ACTION NOW. \
                 Execute a tool or provide your response."
            }
        }
    } else {
        match category {
            "file_read" => {
                "You've already read this file. Use the content you have. \
                 If you need specific information, state what you're looking for."
            }
            "search" => {
                "You've already searched for this. \
                 Either use the results you have or try a different search query."
            }
            "analysis" => {
                "You've analyzed this enough. \
                 Proceed with your current understanding and take action."
            }
            "implementation" => {
                "Stop planning and start implementing. \
                 Write the code now based on what you know."
            }
            _ => {
                "Take a different approach or proceed with action. \
                 Repeated thinking without progress is unproductive."
            }
        }
    };

    format!("{}{}", base_guidance, category_advice)
}

/// Count circular patterns in text.
#[pyfunction]
pub fn count_circular_patterns(text: &str) -> usize {
    get_circular_matcher().find_iter(text).count()
}

/// Find all circular pattern matches.
#[pyfunction]
pub fn find_circular_patterns(text: &str) -> Vec<(usize, usize, String)> {
    get_circular_matcher()
        .find_iter(text)
        .map(|m| (m.start(), m.end(), text[m.start()..m.end()].to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_circular_phrases() {
        assert!(detect_circular_phrases("Let me read the file"));
        assert!(detect_circular_phrases("I need to check this code"));
        assert!(detect_circular_phrases("First let me review"));
        assert!(!detect_circular_phrases("The file contains data"));
    }

    #[test]
    fn test_detect_stalling() {
        assert!(detect_stalling("Let me read the file first"));
        assert!(detect_stalling("I'll check the code now"));
        assert!(detect_stalling("Now I need to analyze"));
        assert!(!detect_stalling("The code shows that"));
    }

    #[test]
    fn test_extract_keywords() {
        let keywords = extract_keywords("The authentication module needs refactoring");
        assert!(keywords.contains("authentication"));
        assert!(keywords.contains("module"));
        assert!(keywords.contains("needs"));
        assert!(keywords.contains("refactoring"));
        assert!(!keywords.contains("the")); // stopword
    }

    #[test]
    fn test_jaccard_similarity() {
        let a: Arc<AHashSet<String>> = Arc::new(
            ["read", "file", "content"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        let b: Arc<AHashSet<String>> = Arc::new(
            ["read", "file", "data"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );

        let sim = jaccard_similarity(&a, &b);
        assert!((sim - 0.5).abs() < 0.01); // 2/4 = 0.5
    }

    #[test]
    fn test_categorize_thinking() {
        assert_eq!(categorize_thinking("Let me read the file"), "file_read");
        assert_eq!(categorize_thinking("I need to search for"), "search");
        assert_eq!(categorize_thinking("Let me analyze this"), "analysis");
        assert_eq!(
            categorize_thinking("I'll implement the feature"),
            "implementation"
        );
        assert_eq!(categorize_thinking("Okay so"), "general");
    }

    #[test]
    fn test_thinking_detector_exact_repetition() {
        let mut detector = ThinkingDetector::new(3, 0.65, 10, 2);

        let (is_loop, _) = detector.record_thinking("Analyzing the auth module");
        assert!(!is_loop);

        let (is_loop, _) = detector.record_thinking("Analyzing the auth module");
        assert!(!is_loop);

        let (is_loop, guidance) = detector.record_thinking("Analyzing the auth module");
        assert!(is_loop);
        assert!(guidance.contains("LOOP DETECTED"));
    }

    #[test]
    fn test_thinking_detector_stalling() {
        let mut detector = ThinkingDetector::new(3, 0.65, 10, 2);

        let (is_loop, _) = detector.record_thinking("Let me read the file");
        assert!(!is_loop);

        let (is_loop, guidance) = detector.record_thinking("Let me check the code");
        assert!(is_loop);
        assert!(guidance.contains("STALLING DETECTED"));
    }

    #[test]
    fn test_thinking_detector_reset() {
        Python::initialize();
        let mut detector = ThinkingDetector::new(3, 0.65, 10, 2);

        detector.record_thinking("Test content");
        detector.record_thinking("Test content");

        detector.reset();

        let stats = detector.get_stats().unwrap();
        Python::attach(|py| {
            let dict = stats.bind(py).cast::<pyo3::types::PyDict>().unwrap();
            assert_eq!(
                dict.get_item("history_size")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                0
            );
        });
    }
}
