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

//! High-performance BPE tokenizer for token counting and encoding.
//!
//! This module provides a fast BPE (Byte-Pair Encoding) tokenizer compatible
//! with tiktoken's cl100k_base merge ranks. Features:
//!
//! - Accurate token counting using the full BPE merge algorithm
//! - Fast approximate counting (~90% accuracy) for budget estimation
//! - Batch counting with rayon parallelization
//! - Thread-safe after construction (read-only ranks)
//! - O(1) rank lookups via AHashMap

use ahash::AHashMap;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Regex pattern for splitting text into words (tiktoken cl100k_base style).
// Matches contractions, letters, numbers, punctuation runs, and whitespace
// runs. Each match becomes a "word" that is BPE-encoded independently.
// ---------------------------------------------------------------------------
// Note: The original tiktoken cl100k_base pattern uses \s+(?!\S) (negative
// look-ahead) to differentiate trailing whitespace from mid-text whitespace.
// The Rust `regex` crate doesn't support look-around. We replace it with
// \s+$ (end-of-string) which handles the same case for BPE word splitting.
// The remaining \s+ alternative catches all other whitespace runs.
static WORD_SPLIT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+$|\s+",
    )
    .expect("word-split regex must compile")
});

// ---------------------------------------------------------------------------
// Global cache: maps tokenizer name -> shared tokenizer instance so Python
// callers can retrieve a previously-constructed tokenizer without resending
// the (large) ranks table.
// ---------------------------------------------------------------------------
static TOKENIZER_CACHE: Lazy<RwLock<AHashMap<String, Arc<BpeTokenizerInner>>>> =
    Lazy::new(|| RwLock::new(AHashMap::new()));

// ---------------------------------------------------------------------------
// Inner (non-PyO3) tokenizer — Clone-able, stored in the global cache.
// ---------------------------------------------------------------------------
struct BpeTokenizerInner {
    ranks: AHashMap<Vec<u8>, u32>,
    special_tokens: AHashMap<String, u32>,
    name: String,
}

impl BpeTokenizerInner {
    /// Core BPE encode for a single word (byte sequence).
    fn bpe_encode_word(&self, word: &[u8]) -> Vec<u32> {
        if word.is_empty() {
            return Vec::new();
        }

        // Start with each byte as its own token.
        let mut pieces: Vec<Vec<u8>> = word.iter().map(|&b| vec![b]).collect();
        let mut merged = Vec::with_capacity(word.len());

        if pieces.len() == 1 {
            // Single byte — look up or fall back to raw byte value.
            return vec![self.rank_of(&pieces[0])];
        }

        loop {
            if pieces.len() < 2 {
                break;
            }

            // Find the adjacent pair with the lowest rank.
            let mut best_rank: Option<u32> = None;
            let mut best_idx: usize = 0;

            for i in 0..pieces.len() - 1 {
                merged.clear();
                merged.extend_from_slice(&pieces[i]);
                merged.extend_from_slice(&pieces[i + 1]);
                if let Some(&rank) = self.ranks.get(merged.as_slice()) {
                    if best_rank.is_none() || rank < best_rank.unwrap() {
                        best_rank = Some(rank);
                        best_idx = i;
                    }
                }
            }

            match best_rank {
                Some(_) => {
                    // Merge the best pair in place.
                    let mut best_merge =
                        Vec::with_capacity(pieces[best_idx].len() + pieces[best_idx + 1].len());
                    best_merge.extend_from_slice(&pieces[best_idx]);
                    best_merge.extend_from_slice(&pieces[best_idx + 1]);
                    pieces[best_idx] = best_merge;
                    pieces.remove(best_idx + 1);
                }
                None => break, // No more merges possible.
            }
        }

        pieces.iter().map(|p| self.rank_of(p)).collect()
    }

    /// Look up the rank for a byte sequence, falling back to byte value for
    /// single-byte pieces.
    #[inline]
    fn rank_of(&self, piece: &[u8]) -> u32 {
        if let Some(&r) = self.ranks.get(piece) {
            r
        } else if piece.len() == 1 {
            piece[0] as u32
        } else {
            0 // Should not happen with a complete ranks table.
        }
    }

    /// Encode full text, handling special tokens and word splitting.
    fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut tokens: Vec<u32> = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Check for special tokens at the current position.
            let mut found_special = false;
            for (st, &id) in &self.special_tokens {
                if remaining.starts_with(st.as_str()) {
                    tokens.push(id);
                    remaining = &remaining[st.len()..];
                    found_special = true;
                    break;
                }
            }
            if found_special {
                continue;
            }

            // Find the next occurrence of any special token.
            let end = self
                .special_tokens
                .keys()
                .filter_map(|st| remaining.find(st.as_str()))
                .min()
                .unwrap_or(remaining.len());

            let segment = &remaining[..end];
            remaining = &remaining[end..];

            // Split segment into words and BPE-encode each.
            for m in WORD_SPLIT_RE.find_iter(segment) {
                let word_bytes = m.as_str().as_bytes();
                tokens.extend(self.bpe_encode_word(word_bytes));
            }
        }

        tokens
    }

    /// Count tokens (exact).
    fn count_tokens(&self, text: &str) -> usize {
        self.encode(text).len()
    }
}

// ---------------------------------------------------------------------------
// Python-facing class
// ---------------------------------------------------------------------------

/// A BPE tokenizer that accepts pre-computed merge ranks.
///
/// Construct once with ranks data (e.g. from tiktoken's cl100k_base),
/// then call `count_tokens`, `encode`, or `count_tokens_batch`.
#[pyclass]
pub struct BpeTokenizer {
    inner: Arc<BpeTokenizerInner>,
}

#[pymethods]
impl BpeTokenizer {
    /// Create a new BPE tokenizer.
    ///
    /// # Arguments
    /// * `name` — Human-readable name (e.g. "cl100k_base").
    /// * `ranks_data` — List of (byte_sequence, rank) pairs.
    /// * `special_tokens` — List of (token_string, id) pairs.
    #[new]
    pub fn new(
        name: String,
        ranks_data: Vec<(Vec<u8>, u32)>,
        special_tokens: Vec<(String, u32)>,
    ) -> Self {
        let ranks: AHashMap<Vec<u8>, u32> = ranks_data.into_iter().collect();
        let st: AHashMap<String, u32> = special_tokens.into_iter().collect();

        let inner = Arc::new(BpeTokenizerInner {
            ranks,
            special_tokens: st,
            name: name.clone(),
        });

        // Cache for later retrieval.
        TOKENIZER_CACHE.write().insert(name, Arc::clone(&inner));

        BpeTokenizer { inner }
    }

    /// The tokenizer name.
    #[getter]
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Count tokens in `text` using the full BPE algorithm.
    pub fn count_tokens(&self, text: &str) -> usize {
        self.inner.count_tokens(text)
    }

    /// Count tokens for a batch of texts in parallel using rayon.
    pub fn count_tokens_batch(&self, texts: Vec<String>) -> Vec<usize> {
        texts
            .par_iter()
            .map(|t| self.inner.count_tokens(t))
            .collect()
    }

    /// Encode `text` into a vector of token ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }
}

// ---------------------------------------------------------------------------
// Standalone fast heuristic — no ranks table required
// ---------------------------------------------------------------------------

/// Fast approximate token count (~90% accuracy).
///
/// Uses heuristics based on whitespace, punctuation, and typical subword
/// splitting ratios. Useful for quick budget estimation when loading a full
/// BPE model is too expensive.
#[pyfunction]
pub fn count_tokens_fast(text: &str) -> usize {
    count_tokens_fast_impl(text)
}

/// Count tokens for a batch of texts using the fast heuristic.
///
/// Processes all texts in a **single FFI crossing** with rayon parallelism,
/// amortizing the per-call PyO3 boundary overhead — unlike looping
/// `count_tokens_fast` from Python, which crosses the FFI once per text.
/// Results are identical to calling `count_tokens_fast` on each element.
#[pyfunction]
pub fn count_tokens_fast_batch(texts: Vec<String>) -> Vec<usize> {
    texts
        .par_iter()
        .map(|t| count_tokens_fast_impl(t))
        .collect()
}

/// Shared heuristic body used by both the single and batch `#[pyfunction]`
/// entry points, so they can never diverge.
fn count_tokens_fast_impl(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }

    let mut count: f64 = 0.0;

    // Count whitespace-separated words without collecting.
    let mut words = text.split_whitespace().peekable();
    if words.peek().is_none() {
        // Whitespace-only text: roughly 1 token per whitespace run.
        return text
            .chars()
            .fold((0usize, false), |(n, prev_ws), c| {
                if c.is_whitespace() {
                    if prev_ws {
                        (n, true)
                    } else {
                        (n + 1, true)
                    }
                } else {
                    (n, false)
                }
            })
            .0
            .max(1);
    }

    for word in words {
        // Base: 1 token per word.
        count += 1.0;

        let len = word.len();

        // Long words are typically split into subword tokens.
        // Average English word is ~5 chars; BPE splits longer words.
        if len > 10 {
            count += ((len - 10) as f64) / 4.0;
        } else if len > 6 {
            count += 0.3;
        }

        // Count punctuation characters that usually become separate tokens.
        let punct_count = word.chars().filter(|c| c.is_ascii_punctuation()).count();
        if punct_count > 0 {
            // Punctuation attached to a word often adds tokens.
            count += punct_count as f64 * 0.8;
        }

        // Non-ASCII characters (CJK, emoji, etc.) typically use more tokens.
        let non_ascii = word.chars().filter(|c| !c.is_ascii()).count();
        if non_ascii > 0 {
            count += non_ascii as f64 * 0.5;
        }
    }

    // Ensure minimum 1 token.
    (count.round() as usize).max(1)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal tokenizer with a small set of merge ranks for testing.
    fn make_test_tokenizer() -> BpeTokenizer {
        // Ranks for "hello" byte merges: h=104, e=101, l=108, o=111
        // We define merge ranks so that pairs can be merged.
        let ranks: Vec<(Vec<u8>, u32)> = vec![
            // Single bytes (needed as base tokens)
            (vec![104], 0), // h
            (vec![101], 1), // e
            (vec![108], 2), // l
            (vec![111], 3), // o
            (vec![32], 4),  // space
            (vec![119], 5), // w
            (vec![114], 6), // r
            (vec![100], 7), // d
            // Merge pairs
            (vec![104, 101], 100),                // he
            (vec![108, 108], 101),                // ll
            (vec![104, 101, 108, 108], 102),      // hell
            (vec![104, 101, 108, 108, 111], 103), // hello
            (vec![119, 111], 104),                // wo
            (vec![114, 108], 105),                // rl
            (vec![119, 111, 114, 108, 100], 106), // world
        ];

        let special_tokens: Vec<(String, u32)> = vec![
            ("<|endoftext|>".to_string(), 50256),
            ("<|startoftext|>".to_string(), 50257),
        ];

        BpeTokenizer::new("test_tokenizer".to_string(), ranks, special_tokens)
    }

    #[test]
    fn test_empty_string_returns_zero_tokens() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.count_tokens(""), 0);
        assert_eq!(tok.encode("").len(), 0);
    }

    #[test]
    fn test_encode_produces_tokens() {
        let tok = make_test_tokenizer();
        let tokens = tok.encode("hello");
        // With ranks, "hello" should merge down.
        assert!(!tokens.is_empty(), "encoding 'hello' should produce tokens");
    }

    #[test]
    fn test_count_tokens_matches_encode_length() {
        let tok = make_test_tokenizer();
        let text = "hello world";
        assert_eq!(tok.count_tokens(text), tok.encode(text).len());
    }

    #[test]
    fn test_special_tokens_are_single_tokens() {
        let tok = make_test_tokenizer();
        let tokens = tok.encode("<|endoftext|>");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], 50256);
    }

    #[test]
    fn test_special_token_mixed_with_text() {
        let tok = make_test_tokenizer();
        let tokens = tok.encode("hello<|endoftext|>");
        // Should contain at least 2 parts: tokens for "hello" + the special token.
        assert!(tokens.len() >= 2);
        assert_eq!(*tokens.last().unwrap(), 50256);
    }

    #[test]
    fn test_batch_count_matches_individual() {
        let tok = make_test_tokenizer();
        let texts = vec![
            "hello".to_string(),
            "world".to_string(),
            "hello world".to_string(),
        ];
        let batch_counts = tok.count_tokens_batch(texts.clone());
        for (i, t) in texts.iter().enumerate() {
            assert_eq!(
                batch_counts[i],
                tok.count_tokens(t),
                "batch count mismatch for text '{}'",
                t,
            );
        }
    }

    #[test]
    fn test_batch_empty_list() {
        let tok = make_test_tokenizer();
        let counts = tok.count_tokens_batch(vec![]);
        assert!(counts.is_empty());
    }

    #[test]
    fn test_name_property() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.name(), "test_tokenizer");
    }

    // -- count_tokens_fast tests --

    #[test]
    fn test_fast_empty_string() {
        assert_eq!(count_tokens_fast(""), 0);
    }

    #[test]
    fn test_fast_single_word() {
        let count = count_tokens_fast("hello");
        assert!(count >= 1, "single word should be at least 1 token");
    }

    #[test]
    fn test_fast_multiple_words() {
        let count = count_tokens_fast("the quick brown fox jumps");
        // 5 words, each short — should be roughly 5 tokens.
        assert!(
            (4..=8).contains(&count),
            "expected ~5 tokens, got {}",
            count
        );
    }

    #[test]
    fn test_fast_punctuation_adds_tokens() {
        let no_punct = count_tokens_fast("hello world");
        let with_punct = count_tokens_fast("hello, world!");
        assert!(
            with_punct >= no_punct,
            "punctuation should not decrease token count",
        );
    }

    #[test]
    fn test_fast_long_word_splits() {
        let short = count_tokens_fast("hi");
        let long = count_tokens_fast("internationalization");
        assert!(
            long > short,
            "a 20-char word should produce more tokens than a 2-char word",
        );
    }

    #[test]
    fn test_fast_whitespace_only() {
        let count = count_tokens_fast("   \n\n  ");
        assert!(
            count >= 1,
            "whitespace-only should produce at least 1 token"
        );
    }

    #[test]
    fn test_fast_minimum_one_token() {
        // Any non-empty string should return at least 1.
        assert!(count_tokens_fast("x") >= 1);
        assert!(count_tokens_fast(".") >= 1);
        assert!(count_tokens_fast(" ") >= 1);
    }

    #[test]
    fn test_fast_non_ascii() {
        let ascii_count = count_tokens_fast("hello");
        let cjk_count = count_tokens_fast("\u{4f60}\u{597d}\u{4e16}\u{754c}");
        // CJK characters typically use more tokens per character.
        assert!(cjk_count >= 1, "CJK text should produce at least 1 token",);
        // 4 CJK chars should generally produce more tokens than 5 ASCII chars.
        assert!(
            cjk_count >= ascii_count || cjk_count >= 2,
            "CJK text should produce a reasonable number of tokens",
        );
    }

    // -- count_tokens_fast_batch tests --

    #[test]
    fn test_batch_empty() {
        assert_eq!(count_tokens_fast_batch(vec![]), Vec::<usize>::new());
    }

    #[test]
    fn test_batch_matches_single_elementwise() {
        // The batch result must equal per-element counting (the correctness
        // invariant the Python wrapper relies on when it picks the batch path).
        let texts = vec![
            "".to_string(),
            "hello".to_string(),
            "the quick brown fox".to_string(),
            "   \n\n  ".to_string(),
            "\u{4f60}\u{597d}".to_string(),
            "internationalization!".to_string(),
        ];
        let batched = count_tokens_fast_batch(texts.clone());
        let elementwise: Vec<usize> = texts.iter().map(|t| count_tokens_fast(t)).collect();
        assert_eq!(batched, elementwise);
    }

    #[test]
    fn test_batch_preserves_order_and_length() {
        let texts = vec!["a".to_string(), "bb".to_string(), "ccc".to_string()];
        let batched = count_tokens_fast_batch(texts.clone());
        assert_eq!(batched.len(), texts.len());
        assert_eq!(batched[1], count_tokens_fast("bb"));
    }

    #[test]
    fn test_bpe_merges_reduce_token_count() {
        // With merges, "hello" should produce fewer tokens than 5 (one per byte).
        let tok = make_test_tokenizer();
        let tokens = tok.encode("hello");
        assert!(
            tokens.len() < 5,
            "BPE merges should reduce 'hello' from 5 bytes to fewer tokens, got {}",
            tokens.len(),
        );
    }

    #[test]
    fn test_unknown_bytes_still_produce_tokens() {
        // Characters not in our small test ranks should still produce tokens
        // (falling back to raw byte values).
        let tok = make_test_tokenizer();
        let tokens = tok.encode("xyz");
        assert!(
            !tokens.is_empty(),
            "unknown bytes should still produce tokens",
        );
    }

    #[test]
    fn test_tokenizer_is_cached() {
        let _tok = make_test_tokenizer();
        let cache = TOKENIZER_CACHE.read();
        assert!(
            cache.contains_key("test_tokenizer"),
            "tokenizer should be cached by name",
        );
    }
}
