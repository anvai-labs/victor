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

//! High-performance content deduplication using rolling hash.
//!
//! This module provides fast block-based deduplication using xxHash3
//! for content hashing, which is significantly faster than MD5 while
//! providing good collision resistance for deduplication purposes.

use ahash::AHashSet;
use pyo3::prelude::*;
use smallvec::SmallVec;
use xxhash_rust::xxh3::xxh3_64;

/// Normalize a content block for consistent hashing.
///
/// This function:
/// - Strips leading/trailing whitespace
/// - Collapses multiple whitespace to single space
/// - Removes trailing punctuation
/// - Converts to lowercase
///
/// # Arguments
/// * `block` - The content block to normalize
///
/// # Returns
/// The normalized string
#[pyfunction]
pub fn normalize_block(block: &str) -> String {
    normalize_block_internal(block)
}

/// Internal normalize function without Python GIL overhead
#[inline]
fn normalize_block_internal(block: &str) -> String {
    let trimmed = block.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // Pre-allocate with reasonable capacity
    let mut result = String::with_capacity(trimmed.len());
    let mut prev_was_whitespace = false;

    for c in trimmed.chars() {
        if c.is_whitespace() {
            if !prev_was_whitespace {
                result.push(' ');
                prev_was_whitespace = true;
            }
        } else {
            result.push(c.to_ascii_lowercase());
            prev_was_whitespace = false;
        }
    }

    // Remove trailing punctuation
    let trimmed_result = result.trim_end_matches(['.', ',', ';', ':']);
    trimmed_result.to_string()
}

/// Compute xxHash3 hash of a normalized block, returning hex string.
#[inline]
fn hash_block(block: &str) -> String {
    let normalized = normalize_block_internal(block);
    let hash = xxh3_64(normalized.as_bytes());
    // Return first 12 hex chars (48 bits) - sufficient for dedup with low collision
    format!("{:012x}", hash & 0xFFFFFFFFFFFF)
}

/// Split content into logical blocks.
///
/// Blocks are separated by:
/// - Double newlines (paragraph breaks)
/// - Code block boundaries (``` ... ```)
///
/// Code blocks are preserved as single units.
fn split_into_blocks(content: &str) -> Vec<&str> {
    // Fast path for empty content
    if content.is_empty() {
        return Vec::new();
    }

    // All code-fence marker start offsets, precomputed once (linear scan).
    // Scanning the remaining tail for "```" on every block iteration made this
    // loop quadratic on large fence-free content, and an unclosed fence made it
    // spin forever: `next_code` landed on the marker at `pos`, `chunk_end` was
    // 0, and `pos` never advanced — a multi-megabyte model response with an
    // unterminated fence pinned a chat session at 100% CPU indefinitely
    // (observed 2026-11-06, qwen3.7-plus streaming through the stale 0.15.0
    // wheel).
    let fences: Vec<usize> = content.match_indices("```").map(|(i, _)| i).collect();

    let mut blocks: SmallVec<[&str; 32]> = SmallVec::new();
    let mut pos = 0;

    while pos < content.len() {
        let remaining = &content[pos..];

        // Check for code block start
        if remaining.starts_with("```") {
            // Find the end of the code block: first fence marker at or after
            // pos + 3 (same match the old `content[pos + 3..].find("```")`
            // produced, found here in O(log n) instead of a tail rescan).
            let close_idx = fences.partition_point(|&p| p < pos + 3);
            if let Some(&close) = fences.get(close_idx) {
                let block_end = close + 3;
                blocks.push(&content[pos..block_end]);
                pos = block_end;
                continue;
            }
            // Unclosed fence: treat the marker as plain text — the Python
            // fallback only pairs complete fences — and keep scanning past it
            // so the loop always advances.
        }

        // Find next paragraph break or code block. Boundary fence candidates
        // start at pos + 3 when the marker at pos is an unclosed fence, so the
        // marker itself can never produce chunk_end == 0.
        let search_from = if remaining.starts_with("```") {
            pos + 3
        } else {
            pos
        };
        let next_break = remaining.find("\n\n").unwrap_or(remaining.len());
        let fence_idx = fences.partition_point(|&p| p < search_from);
        let next_code = match fences.get(fence_idx) {
            Some(&p) => p - pos,
            None => remaining.len(),
        };

        let chunk_end = next_break.min(next_code);

        if chunk_end > 0 {
            let block = &remaining[..chunk_end];
            if !block.trim().is_empty() {
                blocks.push(block);
            }
        }

        // Move past the paragraph break if that's what we found
        if chunk_end == next_break && next_break < remaining.len() {
            pos += chunk_end;
            // Skip the newlines
            while pos < content.len() && content.as_bytes()[pos] == b'\n' {
                pos += 1;
            }
        } else {
            pos += chunk_end;
        }
    }

    blocks.into_vec()
}

/// Process content and compute hashes for all blocks.
///
/// Returns a list of (hash, block_content, is_duplicate) tuples.
///
/// # Arguments
/// * `content` - The full content to process
/// * `min_block_length` - Minimum length for a block to be considered for dedup
///
/// # Returns
/// List of tuples: (hash string, original block, is_duplicate boolean)
#[pyfunction]
#[pyo3(signature = (content, min_block_length = 50))]
pub fn rolling_hash_blocks(content: &str, min_block_length: usize) -> Vec<(String, String, bool)> {
    let blocks = split_into_blocks(content);
    let mut seen_hashes: AHashSet<String> = AHashSet::with_capacity(blocks.len());
    let mut results: Vec<(String, String, bool)> = Vec::with_capacity(blocks.len());

    for block in blocks {
        let block_str = block.to_string();

        // Short blocks are never marked as duplicates
        if block.trim().len() < min_block_length {
            results.push((String::new(), block_str, false));
            continue;
        }

        let block_hash = hash_block(block);
        let is_duplicate = !seen_hashes.insert(block_hash.clone());

        results.push((block_hash, block_str, is_duplicate));
    }

    results
}

/// Find duplicate blocks in content and return their indices.
///
/// # Arguments
/// * `content` - The content to analyze
/// * `min_block_length` - Minimum length for dedup consideration
///
/// # Returns
/// List of (block_index, hash) for duplicate blocks
#[pyfunction]
#[pyo3(signature = (content, min_block_length = 50))]
pub fn find_duplicate_blocks(content: &str, min_block_length: usize) -> Vec<(usize, String)> {
    let blocks = split_into_blocks(content);
    let mut seen_hashes: AHashSet<String> = AHashSet::with_capacity(blocks.len());
    let mut duplicates: Vec<(usize, String)> = Vec::new();

    for (idx, block) in blocks.iter().enumerate() {
        if block.trim().len() < min_block_length {
            continue;
        }

        let block_hash = hash_block(block);
        if !seen_hashes.insert(block_hash.clone()) {
            duplicates.push((idx, block_hash));
        }
    }

    duplicates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_block() {
        assert_eq!(normalize_block("  Hello   World  "), "hello world");
        assert_eq!(normalize_block("Test."), "test");
        assert_eq!(normalize_block("Test,"), "test");
        assert_eq!(normalize_block(""), "");
    }

    #[test]
    fn test_hash_block_consistency() {
        let hash1 = hash_block("Hello World");
        let hash2 = hash_block("  hello   world  ");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_split_into_blocks() {
        let content = "Block 1\n\nBlock 2\n\nBlock 3";
        let blocks = split_into_blocks(content);
        assert_eq!(blocks.len(), 3);
    }

    #[test]
    fn test_split_preserves_code_blocks() {
        let content = "Text before\n\n```rust\nfn main() {\n\n}\n```\n\nText after";
        let blocks = split_into_blocks(content);
        // Should preserve the code block as one unit
        let has_code_block = blocks.iter().any(|b| b.contains("```rust"));
        assert!(has_code_block);
    }

    #[test]
    fn test_split_unclosed_fence_terminates() {
        // Regression: an unterminated fence used to spin forever (chunk_end == 0,
        // pos never advanced). The marker must stay in the paragraph text, matching
        // the Python fallback, which only pairs complete fences.
        let content = "Intro paragraph.\n\n```rust\nfn main() {}";
        let blocks = split_into_blocks(content);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0], "Intro paragraph.");
        assert_eq!(blocks[1], "```rust\nfn main() {}");
    }

    #[test]
    fn test_split_lone_fence_marker() {
        let blocks = split_into_blocks("```");
        assert_eq!(blocks, vec!["```"]);
    }

    #[test]
    fn test_split_fence_mid_paragraph_starts_unit() {
        // A fence marker not aligned with a paragraph break still starts a code
        // block unit at the marker; preceding text becomes its own block.
        let content = "text```py\nx = 1```tail";
        let blocks = split_into_blocks(content);
        assert_eq!(blocks, vec!["text", "```py\nx = 1```", "tail"]);
    }

    #[test]
    fn test_split_large_fence_free_content_completes() {
        // ~8 MB of fence-free paragraphs must split in linear time; the old
        // implementation rescanned the remaining tail for "```" per block.
        let paragraph = "word ".repeat(20);
        let content = (vec![paragraph; 80_000]).join("\n\n");
        let blocks = split_into_blocks(&content);
        assert_eq!(blocks.len(), 80_000);
    }

    #[test]
    fn test_split_large_content_with_trailing_unclosed_fence() {
        // The exact shape of the 2026-11-06 livelock: many blocks, then one
        // unterminated fence. Must terminate with the marker kept as text.
        let paragraph = "word ".repeat(20);
        let content = format!("{}\n\n```", (vec![paragraph; 20]).join("\n\n"));
        let blocks = split_into_blocks(&content);
        assert_eq!(blocks.len(), 21);
        assert_eq!(blocks[20], "```");
    }

    #[test]
    fn test_rolling_hash_blocks() {
        let content = "First block\n\nSecond block\n\nFirst block";
        let results = rolling_hash_blocks(content, 5);
        assert_eq!(results.len(), 3);
        // Third block should be duplicate
        assert!(results[2].2);
    }

    #[test]
    fn test_short_blocks_not_duplicated() {
        let content = "Hi\n\nHi\n\nHi";
        let results = rolling_hash_blocks(content, 10);
        // Short blocks should not be marked as duplicates
        assert!(!results.iter().any(|(_, _, is_dup)| *is_dup));
    }

    #[test]
    fn test_find_duplicate_blocks() {
        let content =
            "This is a test paragraph.\n\nAnother paragraph here.\n\nThis is a test paragraph.";
        let duplicates = find_duplicate_blocks(content, 10);
        assert_eq!(duplicates.len(), 1);
        assert_eq!(duplicates[0].0, 2); // Third block is duplicate
    }
}
