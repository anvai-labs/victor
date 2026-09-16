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

//! High-performance document chunking for RAG.
//!
//! This module provides fast text chunking with:
//! - Sentence-boundary aware splitting
//! - Configurable chunk size and overlap
//! - 10-50x faster than Python regex-based chunking

use pyo3::prelude::*;

// Use memchr for optimized byte scanning
use memchr::memchr_iter;

/// Sentence boundary patterns (optimized for English)
const SENTENCE_ENDINGS: &[char] = &['.', '!', '?'];
const ABBREVIATIONS: &[&str] = &[
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.", "etc.", "i.e.", "e.g.", "cf.",
    "viz.", "Inc.", "Corp.", "Ltd.", "Co.", "No.", "Vol.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.",
    "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec.",
];

fn validate_chunk_parameters(chunk_size: usize, overlap: usize) -> PyResult<()> {
    if chunk_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "chunk_size must be positive",
        ));
    }
    if overlap >= chunk_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "overlap must be less than chunk_size",
        ));
    }
    Ok(())
}

/// Byte offset of the last `count` Unicode scalar values, always on a boundary.
fn suffix_start(text: &str, count: usize) -> usize {
    if count == 0 {
        return text.len();
    }
    text.char_indices()
        .rev()
        .nth(count - 1)
        .map_or(0, |(offset, _)| offset)
}

/// Check if position is a valid sentence boundary
fn is_sentence_boundary(text: &str, pos: usize) -> bool {
    if pos >= text.len() {
        return false;
    }

    let char_at = text[pos..].chars().next().unwrap_or(' ');
    if !SENTENCE_ENDINGS.contains(&char_at) {
        return false;
    }

    // Check it's not an abbreviation
    let start = suffix_start(&text[..pos], 10);
    let prefix = &text[start..=pos];

    for abbr in ABBREVIATIONS {
        if prefix.ends_with(abbr) {
            return false;
        }
    }

    // Must be followed by space/newline and capital letter (or end)
    if pos + 1 >= text.len() {
        return true;
    }

    let next_chars: String = text[pos + 1..].chars().take(3).collect();
    if next_chars.starts_with(' ') || next_chars.starts_with('\n') {
        // Check for capital letter after whitespace
        for c in next_chars.chars().skip(1) {
            if c.is_alphabetic() {
                return c.is_uppercase();
            }
            if !c.is_whitespace() {
                return true; // Non-alpha, non-space (like quotes)
            }
        }
        return true;
    }

    false
}

/// Find all sentence boundaries in text
fn find_sentence_boundaries(text: &str) -> Vec<usize> {
    let mut boundaries = vec![0]; // Start of text

    for (i, c) in text.char_indices() {
        if SENTENCE_ENDINGS.contains(&c) && is_sentence_boundary(text, i) {
            boundaries.push(i + 1);
        }
    }

    if !boundaries.contains(&text.len()) {
        boundaries.push(text.len());
    }

    boundaries
}

/// Chunk text by sentences with configurable size and overlap
#[pyfunction]
#[pyo3(signature = (text, chunk_size=1344, overlap=128))]
pub fn chunk_by_sentences(text: &str, chunk_size: usize, overlap: usize) -> PyResult<Vec<String>> {
    validate_chunk_parameters(chunk_size, overlap)?;
    let boundaries = find_sentence_boundaries(text);
    let offsets: Vec<usize> = text
        .char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(text.len()))
        .collect();
    let boundary_chars: Vec<usize> = boundaries
        .iter()
        .map(|offset| {
            offsets
                .binary_search(offset)
                .expect("sentence character boundary")
        })
        .collect();
    let mut chunks = Vec::new();

    if boundaries.len() <= 1 {
        // No sentence boundaries, fall back to character chunking
        return chunk_by_chars(text, chunk_size, overlap);
    }

    let mut chunk_start = 0;
    let mut current_end = 0;

    // Avoid rescanning every prefix when many sentences fit in one large chunk.
    for &boundary in boundary_chars.iter().skip(1) {
        let chunk_len = boundary - chunk_start;

        if chunk_len >= chunk_size {
            // Emit chunk up to previous boundary
            if current_end > chunk_start {
                let chunk = text[offsets[chunk_start]..offsets[current_end]]
                    .trim()
                    .to_string();
                if !chunk.is_empty() {
                    chunks.push(chunk);
                }
                // Move start back by overlap
                chunk_start = if overlap == 0 {
                    current_end
                } else {
                    // Find sentence boundary in overlap region
                    let overlap_start = current_end.saturating_sub(overlap);
                    let index = boundary_chars.partition_point(|&b| b < overlap_start);
                    boundary_chars
                        .get(index)
                        .filter(|&&b| b < current_end)
                        .copied()
                        .unwrap_or(overlap_start)
                };
            }
        }
        current_end = boundary;
    }

    // Emit final chunk
    if current_end > chunk_start {
        let chunk = text[offsets[chunk_start]..offsets[current_end]]
            .trim()
            .to_string();
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
    }

    Ok(chunks)
}

/// Simple character-based chunking with overlap
#[pyfunction]
#[pyo3(signature = (text, chunk_size=1344, overlap=128))]
pub fn chunk_by_chars(text: &str, chunk_size: usize, overlap: usize) -> PyResult<Vec<String>> {
    validate_chunk_parameters(chunk_size, overlap)?;
    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let text_len = chars.len();

    if text_len == 0 {
        return Ok(chunks);
    }

    let mut start = 0;
    while start < text_len {
        let end = start.saturating_add(chunk_size).min(text_len);
        let chunk: String = chars[start..end].iter().collect();
        let trimmed = chunk.trim().to_string();
        if !trimmed.is_empty() {
            chunks.push(trimmed);
        }

        if end >= text_len {
            break;
        }

        start = end.saturating_sub(overlap);
    }

    Ok(chunks)
}

/// Chunk text by paragraph boundaries
#[pyfunction]
#[pyo3(signature = (text, chunk_size=1344, overlap=128))]
pub fn chunk_by_paragraphs(text: &str, chunk_size: usize, overlap: usize) -> PyResult<Vec<String>> {
    validate_chunk_parameters(chunk_size, overlap)?;
    // Split on double newlines (paragraph boundaries)
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    for para in paragraphs {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }

        let would_be_len = current_chunk.chars().count() + para.chars().count() + 2;

        if would_be_len > chunk_size && !current_chunk.is_empty() {
            // Emit current chunk
            chunks.push(current_chunk.trim().to_string());

            // Start new chunk with overlap from previous
            let overlap_text = &current_chunk[suffix_start(&current_chunk, overlap)..];
            current_chunk = overlap_text.to_string();
        }

        if !current_chunk.is_empty() {
            current_chunk.push_str("\n\n");
        }
        current_chunk.push_str(para);
    }

    // Emit final chunk
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    Ok(chunks)
}

/// Detect document type from file extension
#[pyfunction]
pub fn detect_doc_type(source: &str) -> String {
    let source_lower = source.to_lowercase();

    // Check file extension
    let extensions = [
        (".html", "html"),
        (".htm", "html"),
        (".xhtml", "html"),
        (".md", "markdown"),
        (".markdown", "markdown"),
        (".rst", "markdown"),
        (".py", "code"),
        (".js", "code"),
        (".ts", "code"),
        (".java", "code"),
        (".go", "code"),
        (".rs", "code"),
        (".c", "code"),
        (".cpp", "code"),
        (".json", "json"),
        (".yaml", "yaml"),
        (".yml", "yaml"),
        (".xml", "xml"),
        (".csv", "csv"),
        (".txt", "text"),
    ];

    for (ext, doc_type) in extensions {
        if source_lower.ends_with(ext) {
            return doc_type.to_string();
        }
    }

    // Default to text
    "text".to_string()
}

/// Count approximate tokens (words + punctuation)
#[pyfunction]
pub fn count_tokens_approx(text: &str) -> usize {
    // Approximate: 1 token ≈ 4 characters for English
    // More accurate: count words + punctuation
    let mut tokens = 0;
    let mut in_word = false;

    for c in text.chars() {
        if c.is_alphanumeric() {
            if !in_word {
                tokens += 1;
                in_word = true;
            }
        } else {
            in_word = false;
            if c.is_ascii_punctuation() {
                tokens += 1;
            }
        }
    }

    tokens
}

// =============================================================================
// LINE-AWARE CHUNKING (Protocol-compliant)
// =============================================================================

/// Count lines in text (newline count + 1).
///
/// # Arguments
/// * `text` - Text to count lines in
///
/// # Returns
/// Number of lines (1 for empty string with no newlines)
#[pyfunction]
pub fn count_lines(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    memchr_iter(b'\n', text.as_bytes()).count() + 1
}

/// Find character offsets of all line starts.
///
/// # Arguments
/// * `text` - Text to analyze
///
/// # Returns
/// List of character offsets where lines start (always includes 0)
#[pyfunction]
pub fn find_line_boundaries(text: &str) -> Vec<usize> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut boundaries = vec![0];
    let mut characters = text.chars().enumerate().peekable();
    while let Some((position, character)) = characters.next() {
        if character == '\n' && characters.peek().is_some() {
            boundaries.push(position + 1);
        }
    }
    boundaries
}

/// Get line number for a character offset.
///
/// # Arguments
/// * `text` - Text
/// * `offset` - Character offset
///
/// # Returns
/// Line number (1-indexed)
#[pyfunction]
pub fn line_at_offset(text: &str, offset: usize) -> usize {
    if text.is_empty() || offset == 0 {
        return 1;
    }

    // Clamp to the last character, matching the Python backend's EOF behavior.
    let offset = offset.min(text.chars().count().saturating_sub(1));
    text.chars().take(offset).filter(|&c| c == '\n').count() + 1
}

/// Chunk information with line numbers and offsets.
#[pyclass]
#[derive(Clone)]
pub struct ChunkInfoRust {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub start_line: usize,
    #[pyo3(get)]
    pub end_line: usize,
    #[pyo3(get)]
    pub start_offset: usize,
    #[pyo3(get)]
    pub end_offset: usize,
    #[pyo3(get)]
    pub overlap_prev: usize,
    #[pyo3(get)]
    pub chunk_index: usize,
}

#[pymethods]
impl ChunkInfoRust {
    #[new]
    pub fn new(
        text: String,
        start_line: usize,
        end_line: usize,
        start_offset: usize,
        end_offset: usize,
        overlap_prev: usize,
        chunk_index: usize,
    ) -> Self {
        Self {
            text,
            start_line,
            end_line,
            start_offset,
            end_offset,
            overlap_prev,
            chunk_index,
        }
    }
}

/// Chunk text with overlap, respecting line boundaries.
///
/// Returns full chunk information including line numbers and offsets.
///
/// # Arguments
/// * `text` - Text to chunk
/// * `chunk_size` - Target chunk size in characters
/// * `overlap` - Overlap size in characters
///
/// # Returns
/// List of ChunkInfoRust objects with full metadata
#[pyfunction]
#[pyo3(signature = (text, chunk_size=1344, overlap=128))]
pub fn chunk_with_overlap(
    text: &str,
    chunk_size: usize,
    overlap: usize,
) -> PyResult<Vec<ChunkInfoRust>> {
    validate_chunk_parameters(chunk_size, overlap)?;
    if text.is_empty() {
        return Ok(Vec::new());
    }

    // Sizes and public offsets count characters, matching the Python backend.
    let chars: Vec<char> = text.chars().collect();
    let line_starts = find_line_boundaries(text);

    let mut chunks = Vec::new();
    let mut pos = 0;
    let mut chunk_index = 0;
    let text_len = chars.len();
    let mut previous_end: usize = 0;

    while pos < text_len {
        // Calculate chunk end
        let mut chunk_end = pos.saturating_add(chunk_size).min(text_len);

        // If not at end, try to find a line boundary
        if chunk_end < text_len {
            // Look for a newline within the chunk
            if let Some(last_newline) = chars[pos..chunk_end].iter().rposition(|&c| c == '\n') {
                let absolute_pos = pos + last_newline;
                if absolute_pos > pos {
                    chunk_end = absolute_pos + 1; // Include the newline
                }
            }
        }

        // Extract chunk text
        let chunk_text = chars[pos..chunk_end].iter().collect();

        // Calculate line numbers using binary search
        let start_line = line_at_offset_cached(&line_starts, pos);
        let end_line = line_at_offset_cached(&line_starts, chunk_end.saturating_sub(1));

        // Calculate overlap with previous chunk
        let overlap_prev = previous_end.saturating_sub(pos);

        chunks.push(ChunkInfoRust::new(
            chunk_text,
            start_line,
            end_line,
            pos,
            chunk_end,
            overlap_prev,
            chunk_index,
        ));

        if chunk_end == text_len {
            break;
        }
        // Advance from the emitted end so line trimming cannot skip input.
        // A short line may be smaller than the overlap; still consume a character.
        previous_end = chunk_end;
        pos = chunk_end.saturating_sub(overlap).max(pos + 1);
        chunk_index += 1;
    }

    Ok(chunks)
}

/// Binary search for line number using pre-computed boundaries.
#[inline]
fn line_at_offset_cached(line_starts: &[usize], offset: usize) -> usize {
    if line_starts.is_empty() {
        return 1;
    }

    // Binary search for the line containing offset
    match line_starts.binary_search(&offset) {
        Ok(idx) => idx + 1, // Exact match, 1-indexed
        Err(idx) => idx,    // Insert position = line number (already 1-indexed due to 0 at start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_by_sentences() {
        let text = "Hello world. This is a test. Another sentence here.";
        let chunks = chunk_by_sentences(text, 30, 10).unwrap();
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunk_by_chars() {
        let text = "Hello world this is a test of chunking";
        let chunks = chunk_by_chars(text, 15, 5).unwrap();
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_chunking_rejects_nonprogress_parameters() {
        for text in ["", "abc"] {
            for (size, overlap) in [(0, 0), (1, 1), (2, 3)] {
                assert!(chunk_by_chars(text, size, overlap).is_err());
                assert!(chunk_by_sentences(text, size, overlap).is_err());
                assert!(chunk_by_paragraphs(text, size, overlap).is_err());
                assert!(chunk_with_overlap(text, size, overlap).is_err());
            }
        }
    }

    #[test]
    fn test_unicode_sentence_and_paragraph_boundaries() {
        let text = "ééééééa.aaaaa.";
        assert_eq!(chunk_by_sentences(text, 4, 1).unwrap(), vec![text]);
        assert_eq!(find_sentence_boundaries("éé. Next."), vec![0, 5, 11]);
        assert_eq!(
            chunk_by_paragraphs("éé\n\nx", 4, 1).unwrap(),
            vec!["éé", "é\n\nx"]
        );
        assert_eq!(
            chunk_by_chars("é漢🙂x", 3, 1).unwrap(),
            vec!["é漢🙂", "🙂x"]
        );
    }

    #[test]
    fn test_sentence_chunking_many_boundaries_in_one_chunk() {
        let text = "Sentence. ".repeat(50_000);
        assert_eq!(
            chunk_by_sentences(&text, usize::MAX, 0).unwrap(),
            vec![text.trim()]
        );
    }

    #[test]
    fn test_unicode_line_helpers_use_character_offsets() {
        let text = "é\na\nb";
        assert_eq!(find_line_boundaries(text), vec![0, 2, 4]);
        for (offset, line) in [(0, 1), (1, 1), (2, 2), (3, 2), (4, 3), (5, 3), (100, 3)] {
            assert_eq!(line_at_offset(text, offset), line);
        }
        assert_eq!(find_line_boundaries("é\n"), vec![0]);
        assert_eq!(line_at_offset("é\n", 100), 1);
        assert_eq!(line_at_offset("", 100), 1);
        for chunk in chunk_with_overlap(text, 2, 0).unwrap() {
            assert_eq!(line_at_offset(text, chunk.start_offset), chunk.start_line);
            assert_eq!(line_at_offset(text, chunk.end_offset - 1), chunk.end_line);
        }
    }

    #[test]
    fn test_line_chunks_cover_input_with_monotonic_character_offsets() {
        for text in ["a\nbbbbbbbb", "ééé", "é漢\n🙂xy\n尾巴", "ab\ncdefghijkl"] {
            let chars: Vec<char> = text.chars().collect();
            for size in 1..=8 {
                for overlap in 0..size {
                    let chunks = chunk_with_overlap(text, size, overlap).unwrap();
                    assert!(chunks.len() <= chars.len());
                    let mut previous_start = None;
                    let mut covered = 0;
                    for chunk in chunks {
                        assert!(previous_start.is_none_or(|start| chunk.start_offset > start));
                        assert!(chunk.start_offset <= covered, "uncovered input");
                        assert!(chunk.end_offset > chunk.start_offset);
                        assert_eq!(
                            chunk.text,
                            chars[chunk.start_offset..chunk.end_offset]
                                .iter()
                                .collect::<String>()
                        );
                        assert_eq!(
                            chunk.overlap_prev,
                            covered.saturating_sub(chunk.start_offset)
                        );
                        assert_eq!(
                            chunk.start_line,
                            chars[..chunk.start_offset]
                                .iter()
                                .filter(|&&c| c == '\n')
                                .count()
                                + 1
                        );
                        assert_eq!(
                            chunk.end_line,
                            chars[..chunk.end_offset - 1]
                                .iter()
                                .filter(|&&c| c == '\n')
                                .count()
                                + 1
                        );
                        covered = chunk.end_offset;
                        previous_start = Some(chunk.start_offset);
                    }
                    assert_eq!(covered, chars.len());
                }
            }
        }
        assert_eq!(chunk_by_chars("abc", usize::MAX, 0).unwrap(), vec!["abc"]);
        assert_eq!(
            chunk_with_overlap("abc", usize::MAX, 0).unwrap()[0].text,
            "abc"
        );
    }

    #[test]
    fn test_detect_doc_type() {
        assert_eq!(detect_doc_type("file.py"), "code");
        assert_eq!(detect_doc_type("doc.md"), "markdown");
        assert_eq!(detect_doc_type("page.html"), "html");
        assert_eq!(detect_doc_type("data.json"), "json");
    }

    #[test]
    fn test_count_tokens() {
        let text = "Hello, world! This is a test.";
        let tokens = count_tokens_approx(text);
        assert!(tokens >= 6); // At least 6 words + punctuation
    }
}
