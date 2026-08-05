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

//! High-performance signature hashing for loop detection.
//!
//! This module provides fast signature computation for tool call sequences,
//! used by the loop detector to identify repetitive patterns.
//!
//! Features:
//! - xxHash3 for fast, quality hashing
//! - Batch processing for multiple tool calls
//! - Signature similarity computation for fuzzy matching

use pyo3::prelude::*;
use pyo3::types::PyDict;
use smallvec::SmallVec;
use xxhash_rust::xxh3::xxh3_128;

/// Compute a signature hash for a tool call.
///
/// The signature captures the tool name and a normalized representation
/// of its arguments, allowing detection of repeated identical calls.
///
/// # Arguments
/// * `tool_name` - Name of the tool being called
/// * `arguments` - Tool arguments as a Python dictionary
///
/// # Returns
/// 16-character hex string signature
///
/// # Note
/// Arguments are serialized in a consistent order (sorted keys) to ensure
/// identical logical calls produce identical signatures.
#[pyfunction]
pub fn compute_signature(tool_name: &str, arguments: &Bound<'_, PyDict>) -> PyResult<String> {
    let args_str = serialize_dict_stable(arguments)?;
    let combined = format!("{}:{}", tool_name, args_str);
    let hash = xxh3_128(combined.as_bytes());
    Ok(format!("{:016x}", hash as u64))
}

/// Serialize a Python dict to a stable string representation.
///
/// Keys are sorted to ensure deterministic output regardless of
/// insertion order.
fn serialize_dict_stable(dict: &Bound<'_, PyDict>) -> PyResult<String> {
    // Extract key-value pairs and sort by key
    let mut pairs: SmallVec<[(String, String); 16]> = SmallVec::new();

    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let value_str = python_value_to_string(&value)?;
        pairs.push((key_str, value_str));
    }

    // Sort by key for deterministic output
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    // Build output string
    let parts: Vec<String> = pairs
        .into_iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect();

    Ok(parts.join(","))
}

/// Convert a Python value to a stable string representation.
fn python_value_to_string(value: &Bound<'_, PyAny>) -> PyResult<String> {
    // Try to extract as various types
    if let Ok(s) = value.extract::<String>() {
        return Ok(format!("\"{}\"", s.replace('"', "\\\"")));
    }
    if let Ok(n) = value.extract::<i64>() {
        return Ok(n.to_string());
    }
    if let Ok(n) = value.extract::<f64>() {
        return Ok(format!("{:.6}", n));
    }
    if let Ok(b) = value.extract::<bool>() {
        return Ok(if b { "true" } else { "false" }.to_string());
    }
    if value.is_none() {
        return Ok("null".to_string());
    }

    // For complex types, use Python's repr
    let repr = value.repr()?;
    Ok(repr.to_string())
}

/// Compute signatures for multiple tool calls in batch.
///
/// This is more efficient than calling compute_signature repeatedly
/// due to reduced Python/Rust boundary crossing overhead.
///
/// # Arguments
/// * `tool_calls` - List of (tool_name, arguments_dict) tuples
///
/// # Returns
/// List of signature strings, one per tool call
#[pyfunction]
pub fn compute_batch_signatures(
    tool_calls: Vec<(String, Bound<'_, PyDict>)>,
) -> PyResult<Vec<String>> {
    let mut signatures = Vec::with_capacity(tool_calls.len());

    for (tool_name, arguments) in tool_calls {
        let args_str = serialize_dict_stable(&arguments)?;
        let combined = format!("{}:{}", tool_name, args_str);
        let hash = xxh3_128(combined.as_bytes());
        signatures.push(format!("{:016x}", hash as u64));
    }

    Ok(signatures)
}

/// Compute similarity between two signatures.
///
/// Returns 1.0 for identical signatures, 0.0 for completely different.
/// Uses Hamming distance on the hex characters for a simple similarity metric.
///
/// # Arguments
/// * `sig1` - First signature
/// * `sig2` - Second signature
///
/// # Returns
/// Similarity score between 0.0 and 1.0
#[pyfunction]
pub fn signature_similarity(sig1: &str, sig2: &str) -> f64 {
    if sig1 == sig2 {
        return 1.0;
    }

    if sig1.len() != sig2.len() {
        return 0.0;
    }

    let matching: usize = sig1
        .chars()
        .zip(sig2.chars())
        .filter(|(a, b)| a == b)
        .count();

    matching as f64 / sig1.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signature_consistency() {
        // Same input should produce same signature
        let input1 = "test_tool:arg1=\"value1\",arg2=\"value2\"";
        let input2 = "test_tool:arg1=\"value1\",arg2=\"value2\"";
        assert_eq!(
            format!("{:016x}", xxh3_128(input1.as_bytes()) as u64),
            format!("{:016x}", xxh3_128(input2.as_bytes()) as u64)
        );
    }

    #[test]
    fn test_signature_difference() {
        // Different input should produce different signatures (usually)
        let input1 = "tool_a:arg=\"value\"";
        let input2 = "tool_b:arg=\"value\"";
        assert_ne!(
            format!("{:016x}", xxh3_128(input1.as_bytes()) as u64),
            format!("{:016x}", xxh3_128(input2.as_bytes()) as u64)
        );
    }

    #[test]
    fn test_signature_similarity_identical() {
        let sim = signature_similarity("1234567890abcdef", "1234567890abcdef");
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_signature_similarity_different() {
        let sim = signature_similarity("1234567890abcdef", "fedcba0987654321");
        assert!(sim < 0.5);
    }

    #[test]
    fn test_signature_similarity_partial() {
        // Half matching
        let sim = signature_similarity("1111111122222222", "1111111133333333");
        assert!((sim - 0.5).abs() < 0.1);
    }
}
