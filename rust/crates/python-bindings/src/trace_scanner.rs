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

//! Fast JSONL trace scanner for usage.jsonl files.
//!
//! Scans usage log files (plain text and gzip-compressed) and aggregates
//! per-session statistics for the GEPA prompt optimizer.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use flate2::read::GzDecoder;
use pyo3::prelude::*;
use serde_json::Value;

/// Per-session aggregated statistics from usage.jsonl scanning.
#[pyclass]
#[derive(Clone)]
pub struct SessionStats {
    #[pyo3(get)]
    pub session_id: String,
    #[pyo3(get)]
    pub tool_calls: u32,
    #[pyo3(get)]
    pub tool_failures: u32,
    #[pyo3(get)]
    pub task_type: String,
    #[pyo3(get)]
    pub tokens: u32,
    #[pyo3(get)]
    pub completion_score: f32,
}

/// Scan a single JSONL file (plain or gzip) and aggregate per-session stats.
///
/// Handles both `.jsonl` and `.jsonl.gz` files automatically based on extension.
/// Malformed lines are silently skipped.
#[pyfunction]
#[pyo3(signature = (file_path))]
pub fn scan_usage_file(file_path: &str) -> Vec<SessionStats> {
    let path = Path::new(file_path);
    if !path.exists() {
        return Vec::new();
    }

    let mut sessions: HashMap<String, RawSession> = HashMap::new();

    let result = if file_path.ends_with(".gz") {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let decoder = GzDecoder::new(file);
        let reader = BufReader::new(decoder);
        process_lines(reader, &mut sessions)
    } else {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let reader = BufReader::new(file);
        process_lines(reader, &mut sessions)
    };

    if result.is_err() {
        // Partial results are fine — return what we have
    }

    // Convert to output format with completion scoring
    let mut stats: Vec<SessionStats> = sessions
        .into_iter()
        .filter(|(_, s)| s.tool_calls >= 2) // Skip trivially broken sessions
        .map(|(sid, s)| {
            let failure_rate = s.tool_failures as f32 / s.tool_calls.max(1) as f32;
            let completion_score = (1.0 - failure_rate * 1.5).max(0.0);
            SessionStats {
                session_id: sid,
                tool_calls: s.tool_calls,
                tool_failures: s.tool_failures,
                task_type: s.task_type,
                tokens: s.tokens,
                completion_score,
            }
        })
        .collect();

    // Sort by completion score descending
    stats.sort_by(|a, b| {
        b.completion_score
            .partial_cmp(&a.completion_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    stats
}

/// Scan multiple JSONL files and merge results.
#[pyfunction]
#[pyo3(signature = (file_paths))]
pub fn scan_usage_files(file_paths: Vec<String>) -> Vec<SessionStats> {
    let mut all_sessions: HashMap<String, RawSession> = HashMap::new();

    for path_str in &file_paths {
        let path = Path::new(path_str);
        if !path.exists() {
            continue;
        }

        let _ = if path_str.ends_with(".gz") {
            let file = match File::open(path) {
                Ok(f) => f,
                Err(_) => continue,
            };
            process_lines(BufReader::new(GzDecoder::new(file)), &mut all_sessions)
        } else {
            let file = match File::open(path) {
                Ok(f) => f,
                Err(_) => continue,
            };
            process_lines(BufReader::new(file), &mut all_sessions)
        };
    }

    let mut stats: Vec<SessionStats> = all_sessions
        .into_iter()
        .filter(|(_, s)| s.tool_calls >= 2)
        .map(|(sid, s)| {
            let failure_rate = s.tool_failures as f32 / s.tool_calls.max(1) as f32;
            let completion_score = (1.0 - failure_rate * 1.5).max(0.0);
            SessionStats {
                session_id: sid,
                tool_calls: s.tool_calls,
                tool_failures: s.tool_failures,
                task_type: s.task_type,
                tokens: s.tokens,
                completion_score,
            }
        })
        .collect();

    stats.sort_by(|a, b| {
        b.completion_score
            .partial_cmp(&a.completion_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    stats
}

// Internal accumulator
struct RawSession {
    tool_calls: u32,
    tool_failures: u32,
    task_type: String,
    tokens: u32,
}

fn process_lines<R: BufRead>(
    reader: R,
    sessions: &mut HashMap<String, RawSession>,
) -> Result<(), std::io::Error> {
    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => continue,
        };

        let event: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let sid = match event.get("session_id").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let etype = event
            .get("event_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let session = sessions.entry(sid).or_insert_with(|| RawSession {
            tool_calls: 0,
            tool_failures: 0,
            task_type: "default".to_string(),
            tokens: 0,
        });

        match etype {
            "tool_call" => {
                session.tool_calls += 1;
            }
            "tool_result" => {
                let data = event.get("data").unwrap_or(&Value::Null);
                let success = data
                    .get("success")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                if !success {
                    session.tool_failures += 1;
                }
            }
            "task_classification" => {
                if let Some(data) = event.get("data") {
                    if let Some(tt) = data.get("task_type").and_then(|v| v.as_str()) {
                        session.task_type = tt.to_string();
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_path() {
        let result = scan_usage_file("/nonexistent/path.jsonl");
        assert!(result.is_empty());
    }
}
