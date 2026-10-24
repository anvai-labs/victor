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

//! Native CCG statement extraction (native-acceleration-strategy, engine
//! phase 1 — the vertical slice).
//!
//! Reproduces `CodeContextGraphBuilder._extract_statement_nodes` exactly:
//! the full-tree walk that classifies statement nodes, extracts spans and
//! content previews, derives visibility, and generates the stable
//! sha256-16 node IDs. In Python this walk touches every tree-sitter node
//! through the GIL-bound binding — the measured hot core of the index
//! parse phase (py-tree-sitter `parse()` holds the GIL; 8 threads
//! measured 1.0x). Here the batch parses and walks under
//! `py.allow_threads` with rayon — real core parallelism, no result IPC.
//!
//! Parity contract: the differential tests walk the same fixtures through
//! the Python implementation and this one and compare record-for-record.
//! Unsupported languages return `Err` per file so the caller falls back
//! to the Python walk (the reference implementation stays).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use tree_sitter::Parser;

/// One extracted statement — the exact field set the Python walk feeds
/// into GraphNode construction (which stays in Python).
#[pyclass(get_all)]
#[derive(Clone, Debug, PartialEq)]
pub struct StatementRecord {
    /// Stable node id: sha256("path:start:end:type")[:16] hex.
    pub node_id: String,
    /// Tree-sitter node type string.
    pub node_type: String,
    /// StatementType value ("condition", "loop", ...) — UNKNOWN nodes are
    /// never emitted, matching the Python walk.
    pub statement_type: String,
    /// 1-based start line.
    pub start_line: i64,
    /// 1-based end line.
    pub end_line: i64,
    /// First 100 chars of the stripped start line ("" when out of range).
    pub content_preview: String,
    /// "public"/"private"/"protected"/None per the Python rules.
    pub visibility: Option<String>,
    /// Node id of the nearest enclosing emitted statement ("" at top level).
    pub scope_id: String,
}

#[pymethods]
impl StatementRecord {
    fn __repr__(&self) -> String {
        format!(
            "StatementRecord(node_type={}, stype={}, lines={}..{}, vis={:?})",
            self.node_type, self.statement_type, self.start_line, self.end_line, self.visibility
        )
    }
}

/// Classify a node type exactly like `_classify_statement`. UNKNOWN (None)
/// means the walk does not emit a node for it.
fn classify_statement(language: &str, node_type: &str) -> Option<&'static str> {
    match language {
        "python" => match node_type {
            "if_statement" | "elif_clause" => Some("condition"),
            "for_statement" | "while_statement" => Some("loop"),
            "try_statement" => Some("try"),
            "except_clause" | "except_block" => Some("catch"),
            "finally_clause" => Some("finally"),
            "match_statement" => Some("switch"),
            "match_clause" | "case_clause" => Some("case"),
            "assignment" | "augmented_assignment" | "named_expression" => Some("assignment"),
            "call" => Some("call"),
            "return_statement" => Some("return"),
            "yield_statement" => Some("yield"),
            "function_definition" | "async_function_definition" => Some("function_def"),
            "class_definition" => Some("class_def"),
            "expression_statement" | "expression_list" => Some("expression"),
            _ => None,
        },
        "javascript" | "typescript" => match node_type {
            "if_statement" | "else_clause" => Some("condition"),
            "for_statement" | "for_in_statement" | "while_statement" | "do_statement" => {
                Some("loop")
            }
            "try_statement" => Some("try"),
            "catch_clause" | "finally_clause" => Some("catch"),
            "switch_statement" => Some("switch"),
            "switch_case" | "switch_default" => Some("case"),
            "variable_declaration" | "lexical_declaration" | "assignment_expression"
            | "augmented_assignment_expression" => Some("assignment"),
            "call_expression" => Some("call"),
            "return_statement" => Some("return"),
            "function_declaration" | "function_expression" | "arrow_function"
            | "method_definition" => Some("function_def"),
            "class_declaration" => Some("class_def"),
            "expression_statement" => Some("expression"),
            _ => None,
        },
        "go" => match node_type {
            "if_statement" => Some("condition"),
            "for_statement" | "for_range" => Some("loop"),
            "assignment_statement" | "short_var_declaration" => Some("assignment"),
            "call_expression" => Some("call"),
            "return_statement" => Some("return"),
            "function_declaration" => Some("function_def"),
            "type_declaration" => Some("class_def"),
            _ => None,
        },
        _ => None,
    }
}

/// Visibility rules exactly like `_determine_visibility` (content-based).
fn determine_visibility(language: &str, content: &str) -> Option<String> {
    let lower = content.to_lowercase();
    match language {
        "python" => {
            if content.starts_with('_') && !content.starts_with("__") {
                Some("private".into())
            } else if content.starts_with("__") && !content.ends_with("__") {
                Some("private".into())
            } else {
                // dunder (both prefix and suffix) and everything else: public
                Some("public".into())
            }
        }
        "javascript" | "typescript" => {
            if lower.contains("private") {
                Some("private".into())
            } else if lower.contains("protected") {
                Some("protected".into())
            } else {
                Some("public".into())
            }
        }
        _ => None,
    }
}

fn statement_id(file_path: &str, start_line: i64, end_line: i64, node_type: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{file_path}:{start_line}:{end_line}:{node_type}"));
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{b:02x}")).collect::<String>()[..16].to_string()
}

fn language_for(language: &str) -> Option<tree_sitter::Language> {
    match language {
        "python" => Some(tree_sitter_python::LANGUAGE.into()),
        "javascript" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "typescript" | "tsx" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        _ => None,
    }
}

/// Walk one parsed tree, emitting statement records in exact Python
/// walk order (pre-order DFS; children inherit the nearest emitted
/// ancestor's node id as scope).
fn extract_statements_from_tree(
    language: &str,
    file_path: &str,
    source: &[u8],
    root: &tree_sitter::Tree,
) -> Vec<StatementRecord> {
    let mut records = Vec::new();
    let mut cursor = root.walk();

    fn walk(
        cursor: &mut tree_sitter::TreeCursor,
        language: &str,
        file_path: &str,
        lines: &[&str],
        parent_scope: &str,
        records: &mut Vec<StatementRecord>,
    ) {
        let node = cursor.node();
        let node_type = node.kind();

        let mut scope_for_children = parent_scope.to_string();
        if let Some(statement_type) = classify_statement(language, node_type) {
            let start_line = (node.start_position().row + 1) as i64;
            let end_line = (node.end_position().row + 1) as i64;
            let raw_line = lines
                .get(start_line as usize - 1)
                .map(|l| l.trim())
                .unwrap_or("");
            let content_preview = if raw_line.len() > 100 { &raw_line[..100] } else { raw_line };
            let node_id = statement_id(file_path, start_line, end_line, node_type);
            records.push(StatementRecord {
                visibility: determine_visibility(language, raw_line),
                node_id: node_id.clone(),
                node_type: node_type.to_string(),
                statement_type: statement_type.to_string(),
                start_line,
                end_line,
                content_preview: content_preview.to_string(),
                scope_id: parent_scope.to_string(),
            });
            scope_for_children = node_id;
        }

        if cursor.goto_first_child() {
            loop {
                walk(cursor, language, file_path, lines, &scope_for_children, records);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }
    }

    // split_once-free line split: Python's splitlines for '\n' sources.
    let text = String::from_utf8_lossy(source);
    let lines: Vec<&str> = text.split('\n').collect();
    walk(&mut cursor, language, file_path, &lines, "", &mut records);
    records
}

/// Extract statement records for a BATCH of files with the GIL released
/// and rayon parallelism inside — one FFI crossing for the whole batch
/// (the native-acceleration FFI contract).
///
/// Each file is `(language, path, source_bytes)`. Returns one
/// `list[StatementRecord]` per input file, in order. A file whose
/// language has no native grammar raises ValueError — callers fall back
/// to the Python walk for that file (reference implementation stays).
#[pyfunction]
#[pyo3(signature = (files))]
pub fn extract_statements_batch(
    py: Python<'_>,
    files: Vec<(String, String, Vec<u8>)>,
) -> PyResult<Vec<Vec<StatementRecord>>> {
    // All inputs are owned values: no Python API use inside detach.
    // PyErr is !Ungil, so per-file failures cross as Strings (the crate's
    // established idiom) and convert to exceptions after the GIL returns.
    let results: Vec<Result<Vec<StatementRecord>, String>> =
        py.detach(move || {
            files
                .into_par_iter()
                .map(|(language, path, source)| -> Result<Vec<StatementRecord>, String> {
                    let grammar = language_for(&language)
                        .ok_or_else(|| format!("no native grammar for language '{language}'"))?;
                    let mut parser = Parser::new();
                    parser
                        .set_language(&grammar)
                        .map_err(|e| format!("parser init failed for '{language}': {e}"))?;
                    let tree = parser
                        .parse(&source, None)
                        .ok_or_else(|| format!("parse failed for '{path}'"))?;
                    Ok(extract_statements_from_tree(&language, &path, &source, &tree))
                })
                .collect()
        });

    results
        .into_iter()
        .map(|r| r.map_err(PyValueError::new_err))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_matches_python_tables() {
        // Spot-checks of the exact tables from _classify_statement.
        assert_eq!(classify_statement("python", "elif_clause"), Some("condition"));
        assert_eq!(classify_statement("python", "async_function_definition"), Some("function_def"));
        assert_eq!(classify_statement("python", "match_statement"), Some("switch"));
        assert_eq!(classify_statement("python", "module"), None);
        assert_eq!(classify_statement("typescript", "arrow_function"), Some("function_def"));
        assert_eq!(classify_statement("typescript", "switch_default"), Some("case"));
        assert_eq!(classify_statement("go", "for_range"), Some("loop"));
        // Go has NO native grammar yet — unsupported languages classify to
        // None at walk time via language_for, but the table itself is pinned.
        assert_eq!(classify_statement("rust", "if_expression"), None);
    }

    #[test]
    fn visibility_rules_match_python() {
        assert_eq!(determine_visibility("python", "_private = 1"), Some("private".into()));
        assert_eq!(determine_visibility("python", "__init__(self)"), Some("private".into()));
        // The dunder check is on the whole stripped LINE: "__x__ = 1" ends
        // with "= 1" so Python classifies it private, not public.
        assert_eq!(determine_visibility("python", "__all__ = 1"), Some("private".into()));
        // A line that is ONLY the dunder name is public.
        assert_eq!(determine_visibility("python", "__init__"), Some("public".into()));
        assert_eq!(determine_visibility("typescript", "private x = 1"), Some("private".into()));
        assert_eq!(determine_visibility("go", "x := 1"), None);
    }

    #[test]
    fn statement_id_is_sha256_prefix_hex16() {
        // Python: hashlib.sha256(b"a.py:3:5:if_statement").hexdigest()[:16]
        let expected = {
            let mut h = Sha256::new();
            h.update(b"a.py:3:5:if_statement");
            h.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>()[..16].to_string()
        };
        assert_eq!(statement_id("a.py", 3, 5, "if_statement"), expected);
        assert_eq!(statement_id("a.py", 3, 5, "if_statement").len(), 16);
    }

    #[test]
    fn walk_emits_records_with_scope_cascade() {
        let source = b"def f():\n    if x:\n        return 1\n";
        let grammar = language_for("python").unwrap();
        let mut parser = Parser::new();
        parser.set_language(&grammar).unwrap();
        let tree = parser.parse(source, None).unwrap();
        let records =
            extract_statements_from_tree("python", "a.py", source, &tree);
        // function_def -> condition -> return, nested scopes cascade.
        let kinds: Vec<&str> = records.iter().map(|r| r.statement_type.as_str()).collect();
        assert_eq!(kinds, vec!["function_def", "condition", "return"]);
        assert_eq!(records[0].scope_id, "");
        assert_eq!(records[1].scope_id, records[0].node_id);
        assert_eq!(records[2].scope_id, records[1].node_id);
        assert_eq!(records[0].start_line, 1);
        assert_eq!(records[1].start_line, 2);
    }

    #[test]
    fn unsupported_language_is_an_error_not_a_guess() {
        let grammar = language_for("go");
        assert!(grammar.is_none());
    }
}
