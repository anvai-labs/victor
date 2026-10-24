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
            "variable_declaration"
            | "lexical_declaration"
            | "assignment_expression"
            | "augmented_assignment_expression" => Some("assignment"),
            "call_expression" => Some("call"),
            "return_statement" => Some("return"),
            "function_declaration"
            | "function_expression"
            | "arrow_function"
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
    digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()[..16]
        .to_string()
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
            let content_preview = if raw_line.len() > 100 {
                &raw_line[..100]
            } else {
                raw_line
            };
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
                walk(
                    cursor,
                    language,
                    file_path,
                    lines,
                    &scope_for_children,
                    records,
                );
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
    let results: Vec<Result<Vec<StatementRecord>, String>> = py.detach(move || {
        files
            .into_par_iter()
            .map(
                |(language, path, source)| -> Result<Vec<StatementRecord>, String> {
                    let grammar = language_for(&language)
                        .ok_or_else(|| format!("no native grammar for language '{language}'"))?;
                    let mut parser = Parser::new();
                    parser
                        .set_language(&grammar)
                        .map_err(|e| format!("parser init failed for '{language}': {e}"))?;
                    let tree = parser
                        .parse(&source, None)
                        .ok_or_else(|| format!("parse failed for '{path}'"))?;
                    Ok(extract_statements_from_tree(
                        &language, &path, &source, &tree,
                    ))
                },
            )
            .collect()
    });

    results
        .into_iter()
        .map(|r| r.map_err(PyValueError::new_err))
        .collect()
}

// ============================================================================
// Engine phase 2: the complete CCG (CFG + CDG + DDG edge builders).
//
// Reproduces _build_cfg_edges (sequential + structural), _build_cdg
// (reachability domination) and _build_ddg (def-use chains) exactly.
// Parity notes baked into the design:
//   - sequential edges: stable sort by line (walk order breaks ties);
//   - structural edges: LIFO stack order (children visited last-first,
//     matching Python's stack.pop()) and (src,dst,type) dedup against the
//     sequential edges;
//   - CDG edges: Python iterates a hash-randomized set, so consumers must
//     be order-insensitive (the store upserts by key) — Rust returns them
//     sorted for determinism;
//   - DDG: symbol table preserves first-definition insertion order;
//     duplicate use-sites are REPRODUCED (Python appends without dedup).
// ============================================================================

/// One CCG edge — the (src, dst, type) core plus the DDG variable when
/// applicable. GraphEdge construction (weight, metadata) stays in Python;
/// ``structural`` distinguishes sequential edges (metadata {"file": ...})
/// from structural ones (default metadata) exactly like the Python builders.
#[pyclass(get_all)]
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeRecord {
    pub src: String,
    pub dst: String,
    pub edge_type: String,
    /// DDG_DEF_USE only: the variable name riding in edge metadata.
    pub variable: Option<String>,
    /// CFG only: True when emitted by the structural pass (no file metadata).
    pub structural: bool,
}

#[pymethods]
impl EdgeRecord {
    fn __repr__(&self) -> String {
        format!(
            "EdgeRecord({} -{}-> {})",
            self.src, self.edge_type, self.dst
        )
    }
}

/// The complete per-file CCG from one native pass.
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct CcgFileResult {
    pub statements: Vec<StatementRecord>,
    pub cfg_edges: Vec<EdgeRecord>,
    pub cdg_edges: Vec<EdgeRecord>,
    pub ddg_edges: Vec<EdgeRecord>,
}

const IF_NODES: &[&str] = &["if_statement", "if_expression"];
const ELSE_NODES: &[&str] = &["elif_clause", "else_clause", "else_statement"];
const LOOP_NODES: &[&str] = &[
    "for_statement",
    "while_statement",
    "do_statement",
    "for_in_statement",
    "for_of_statement",
    "enhanced_for_statement",
    "for_range_loop",
    "loop_expression",
    "while_expression",
    "for_expression",
];
const TRY_NODES: &[&str] = &["try_statement", "try_expression"];
const CATCH_NODES: &[&str] = &["except_clause", "except_group_clause", "catch_clause"];
const FINALLY_NODES: &[&str] = &["finally_clause"];
const SWITCH_NODES: &[&str] = &[
    "switch_statement",
    "switch_expression",
    "match_statement",
    "match_expression",
];
const CASE_NODES: &[&str] = &[
    "case_clause",
    "switch_case",
    "case_statement",
    "match_arm",
    "switch_block_statement_group",
    "switch_rule",
];
const DEFAULT_CASE_NODES: &[&str] = &["switch_default", "default_case"];

const KEYWORDS: &[&str] = &[
    // Python
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
    // JavaScript/TypeScript
    "case",
    "catch",
    "const",
    "debugger",
    "default",
    "delete",
    "do",
    "enum",
    "export",
    "extends",
    "false",
    "function",
    "implements",
    "instanceof",
    "interface",
    "let",
    "new",
    "null",
    "of",
    "package",
    "private",
    "protected",
    "public",
    "super",
    "switch",
    "static",
    "this",
    "throw",
    "true",
    "typeof",
    "var",
    "void",
    // Go
    "chan",
    "defer",
    "fallthrough",
    "func",
    "go",
    "goto",
    "map",
    "range",
    "select",
    "struct",
    "type",
];

fn is_keyword(name: &str) -> bool {
    KEYWORDS.contains(&name)
}

fn is_definition_node(language: &str, node_type: &str) -> bool {
    match language {
        "python" => matches!(
            node_type,
            "assignment" | "named_expression" | "for_statement" | "with_clause" | "pattern"
        ),
        "javascript" | "typescript" => {
            matches!(node_type, "variable_declaration" | "assignment_expression")
        }
        "go" => matches!(
            node_type,
            "assignment_statement" | "short_var_declaration" | "for_statement" | "range_clause"
        ),
        _ => false,
    }
}

fn is_scope_boundary(node_type: &str) -> bool {
    matches!(
        node_type,
        "function_definition"
            | "class_definition"
            | "if_statement"
            | "for_statement"
            | "while_statement"
            | "try_statement"
            | "with_statement"
    )
}

/// Statement node id for a line: exact start match, else narrowest span
/// (Python `_statement_id_at_line`).
fn statement_id_at_line(records: &[StatementRecord], line: i64) -> Option<String> {
    let mut best_id: Option<String> = None;
    let mut best_span = i64::MAX;
    for record in records {
        if record.start_line == 0 {
            continue;
        }
        if record.start_line == line {
            return Some(record.node_id.clone());
        }
        let end = if record.end_line != 0 {
            record.end_line
        } else {
            record.start_line
        };
        if record.start_line <= line && line <= end {
            let span = end - record.start_line;
            if span < best_span {
                best_span = span;
                best_id = Some(record.node_id.clone());
            }
        }
    }
    best_id
}

/// First node id whose SPAN CONTAINS the line, in walk order (Python
/// `_find_node_at_line` — note: different resolution rule from above).
fn find_node_at_line(records: &[StatementRecord], line: i64) -> Option<String> {
    for record in records {
        if record.start_line == 0 {
            continue;
        }
        let end = if record.end_line != 0 {
            record.end_line
        } else {
            line
        };
        if record.start_line <= line && line <= end {
            return Some(record.node_id.clone());
        }
    }
    None
}

fn first_statement_in_span(records: &[StatementRecord], start: i64, end: i64) -> Option<String> {
    let mut best_id: Option<String> = None;
    let mut best_line = end + 1;
    for record in records {
        if record.start_line != 0
            && start <= record.start_line
            && record.start_line <= end
            && record.start_line < best_line
        {
            best_line = record.start_line;
            best_id = Some(record.node_id.clone());
        }
    }
    best_id
}

fn last_statement_in_span(records: &[StatementRecord], start: i64, end: i64) -> Option<String> {
    let mut best_id: Option<String> = None;
    let mut best_line = start;
    for record in records {
        if record.start_line != 0
            && start < record.start_line
            && record.start_line <= end
            && record.start_line > best_line
        {
            best_line = record.start_line;
            best_id = Some(record.node_id.clone());
        }
    }
    best_id
}

fn sequential_cfg_edge_type(statement_type: &str) -> &'static str {
    match statement_type {
        "condition" => "CFG_TRUE",
        "loop" => "CFG_LOOP_ENTRY",
        "switch" | "case" => "CFG_CASE",
        "return" => "CFG_RETURN",
        _ => "CFG_SUCCESSOR",
    }
}

/// Sequential adjacency edges: stable sort by line, connect i→i+1 when the
/// line gap is at most 30. NOTE: the class defines `_should_connect_cfg`
/// TWICE — the second definition (file-based, `abs(gap) <= 30`, same-line
/// pairs connected) is the one Python actually executes; the first
/// (10-line, strictly-increasing) is dead code shadowed by the later def.
fn build_sequential_cfg_edges(records: &[StatementRecord]) -> Vec<EdgeRecord> {
    let mut sorted: Vec<&StatementRecord> = records.iter().collect();
    sorted.sort_by_key(|r| r.start_line);
    let mut edges = Vec::new();
    for pair in sorted.windows(2) {
        let (current, next) = (pair[0], pair[1]);
        if current.start_line != 0
            && next.start_line != 0
            && (next.start_line - current.start_line).abs() <= 30
        {
            edges.push(EdgeRecord {
                src: current.node_id.clone(),
                dst: next.node_id.clone(),
                edge_type: sequential_cfg_edge_type(&current.statement_type).to_string(),
                variable: None,
                structural: false,
            });
        }
    }
    edges
}

fn node_line(node: &tree_sitter::Node) -> i64 {
    (node.start_position().row + 1) as i64
}

fn node_end_line(node: &tree_sitter::Node) -> i64 {
    (node.end_position().row + 1) as i64
}

/// Structural edges from the AST: false branches, loop back-edges,
/// exception flow, switch dispatch — in Python's exact LIFO visit order,
/// deduped against the sequential edges by (src, dst, type).
fn build_structural_cfg_edges(
    records: &[StatementRecord],
    root: &tree_sitter::Node,
) -> Vec<EdgeRecord> {
    let mut edges: Vec<EdgeRecord> = Vec::new();
    let mut seen: std::collections::HashSet<(String, String, String)> = Default::default();

    let mut stack: Vec<tree_sitter::Node> = vec![*root];
    while let Some(n) = stack.pop() {
        let kind = n.kind();
        let line = node_line(&n);
        let children: Vec<tree_sitter::Node> = n.children(&mut n.walk()).collect();

        let mut push_edge =
            |edges: &mut Vec<EdgeRecord>, src_line: i64, dst: tree_sitter::Node, ty: &str| {
                let dst_head = node_line(&dst);
                let dst_end = node_end_line(&dst);
                if let (Some(src), Some(dst_id)) = (
                    statement_id_at_line(records, src_line),
                    first_statement_in_span(records, dst_head, dst_end),
                ) {
                    if src != dst_id {
                        let key = (src.clone(), dst_id.clone(), ty.to_string());
                        if seen.insert(key) {
                            edges.push(EdgeRecord {
                                src,
                                dst: dst_id,
                                edge_type: ty.to_string(),
                                variable: None,
                                structural: true,
                            });
                        }
                    }
                }
            };

        if IF_NODES.contains(&kind) {
            let mut seen_alt_lines: Vec<i64> = Vec::new();
            if let Some(alt) = n.child_by_field_name("alternative") {
                let alt_line = node_line(&alt);
                if !seen_alt_lines.contains(&alt_line) {
                    seen_alt_lines.push(alt_line);
                    push_edge(&mut edges, line, alt, "CFG_FALSE");
                }
            }
            for ch in &children {
                if ELSE_NODES.contains(&ch.kind()) {
                    let alt_line = node_line(ch);
                    if !seen_alt_lines.contains(&alt_line) {
                        seen_alt_lines.push(alt_line);
                        push_edge(&mut edges, line, *ch, "CFG_FALSE");
                    }
                }
            }
        } else if LOOP_NODES.contains(&kind) {
            let last_line = node_end_line(&n);
            if last_line > line {
                if let (Some(src), Some(dst)) = (
                    last_statement_in_span(records, line, last_line),
                    statement_id_at_line(records, line),
                ) {
                    if src != dst {
                        let key = (src.clone(), dst.clone(), "CFG_LOOP_BACK".to_string());
                        if seen.insert(key) {
                            edges.push(EdgeRecord {
                                src,
                                dst,
                                edge_type: "CFG_LOOP_BACK".to_string(),
                                variable: None,
                                structural: true,
                            });
                        }
                    }
                }
            }
        } else if TRY_NODES.contains(&kind) {
            for ch in &children {
                if CATCH_NODES.contains(&ch.kind()) {
                    push_edge(&mut edges, line, *ch, "CFG_CATCH");
                } else if FINALLY_NODES.contains(&ch.kind()) {
                    push_edge(&mut edges, line, *ch, "CFG_FINALLY");
                }
            }
        } else if SWITCH_NODES.contains(&kind) {
            let body = n.child_by_field_name("body").unwrap_or(n);
            let body_children: Vec<tree_sitter::Node> = body.children(&mut body.walk()).collect();
            for ch in &body_children {
                if CASE_NODES.contains(&ch.kind()) {
                    push_edge(&mut edges, line, *ch, "CFG_CASE");
                } else if DEFAULT_CASE_NODES.contains(&ch.kind()) {
                    push_edge(&mut edges, line, *ch, "CFG_DEFAULT");
                }
            }
        }

        stack.extend(children);
    }
    edges
}

/// Control dependence: for each condition/loop/switch node, every node
/// reachable from it in the CFG (Python's reachability approximation).
/// Returned sorted — Python's set-iteration order is hash-randomized, so
/// consumers must be order-insensitive (they are: keyed upserts).
fn build_cdg_edges(records: &[StatementRecord], cfg_edges: &[EdgeRecord]) -> Vec<EdgeRecord> {
    use std::collections::HashMap;
    if records.is_empty() || cfg_edges.is_empty() {
        return Vec::new();
    }
    let mut successors: HashMap<&str, Vec<&str>> = HashMap::new();
    for record in records {
        successors.entry(record.node_id.as_str()).or_default();
    }
    for edge in cfg_edges {
        if let Some(list) = successors.get_mut(edge.src.as_str()) {
            list.push(edge.dst.as_str());
        }
    }

    let mut edges: Vec<EdgeRecord> = Vec::new();
    for condition in records
        .iter()
        .filter(|r| matches!(r.statement_type.as_str(), "condition" | "loop" | "switch"))
    {
        let edge_kind = if condition.statement_type == "loop" {
            "CDG_LOOP"
        } else {
            "CDG"
        };
        // Reachability from the condition node.
        let mut visited: std::collections::HashSet<&str> = Default::default();
        let mut to_visit: Vec<&str> = vec![condition.node_id.as_str()];
        while let Some(current) = to_visit.pop() {
            if !visited.insert(current) {
                continue;
            }
            if current != condition.node_id.as_str() {
                edges.push(EdgeRecord {
                    src: condition.node_id.clone(),
                    dst: current.to_string(),
                    edge_type: edge_kind.to_string(),
                    variable: None,
                    structural: false,
                });
            }
            if let Some(succs) = successors.get(current) {
                for succ in succs {
                    if !visited.contains(succ) {
                        to_visit.push(succ);
                    }
                }
            }
        }
    }
    edges.sort_by(|a, b| (&a.src, &a.dst, &a.edge_type).cmp(&(&b.src, &b.dst, &b.edge_type)));
    edges
}

struct VariableInfo {
    name: String,
    defining_node: Option<String>,
    use_sites: Vec<String>,
}

/// Data dependence: def-use chains from the AST walk. Reproduces the
/// Python semantics exactly — including the per-ancestor redundant use
/// extraction (duplicates preserved) and the first-span-containing node
/// resolution for definitions/uses.
fn build_ddg_edges(
    language: &str,
    records: &[StatementRecord],
    root: &tree_sitter::Node,
    file_path: &str,
    source: &[u8],
) -> Vec<EdgeRecord> {
    use std::collections::HashMap;

    // Ordered symbol table: first-definition insertion order matters for
    // edge emission order.
    let mut order: Vec<String> = Vec::new();
    let mut symbol_table: HashMap<String, VariableInfo> = HashMap::new();
    let mut scope_stack: Vec<String> = vec![String::new()];

    fn identifier_text<'a>(node: &tree_sitter::Node, source: &'a [u8]) -> Option<&'a str> {
        node.utf8_text(source).ok()
    }

    fn collect_identifiers<'node>(
        node: &tree_sitter::Node<'node>,
        source: &[u8],
        out: &mut Vec<String>,
    ) {
        if node.kind() == "identifier" {
            if let Some(text) = identifier_text(node, source) {
                if !is_keyword(text) {
                    out.push(text.to_string());
                }
            }
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            collect_identifiers(&child, source, out);
        }
    }

    fn walk_ddg<'node>(
        node: &tree_sitter::Node<'node>,
        language: &str,
        records: &[StatementRecord],
        file_path: &str,
        source: &[u8],
        order: &mut Vec<String>,
        symbol_table: &mut HashMap<String, VariableInfo>,
        scope_stack: &mut Vec<String>,
    ) {
        let kind = node.kind();

        // Definitions: first DIRECT identifier child.
        if is_definition_node(language, kind) {
            let mut var_name: Option<&str> = None;
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "identifier" {
                    if let Some(text) = identifier_text(&child, source) {
                        var_name = Some(text);
                    }
                    break;
                }
            }
            if let Some(name) = var_name {
                let current_scope = scope_stack.last().cloned().unwrap_or_default();
                let node_id = find_node_at_line(records, node_line(node));
                let key = format!("{current_scope}:{name}");
                // Python assigns a FRESH VariableInfo on every (re)definition
                // sight — use_sites reset with it; emit order = first sight.
                if !symbol_table.contains_key(&key) {
                    order.push(key.clone());
                }
                symbol_table.insert(
                    key,
                    VariableInfo {
                        name: name.to_string(),
                        defining_node: node_id,
                        use_sites: Vec::new(),
                    },
                );
            }
        }

        // Uses: identifiers in this node's SUBTREE (redundant per ancestor,
        // duplicates preserved — Python semantics).
        {
            let mut uses: Vec<String> = Vec::new();
            collect_identifiers(node, source, &mut uses);
            if !uses.is_empty() {
                let current_scope = scope_stack.last().cloned().unwrap_or_default();
                for name in uses {
                    let key = format!("{current_scope}:{name}");
                    if let Some(info) = symbol_table.get_mut(&key) {
                        let use_node_id = find_node_at_line(records, node_line(node));
                        if let Some(use_id) = use_node_id {
                            if Some(&use_id) != info.defining_node.as_ref() {
                                info.use_sites.push(use_id);
                            }
                        }
                    }
                }
            }
        }

        // Scope boundaries push/pop around child recursion.
        let pushed = if is_scope_boundary(kind) {
            let scope_id = statement_id(file_path, node_line(node), node_end_line(node), kind);
            scope_stack.push(scope_id);
            true
        } else {
            false
        };

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            walk_ddg(
                &child,
                language,
                records,
                file_path,
                source,
                order,
                symbol_table,
                scope_stack,
            );
        }

        if pushed {
            scope_stack.pop();
        }
    }

    // THE ACTUAL PYTHON CONTROL FLOW (reproduced exactly — this is what the
    // differential tests pin): `_build_ddg` defines a `walk` visitor that
    // ITSELF recurses children, then drives it through `_walk_ast` — which
    // ALSO recurses children. Every subtree is therefore visited once per
    // ancestor path (63k visits on a 750-line file): massively redundant,
    // and exactly the Python-side traversal cost this native path replaces.
    // The shared scope_stack interleaves pushes/pops across BOTH
    // recursions, reproducing the re-registrations and re-appends.
    fn walk_ast<'node>(
        node: &tree_sitter::Node<'node>,
        visit: &mut dyn FnMut(&tree_sitter::Node<'node>),
    ) {
        visit(node);
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            walk_ast(&child, visit);
        }
    }

    walk_ast(root, &mut |node| {
        walk_ddg(
            node,
            language,
            records,
            file_path,
            source,
            &mut order,
            &mut symbol_table,
            &mut scope_stack,
        )
    });

    let mut edges = Vec::new();
    for key in &order {
        if let Some(info) = symbol_table.get(key) {
            if let Some(defining) = &info.defining_node {
                for site in &info.use_sites {
                    edges.push(EdgeRecord {
                        src: defining.clone(),
                        dst: site.clone(),
                        edge_type: "DDG_DEF_USE".to_string(),
                        variable: Some(info.name.clone()),
                        structural: false,
                    });
                }
            }
        }
    }
    edges
}

/// The complete CCG for one parsed file: statements + CFG + CDG + DDG.
fn build_ccg_for_tree(
    language: &str,
    file_path: &str,
    source: &[u8],
    tree: &tree_sitter::Tree,
) -> CcgFileResult {
    let statements = extract_statements_from_tree(language, file_path, source, tree);
    let mut cfg_edges = build_sequential_cfg_edges(&statements);
    {
        let structural = build_structural_cfg_edges(&statements, &tree.root_node());
        let mut seen: std::collections::HashSet<(String, String, String)> = cfg_edges
            .iter()
            .map(|e| (e.src.clone(), e.dst.clone(), e.edge_type.clone()))
            .collect();
        for edge in structural {
            let key = (edge.src.clone(), edge.dst.clone(), edge.edge_type.clone());
            if seen.insert(key) {
                cfg_edges.push(edge);
            }
        }
    }
    let cdg_edges = build_cdg_edges(&statements, &cfg_edges);
    let ddg_edges = build_ddg_edges(language, &statements, &tree.root_node(), file_path, source);
    CcgFileResult {
        statements,
        cfg_edges,
        cdg_edges,
        ddg_edges,
    }
}

/// Build the complete CCG (statements + CFG + CDG + DDG edges) for a
/// batch of files — engine phase 2. Same FFI contract as
/// `extract_statements_batch`; same ValueError/fallback behavior.
#[pyfunction]
#[pyo3(signature = (files))]
pub fn build_ccg_batch(
    py: Python<'_>,
    files: Vec<(String, String, Vec<u8>)>,
) -> PyResult<Vec<CcgFileResult>> {
    let results: Vec<Result<CcgFileResult, String>> = py.detach(move || {
        files
            .into_par_iter()
            .map(
                |(language, path, source)| -> Result<CcgFileResult, String> {
                    let grammar = language_for(&language)
                        .ok_or_else(|| format!("no native grammar for language '{language}'"))?;
                    let mut parser = Parser::new();
                    parser
                        .set_language(&grammar)
                        .map_err(|e| format!("parser init failed for '{language}': {e}"))?;
                    let tree = parser
                        .parse(&source, None)
                        .ok_or_else(|| format!("parse failed for '{path}'"))?;
                    Ok(build_ccg_for_tree(&language, &path, &source, &tree))
                },
            )
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
        assert_eq!(
            classify_statement("python", "elif_clause"),
            Some("condition")
        );
        assert_eq!(
            classify_statement("python", "async_function_definition"),
            Some("function_def")
        );
        assert_eq!(
            classify_statement("python", "match_statement"),
            Some("switch")
        );
        assert_eq!(classify_statement("python", "module"), None);
        assert_eq!(
            classify_statement("typescript", "arrow_function"),
            Some("function_def")
        );
        assert_eq!(
            classify_statement("typescript", "switch_default"),
            Some("case")
        );
        assert_eq!(classify_statement("go", "for_range"), Some("loop"));
        // Go has NO native grammar yet — unsupported languages classify to
        // None at walk time via language_for, but the table itself is pinned.
        assert_eq!(classify_statement("rust", "if_expression"), None);
    }

    #[test]
    fn visibility_rules_match_python() {
        assert_eq!(
            determine_visibility("python", "_private = 1"),
            Some("private".into())
        );
        assert_eq!(
            determine_visibility("python", "__init__(self)"),
            Some("private".into())
        );
        // The dunder check is on the whole stripped LINE: "__x__ = 1" ends
        // with "= 1" so Python classifies it private, not public.
        assert_eq!(
            determine_visibility("python", "__all__ = 1"),
            Some("private".into())
        );
        // A line that is ONLY the dunder name is public.
        assert_eq!(
            determine_visibility("python", "__init__"),
            Some("public".into())
        );
        assert_eq!(
            determine_visibility("typescript", "private x = 1"),
            Some("private".into())
        );
        assert_eq!(determine_visibility("go", "x := 1"), None);
    }

    #[test]
    fn statement_id_is_sha256_prefix_hex16() {
        // Python: hashlib.sha256(b"a.py:3:5:if_statement").hexdigest()[:16]
        let expected = {
            let mut h = Sha256::new();
            h.update(b"a.py:3:5:if_statement");
            h.finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()[..16]
                .to_string()
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
        let records = extract_statements_from_tree("python", "a.py", source, &tree);
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
