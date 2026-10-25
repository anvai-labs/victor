//! Native plugin-shape statement extraction (ADR: native-acceleration-strategy,
//! Option B phase 1).
//!
//! Serves the victor-coding enhanced CCG builder's statement walk: the same
//! statement set and per-statement enrichment (definitions/uses/calls,
//! preview, visibility, symbol hints) the Python walk produces, minus the
//! ~37 ms/file interpreter floor. The ENGINE is generic — every
//! language-syntax decision arrives as data (the classification/identifier
//! tables and symbol hints are passed per call; victor-coding stays the
//! single owner of that knowledge). The Python walk remains the typed
//! reference implementation; the golden corpus at
//! `verticals/victor-coding/tests/codebase/goldens/` is the parity oracle,
//! read as pure data by the root-side differential test.
//!
//! Parity notes (all pinned by the golden differential):
//! - walk order: pre-order DFS, children in source order, descent pruned
//!   into `skip_descent` node types for BOTH statements and identifiers;
//! - node id: sha256("path:start:end:type")[:16] with 1-based lines;
//! - symbol hints key on (start_byte, end_byte, node_type) of the OWNER;
//! - definitions: only def/class statements keyword-filter their names;
//!   the field-target branch collects identifiers unfiltered;
//! - qualified call names join identifier pieces with "." keeping the
//!   LAST three pieces (Python `pieces[-3:]`);
//! - enrichment lists are sorted and capped at 25; preview at 160.

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use tree_sitter::{Node, Parser, Tree};

/// The language-syntax tables, passed as data from victor-coding.
#[derive(Clone, Debug)]
pub struct PluginTables {
    /// node type -> statement type (empty = walk does not emit).
    pub classify: HashMap<String, String>,
    /// Node types whose text is an identifier.
    pub identifier_types: HashSet<String>,
    /// Node types whose subtrees statements and identifiers never enter.
    pub skip_descent: HashSet<String>,
    /// Statement types that open a scope for their descendants.
    pub scope_types: HashSet<String>,
    /// Statement types that define names.
    pub definition_types: HashSet<String>,
    /// Keyword set for name filtering (compared lowercased).
    pub keywords: HashSet<String>,
    pub function_def: String,
    pub class_def: String,
    pub call_type: String,
    /// Field names for definition targets / declared names / parameters /
    /// call callees, in Python order (first match wins where it wins).
    pub def_fields: Vec<String>,
    pub name_fields: Vec<String>,
    pub param_fields: Vec<String>,
    pub call_fields: Vec<String>,
}

/// One extracted plugin-shape statement — the exact field set the Python
/// walk feeds GraphNode construction (which stays in Python).
#[pyclass(get_all)]
#[derive(Clone, Debug, PartialEq)]
pub struct PluginStatementRecord {
    /// sha256("path:start:end:type")[:16] hex.
    pub node_id: String,
    pub node_type: String,
    pub statement_type: String,
    /// "{statement_type}:{symbol}:{line}" or "{node_type}:{line}".
    pub name: String,
    pub start_line: i64,
    pub end_line: i64,
    pub visibility: Option<String>,
    /// Node id of the nearest enclosing scope statement ("" at top level).
    pub scope_id: String,
    pub symbol_name: Option<String>,
    pub symbol_type: Option<String>,
    /// Sorted, capped at 25 — the metadata view.
    pub definitions: Vec<String>,
    pub uses: Vec<String>,
    pub calls: Vec<String>,
    /// Whitespace-normalized source text, capped at 160 chars.
    pub content_preview: String,
}

impl PluginTables {
    fn statement_type_of(&self, node_type: &str) -> Option<&str> {
        self.classify.get(node_type).map(String::as_str)
    }

    fn is_keyword(&self, text: &str) -> bool {
        self.keywords.contains(text.to_lowercase().as_str())
    }
}

fn node_text<'s>(node: &Node, source: &'s [u8]) -> std::borrow::Cow<'s, str> {
    String::from_utf8_lossy(&source[node.byte_range()])
}

/// Python `str.strip()` semantics: Unicode whitespace trim.
fn trimmed(text: &str) -> &str {
    text.trim()
}

fn stable_id(path: &str, start_line: i64, end_line: i64, node_type: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{path}:{start_line}:{end_line}:{node_type}"));
    let digest = hasher.finalize();
    digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()[..16]
        .to_string()
}

/// Identifier set of a subtree, pruning skip-descent types exactly like
/// `_collect_identifiers` (the root prunes too).
fn collect_identifiers(node: Node, source: &[u8], tables: &PluginTables) -> HashSet<String> {
    let mut out = HashSet::new();
    collect_identifiers_into(node, source, tables, &mut out);
    out
}

/// Per-node (identifiers, calls) from ONE post-order pass — the single-pass
/// memo. The phase-1 port faithfully reproduced the plugin's per-statement
/// subtree re-walks (~410k node visits on the boundary corpus); computing
/// each node's aggregate exactly once cuts that to one visit per node.
/// Keyed by `Node::id()` (stable within a tree). Nodes inside skip
/// subtrees are never visited — a lookup miss falls back to the recursive
/// walk, so the memo is exact, never approximate.
type EnrichmentMemo = HashMap<usize, (HashSet<String>, HashSet<String>)>;

fn build_enrichment_memo(
    node: Node,
    source: &[u8],
    tables: &PluginTables,
    memo: &mut EnrichmentMemo,
) -> (HashSet<String>, HashSet<String>) {
    if tables.skip_descent.contains(node.kind()) {
        // Mirrors the recursive walk's root prune: a skip-typed root yields
        // empty sets (a field child CAN be a skip type).
        return (HashSet::new(), HashSet::new());
    }
    let mut identifiers = HashSet::new();
    let mut calls = HashSet::new();
    if tables.identifier_types.contains(node.kind()) {
        let owned = node_text(&node, source);
        let text = trimmed(&owned);
        if !text.is_empty() && !tables.is_keyword(text) {
            identifiers.insert(text.to_string());
        }
    }
    if tables.statement_type_of(node.kind()) == Some(tables.call_type.as_str()) {
        if let Some(name) = call_name(node, source, tables) {
            calls.insert(name);
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let (child_ids, child_calls) = build_enrichment_memo(child, source, tables, memo);
        identifiers.extend(child_ids);
        calls.extend(child_calls);
    }
    memo.insert(node.id(), (identifiers.clone(), calls.clone()));
    (identifiers, calls)
}

/// Exact memo lookups with recursive-walk fallback on miss. The borrow
/// variants avoid cloning a statement's full aggregate (function-level
/// statements carry the largest identifier sets); callers that need an
/// owned set (definitions unions) use the owned variants.
fn memoized_identifiers_ref<'a>(
    node: Node,
    source: &[u8],
    tables: &PluginTables,
    memo: &'a EnrichmentMemo,
) -> std::borrow::Cow<'a, HashSet<String>> {
    match memo.get(&node.id()) {
        Some((identifiers, _)) => std::borrow::Cow::Borrowed(identifiers),
        None => std::borrow::Cow::Owned(collect_identifiers(node, source, tables)),
    }
}

fn memoized_identifiers(
    node: Node,
    source: &[u8],
    tables: &PluginTables,
    memo: &EnrichmentMemo,
) -> HashSet<String> {
    match memo.get(&node.id()) {
        Some((identifiers, _)) => identifiers.clone(),
        None => collect_identifiers(node, source, tables),
    }
}

fn memoized_calls(
    node: Node,
    source: &[u8],
    tables: &PluginTables,
    memo: &EnrichmentMemo,
) -> HashSet<String> {
    match memo.get(&node.id()) {
        Some((_, calls)) => calls.clone(),
        None => extract_call_names(node, source, tables),
    }
}

fn collect_identifiers_into(
    node: Node,
    source: &[u8],
    tables: &PluginTables,
    out: &mut HashSet<String>,
) {
    if tables.skip_descent.contains(node.kind()) {
        return;
    }
    if tables.identifier_types.contains(node.kind()) {
        let owned = node_text(&node, source);
        let text = trimmed(&owned);
        if !text.is_empty() && !tables.is_keyword(text) {
            out.insert(text.to_string());
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_identifiers_into(child, source, tables, out);
    }
}

/// First identifier (truthy, non-keyword) in DFS order — `_first_identifier`.
fn first_identifier(node: Option<Node>, source: &[u8], tables: &PluginTables) -> Option<String> {
    let node = node?;
    // Note: _first_identifier does NOT prune skip types — every child is
    // scanned; only identifier-type membership gates it.
    if tables.identifier_types.contains(node.kind()) {
        let owned = node_text(&node, source);
        let text = trimmed(&owned);
        if !text.is_empty() && !tables.is_keyword(text) {
            return Some(text.to_string());
        }
        // Python returns the keyword text check differently: an empty or
        // keyword identifier falls through to the children scan.
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(name) = first_identifier(Some(child), source, tables) {
            return Some(name);
        }
    }
    None
}

/// `_qualified_identifier`: identifier nodes pass through; other nodes join
/// their direct identifier children with "." keeping the LAST three pieces.
fn qualified_identifier(
    node: Option<Node>,
    source: &[u8],
    tables: &PluginTables,
) -> Option<String> {
    let node = node?;
    if tables.identifier_types.contains(node.kind()) {
        let owned = node_text(&node, source);
        let text = trimmed(&owned);
        if tables.is_keyword(text) {
            return None;
        }
        return Some(text.to_string());
    }
    let mut pieces: Vec<String> = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if tables.identifier_types.contains(child.kind()) {
            let owned = node_text(&child, source);
            let text = trimmed(&owned);
            if !text.is_empty() && !tables.is_keyword(text) {
                pieces.push(text.to_string());
            }
        }
    }
    if pieces.is_empty() {
        return first_identifier(Some(node), source, tables);
    }
    let start = pieces.len().saturating_sub(3);
    Some(pieces[start..].join("."))
}

/// `_call_name`: first field hit wins, else the node itself.
fn call_name(node: Node, source: &[u8], tables: &PluginTables) -> Option<String> {
    for field in &tables.call_fields {
        let child = node.child_by_field_name(field);
        if let Some(name) = qualified_identifier(child, source, tables) {
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    let name = qualified_identifier(Some(node), source, tables)?;
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// `_extract_call_names`: call-classified nodes anywhere in the subtree
/// (skip types pruned) contribute their callee name.
fn extract_call_names(node: Node, source: &[u8], tables: &PluginTables) -> HashSet<String> {
    let mut out = HashSet::new();
    extract_calls_into(node, source, tables, &mut out);
    out
}

fn extract_calls_into(node: Node, source: &[u8], tables: &PluginTables, out: &mut HashSet<String>) {
    if tables.skip_descent.contains(node.kind()) {
        return;
    }
    if tables.statement_type_of(node.kind()) == Some(tables.call_type.as_str()) {
        if let Some(name) = call_name(node, source, tables) {
            out.insert(name);
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        extract_calls_into(child, source, tables, out);
    }
}

/// `_extract_declared_name`.
fn declared_name(
    node: Node,
    statement_type: &str,
    source: &[u8],
    tables: &PluginTables,
) -> Option<String> {
    for field in &tables.name_fields {
        let child = node.child_by_field_name(field);
        if let Some(name) = first_identifier(child, source, tables) {
            return Some(name);
        }
    }
    if statement_type == tables.function_def || statement_type == tables.class_def {
        return first_identifier(Some(node), source, tables);
    }
    None
}

/// `_extract_parameter_names`.
fn parameter_names(
    node: Node,
    source: &[u8],
    tables: &PluginTables,
    memo: &EnrichmentMemo,
) -> HashSet<String> {
    let mut out = HashSet::new();
    for field in &tables.param_fields {
        if let Some(child) = node.child_by_field_name(field) {
            out.extend(memoized_identifiers(child, source, tables, memo));
        }
    }
    out
}

/// `_extract_definitions` — only the def/class branch keyword-filters.
fn extract_definitions(
    node: Node,
    statement_type: &str,
    source: &[u8],
    tables: &PluginTables,
    memo: &EnrichmentMemo,
) -> HashSet<String> {
    if !tables.definition_types.contains(statement_type) {
        return HashSet::new();
    }
    if statement_type == tables.function_def || statement_type == tables.class_def {
        let mut names: HashSet<String> = HashSet::new();
        if let Some(declared) = declared_name(node, statement_type, source, tables) {
            names.insert(declared);
        }
        names.extend(parameter_names(node, source, tables, memo));
        return names
            .into_iter()
            .filter(|name| !tables.is_keyword(name))
            .collect();
    }
    let mut targets: Vec<Node> = Vec::new();
    for field in &tables.def_fields {
        if let Some(child) = node.child_by_field_name(field) {
            targets.push(child);
        }
    }
    if targets.is_empty() {
        let mut cursor = node.walk();
        let first = {
            let mut children = node.children(&mut cursor);
            children.next()
        };
        if let Some(first) = first {
            targets.push(first);
        }
    }
    let mut names = HashSet::new();
    for target in targets {
        names.extend(memoized_identifiers(target, source, tables, memo));
    }
    names
}

/// `_content_preview`: byte-slice text (lossy), fallback to the start line,
/// whitespace-normalized.
fn content_preview(node: Node, source: &[u8]) -> String {
    let mut text = node_text(&node, source).to_string();
    if text.is_empty() {
        let start_row = node.start_position().row;
        let mut line_start = 0usize;
        let mut row = 0usize;
        for (i, b) in source.iter().enumerate() {
            if row == start_row {
                break;
            }
            if *b == b'\n' {
                row += 1;
                line_start = i + 1;
            }
        }
        let line_end = source[line_start..]
            .iter()
            .position(|b| *b == b'\n')
            .map(|p| line_start + p)
            .unwrap_or(source.len());
        text = String::from_utf8_lossy(&source[line_start..line_end]).to_string();
    }
    let normalized: Vec<&str> = trimmed(&text).split_whitespace().collect();
    normalized.join(" ")
}

/// `_determine_visibility`.
fn determine_visibility(symbol_name: Option<&str>, preview: &str) -> Option<String> {
    let lowered = preview.to_lowercase();
    if lowered.contains("private") {
        return Some("private".to_string());
    }
    if lowered.contains("protected") {
        return Some("protected".to_string());
    }
    if lowered.contains("public") || lowered.starts_with("pub ") || lowered.contains(" pub ") {
        return Some("public".to_string());
    }
    if let Some(name) = symbol_name {
        if name.starts_with('_') && !name.starts_with("__") {
            return Some("private".to_string());
        }
    }
    None
}

/// Sorted, UNCAPPED — the Python StatementRecord carries full sets (DDG
/// edges depend on them); the 25-item cap is a metadata-render concern the
/// owning side applies when it builds GraphNode metadata.
fn sorted_full(set: &HashSet<String>) -> Vec<String> {
    let mut items: Vec<String> = set.iter().cloned().collect();
    items.sort();
    items
}

/// The plugin statement walk: pre-order DFS, emitting one record per
/// classified node with its enrichment, tracking scope by node id.
fn extract_plugin_statements_from_tree(
    path: &str,
    source: &[u8],
    tree: &Tree,
    tables: &PluginTables,
    hints: &HashMap<(i64, i64, String), (String, Option<String>)>,
) -> Vec<PluginStatementRecord> {
    let mut out = Vec::new();
    let mut memo: EnrichmentMemo = HashMap::new();
    build_enrichment_memo(tree.root_node(), source, tables, &mut memo);
    walk_plugin_statements(
        tree.root_node(),
        path,
        source,
        tables,
        hints,
        &memo,
        "",
        &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn walk_plugin_statements(
    node: Node,
    path: &str,
    source: &[u8],
    tables: &PluginTables,
    hints: &HashMap<(i64, i64, String), (String, Option<String>)>,
    memo: &EnrichmentMemo,
    parent_scope: &str,
    out: &mut Vec<PluginStatementRecord>,
) {
    let node_type = node.kind();
    let mut scope = parent_scope.to_string();
    if let Some(statement_type) = tables.statement_type_of(node_type) {
        let start_line = (node.start_position().row + 1) as i64;
        let end_line = (node.end_position().row + 1) as i64;
        let node_id = stable_id(path, start_line, end_line, node_type);
        let hint = hints.get(&(
            node.start_byte() as i64,
            node.end_byte() as i64,
            node_type.to_string(),
        ));
        let symbol_name = hint
            .map(|(name, _)| name.clone())
            .or_else(|| declared_name(node, statement_type, source, tables));
        let preview = content_preview(node, source);
        let definitions = extract_definitions(node, statement_type, source, tables, memo);
        let identifiers = memoized_identifiers_ref(node, source, tables, memo);
        let calls = memoized_calls(node, source, tables, memo);
        let uses: HashSet<String> = identifiers.difference(&definitions).cloned().collect();
        let visibility = determine_visibility(symbol_name.as_deref(), &preview);
        let name = match &symbol_name {
            Some(symbol) if !symbol.is_empty() => {
                format!("{statement_type}:{symbol}:{start_line}")
            }
            _ => format!("{node_type}:{start_line}"),
        };
        out.push(PluginStatementRecord {
            node_id: node_id.clone(),
            node_type: node_type.to_string(),
            statement_type: statement_type.to_string(),
            name,
            start_line,
            end_line,
            visibility,
            scope_id: parent_scope.to_string(),
            symbol_name: symbol_name.clone(),
            symbol_type: hint.and_then(|(_, st)| st.clone()),
            definitions: sorted_full(&definitions),
            uses: sorted_full(&uses),
            calls: sorted_full(&calls),
            // Python str[:160] slices CODEPOINTS — byte truncate would
            // come up short whenever multibyte text precedes the cap.
            content_preview: preview.chars().take(160).collect(),
        });
        if tables.scope_types.contains(statement_type) {
            scope = node_id;
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if tables.skip_descent.contains(child.kind()) {
            continue;
        }
        walk_plugin_statements(child, path, source, tables, hints, memo, &scope, out);
    }
}

fn parse_hints(
    raw: &[(i64, i64, String, String, Option<String>)],
) -> HashMap<(i64, i64, String), (String, Option<String>)> {
    raw.iter()
        .map(|(start, end, node_type, name, symbol_type)| {
            (
                (*start, *end, node_type.clone()),
                (name.clone(), symbol_type.clone()),
            )
        })
        .collect()
}

/// Extract plugin-shape statements for a batch of files under one set of
/// tables. `files` is (language, path, source-bytes); `hints_by_file`
/// mirrors `files` order with (start_byte, end_byte, node_type, name,
/// symbol_type) tuples from the Python-side hint pass.
#[pyfunction]
#[pyo3(signature = (files, classify, identifier_types, skip_descent, scope_types, definition_types, keywords, function_def, class_def, call_type, def_fields, name_fields, param_fields, call_fields, hints_by_file))]
// The table set IS the language-syntax contract; one explicit parameter
// per table keeps the FFI surface self-documenting (no nested dict
// decoding on the hot path).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn extract_plugin_statements_batch(
    py: Python<'_>,
    files: Vec<(String, String, Vec<u8>)>,
    classify: HashMap<String, String>,
    identifier_types: HashSet<String>,
    skip_descent: HashSet<String>,
    scope_types: HashSet<String>,
    definition_types: HashSet<String>,
    keywords: HashSet<String>,
    function_def: String,
    class_def: String,
    call_type: String,
    def_fields: Vec<String>,
    name_fields: Vec<String>,
    param_fields: Vec<String>,
    call_fields: Vec<String>,
    hints_by_file: Vec<Vec<(i64, i64, String, String, Option<String>)>>,
) -> PyResult<Vec<Vec<PluginStatementRecord>>> {
    let tables = PluginTables {
        classify,
        identifier_types,
        skip_descent,
        scope_types,
        definition_types,
        keywords,
        function_def,
        class_def,
        call_type,
        def_fields,
        name_fields,
        param_fields,
        call_fields,
    };
    // All inputs owned; per-file failures cross detach as Strings.
    let results: Vec<Result<Vec<PluginStatementRecord>, String>> =
        py.detach(move || {
            files
            .into_par_iter()
            .zip(hints_by_file)
            .map(|((language, path, source), hints)| -> Result<Vec<PluginStatementRecord>, String> {
                let grammar = super::ccg_extract::language_for(&language)
                    .ok_or_else(|| format!("no native grammar for language '{language}'"))?;
                let mut parser = Parser::new();
                parser
                    .set_language(&grammar)
                    .map_err(|e| format!("parser init failed for '{language}': {e}"))?;
                let tree = parser
                    .parse(&source, None)
                    .ok_or_else(|| format!("parse failed for '{path}'"))?;
                let hint_map = parse_hints(&hints);
                Ok(extract_plugin_statements_from_tree(
                    &path, &source, &tree, &tables, &hint_map,
                ))
            })
            .collect()
        });
    results
        .into_iter()
        .map(|r| r.map_err(PyValueError::new_err))
        .collect()
}

#[pymethods]
impl PluginStatementRecord {
    fn __repr__(&self) -> String {
        format!(
            "PluginStatementRecord(node_type={}, stype={}, lines={}..{})",
            self.node_type, self.statement_type, self.start_line, self.end_line
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables() -> PluginTables {
        let mut classify = HashMap::new();
        classify.insert(
            "function_definition".to_string(),
            "function_def".to_string(),
        );
        classify.insert("expression_statement".to_string(), "expression".to_string());
        classify.insert("assignment".to_string(), "assignment".to_string());
        classify.insert("call".to_string(), "call".to_string());
        classify.insert("return_statement".to_string(), "return".to_string());
        PluginTables {
            classify,
            identifier_types: ["identifier"].into_iter().map(String::from).collect(),
            skip_descent: ["comment", "string"]
                .into_iter()
                .map(String::from)
                .collect(),
            scope_types: ["function_def"].into_iter().map(String::from).collect(),
            definition_types: ["function_def", "assignment"]
                .into_iter()
                .map(String::from)
                .collect(),
            keywords: ["def", "return"].into_iter().map(String::from).collect(),
            function_def: "function_def".to_string(),
            class_def: "class_def".to_string(),
            call_type: "call".to_string(),
            def_fields: vec!["left".to_string()],
            name_fields: vec!["name".to_string()],
            param_fields: vec!["parameters".to_string()],
            call_fields: vec!["function".to_string()],
        }
    }

    fn parse(source: &[u8]) -> Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        parser.parse(source, None).unwrap()
    }

    #[test]
    fn scope_tracking_and_walk_order() {
        let source = b"def outer(a):\n    b = a\n    return helper(b)\n";
        let tree = parse(source);
        let records = extract_plugin_statements_from_tree(
            "/t/x.py",
            source,
            &tree,
            &tables(),
            &HashMap::new(),
        );
        // Nested statements emit at every classified level (the plugin's
        // richer shape): expression_statement > assignment > call.
        let stypes: Vec<&str> = records.iter().map(|r| r.statement_type.as_str()).collect();
        assert_eq!(
            stypes,
            vec!["function_def", "expression", "assignment", "return", "call"]
        );
        let root_id = &records[0].node_id;
        assert!(records
            .iter()
            .all(|r| r.scope_id == *root_id || r.scope_id.is_empty()));
        for record in &records[1..] {
            assert_eq!(record.scope_id, *root_id);
        }
        let assignment = records
            .iter()
            .find(|r| r.statement_type == "assignment")
            .expect("assignment statement");
        // Assignment defines `b`, uses `a`.
        assert_eq!(assignment.definitions, vec!["b".to_string()]);
        assert_eq!(assignment.uses, vec!["a".to_string()]);
        // The call node names its callee via the function field.
        let call = records
            .iter()
            .find(|r| r.statement_type == "call")
            .expect("call statement");
        assert_eq!(call.calls, vec!["helper".to_string()]);
    }

    #[test]
    fn skip_descent_excludes_strings_from_enrichment() {
        let source = b"def f():\n    x = \"identifier_in_string\"\n    return x\n";
        let tree = parse(source);
        let records = extract_plugin_statements_from_tree(
            "/t/y.py",
            source,
            &tree,
            &tables(),
            &HashMap::new(),
        );
        let assignment = records
            .iter()
            .find(|r| r.statement_type == "assignment")
            .expect("assignment statement");
        assert!(assignment.uses.is_empty(), "string content must not count");
        assert_eq!(assignment.definitions, vec!["x".to_string()]);
    }

    #[test]
    fn symbol_hints_key_on_owner_byte_range() {
        let source = b"def f():\n    return 1\n";
        let tree = parse(source);
        let root = tree.root_node();
        let def_node = root.named_child(0).expect("function definition present");
        let mut hints = HashMap::new();
        hints.insert(
            (
                def_node.start_byte() as i64,
                def_node.end_byte() as i64,
                "function_definition".to_string(),
            ),
            ("hinted_name".to_string(), Some("function".to_string())),
        );
        let records =
            extract_plugin_statements_from_tree("/t/z.py", source, &tree, &tables(), &hints);
        assert_eq!(records[0].symbol_name.as_deref(), Some("hinted_name"));
        assert_eq!(records[0].symbol_type.as_deref(), Some("function"));
        assert!(records[0].name.starts_with("function_def:hinted_name:"));
    }
}
