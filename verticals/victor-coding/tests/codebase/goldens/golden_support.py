"""Shared helpers for the plugin-CCG golden corpus (generator + test)."""

from __future__ import annotations


def statement_signature(graph_node, scope_index: dict) -> dict:
    """Path-normalized statement record: everything the native port must
    reproduce, with storage identity (node_id/scope_id) index-normalized."""
    metadata = graph_node.metadata or {}
    return {
        "statement_type": graph_node.statement_type,
        "ast_kind": graph_node.ast_kind,
        "name": graph_node.name,
        "line": graph_node.line,
        "end_line": graph_node.end_line,
        "visibility": graph_node.visibility,
        "scope": scope_index.get(graph_node.scope_id, -1),
        "symbol_name": metadata.get("symbol_name"),
        "symbol_type": metadata.get("symbol_type"),
        "definitions": metadata.get("definitions", []),
        "uses": metadata.get("uses", []),
        "calls": metadata.get("calls", []),
        "content_preview": metadata.get("content_preview"),
    }


def scope_index_of(nodes: list) -> dict:
    """node_id -> walk-order index (path-independent scope normalization)."""
    return {node.node_id: i for i, node in enumerate(nodes)}
