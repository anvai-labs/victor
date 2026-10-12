"""Conservative relative ES-module bindings; no package execution or name guessing."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass

from .languages import TREE_SITTER_GRAMMAR
from .model import CodeSymbol, ParsedCode, ParseStatus
from . import treesitter_parser
from .treesitter_parser import _text


@dataclass(frozen=True)
class Binding:
    module: str | None
    name: str
    type_only: bool = False


class JavaScriptRepository:
    """Resolve explicit exports/imports with bounded cycle-safe re-export traversal."""

    def __init__(self, files: dict[str, ParsedCode]) -> None:
        self.files = files
        self.imports: dict[str, dict[str, list[Binding]]] = {}
        self.exports: dict[str, dict[str, list[Binding]]] = {}
        self.dependencies: dict[str, list[tuple[str, str | None]]] = {}
        for path, parsed in files.items():
            if (
                parsed.language not in {"javascript", "typescript", "tsx"}
                or parsed.status != ParseStatus.SUCCESS
            ):
                continue
            src = parsed.source_code.encode("utf-8")
            root = (
                treesitter_parser._get_parser(TREE_SITTER_GRAMMAR[parsed.language])
                .parse(src)
                .root_node
            )
            imports: dict[str, list[Binding]] = {}
            exports: dict[str, list[Binding]] = {}
            dependencies = []
            for node in root.named_children:
                if node.type not in {"import_statement", "export_statement"}:
                    continue
                source = node.child_by_field_name("source")
                module = None
                if source is not None:
                    # Relative literals only; escaped strings and package/tsconfig
                    # aliases remain unresolved until an explicit resolver exists.
                    literal = _text(source, src)[1:-1]
                    module = self.module(path, literal)
                    dependencies.append((_text(node, src), module))
                type_only = any(c.type == "type" for c in node.children)
                table = imports if node.type == "import_statement" else exports
                declaration = node.child_by_field_name("declaration")
                if declaration is not None:
                    name = declaration.child_by_field_name("name")
                    if name is not None:
                        local = _text(name, src)
                        public = (
                            "default" if any(c.type == "default" for c in node.children) else local
                        )
                        table.setdefault(public, []).append(Binding(None, local, type_only))
                if declaration is not None and declaration.type in {
                    "lexical_declaration",
                    "variable_declaration",
                }:
                    for decl in declaration.named_children:
                        name = decl.child_by_field_name("name")
                        value = decl.child_by_field_name("value")
                        if (
                            name is not None
                            and name.type == "identifier"
                            and value is not None
                            and value.type == "arrow_function"
                        ):
                            local = _text(name, src)
                            table.setdefault(local, []).append(Binding(None, local, type_only))
                stack = list(node.named_children)
                while stack:
                    child = stack.pop()
                    if child.type in {"import_specifier", "export_specifier"}:
                        name = child.child_by_field_name("name")
                        alias = child.child_by_field_name("alias")
                        if name is not None:
                            original = _text(name, src)
                            local = _text(alias, src) if alias is not None else original
                            # None with a source means unresolved module, never local.
                            target_module = module if source is None or module else ""
                            table.setdefault(local, []).append(
                                Binding(
                                    target_module,
                                    original,
                                    type_only or any(c.type == "type" for c in child.children),
                                )
                            )
                    elif child.type == "namespace_import":
                        name = child.named_children[-1]
                        table.setdefault(_text(name, src), []).append(
                            Binding(module or "", "*", type_only)
                        )
                    elif child.type == "import_clause":
                        for name in child.named_children:
                            if name.type == "identifier":
                                table.setdefault(_text(name, src), []).append(
                                    Binding(module or "", "default", type_only)
                                )
                        stack.extend(child.named_children)
                    elif child.type in {"named_imports", "export_clause"}:
                        stack.extend(child.named_children)
            self.imports[path] = imports
            self.exports[path] = exports
            self.dependencies[path] = dependencies

    def module(self, importer: str, literal: str) -> str | None:
        if not literal.startswith(".") or "\\" in literal:
            return None
        stem = posixpath.normpath(posixpath.join(posixpath.dirname(importer), literal))
        if stem.startswith("../"):
            return None
        variants = (
            [stem]
            if posixpath.splitext(stem)[1]
            else [stem + ext for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs")]
            + [stem + "/index" + ext for ext in (".ts", ".tsx", ".js")]
        )
        matches = [p for p in variants if p in self.files]
        return matches[0] if len(matches) == 1 else None

    def target(self, importer: str, text: str) -> list[CodeSymbol]:
        parts = text.split(".")
        bindings = self.imports.get(importer, {}).get(parts[0], [])
        if len(bindings) != 1:
            return []
        binding = bindings[0]
        if binding.type_only or not binding.module:
            return []
        if binding.name == "*" and len(parts) == 2:
            return self._export(binding.module, parts[1], set())
        if len(parts) == 1:
            return self._export(binding.module, binding.name, set())
        return []

    def _export(self, file: str, name: str, visited: set[tuple[str, str]]) -> list[CodeSymbol]:
        key = (file, name)
        if key in visited or len(visited) >= 64:
            return []
        visited.add(key)
        bindings = self.exports.get(file, {}).get(name, [])
        if len(bindings) != 1 or bindings[0].type_only:
            return []
        binding = bindings[0]
        if binding.module is not None:
            return self._export(binding.module, binding.name, visited) if binding.module else []
        return [
            s
            for s in self.files[file].symbols
            if not s.scope_chain
            and s.simple_name == binding.name
            and binding.name not in s.metadata.get("javascript_bound_names", [])
            and not s.metadata.get("javascript_conditional")
        ]
