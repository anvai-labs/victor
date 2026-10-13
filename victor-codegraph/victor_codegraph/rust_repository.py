"""Conservative Rust use-path resolution within declared local Cargo packages.

No macro expansion, cfg evaluation, re-export traversal or receiver inference.
Those targets remain unresolved rather than becoming name-only dependency edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from .treesitter_parser import _get_parser, _rust_cfg_attributes
from .model import ParsedCode, CodeSymbolType


@dataclass(frozen=True)
class RustBinding:
    file: str
    path: tuple[str, ...]
    name: str


@dataclass(frozen=True)
class RustUse:
    path: tuple[str, ...]
    name: str
    conditional: bool


def use_paths(raw: str) -> list[RustUse]:
    """Expand grouped use declarations through the Rust grammar, preserving aliases."""
    source = raw.encode()
    tree = _get_parser("rust").parse(source)

    def text(node):
        return source[node.start_byte : node.end_byte].decode()

    def walk(node, prefix=()):
        children = node.named_children
        if node.type in {"source_file", "use_declaration", "use_list"}:
            return [item for child in children for item in walk(child, prefix)]
        if node.type == "scoped_use_list":
            path = node.child_by_field_name("path")
            head = tuple(text(path).split("::")) if path else ()
            return walk(children[-1], prefix + head)
        if node.type == "use_as_clause":
            parts = tuple(text(children[0]).split("::"))
            return [
                (prefix if parts == ("self",) and prefix else prefix + parts, text(children[-1]))
            ]
        if node.type == "use_wildcard":
            return (
                [(prefix + tuple(text(children[0]).split("::")), "*")]
                if children
                else [(prefix, "*")]
            )
        if node.type in {"identifier", "scoped_identifier", "self", "crate", "super"}:
            parts = tuple(text(node).split("::"))
            if parts == ("self",) and prefix:
                return [(prefix, prefix[-1])]
            return [(prefix + parts, parts[-1])]
        return []

    conditional = any(
        _rust_cfg_attributes(child, source)
        for child in tree.root_node.named_children
        if child.type == "use_declaration"
    )
    return [RustUse(path, name, conditional) for path, name in walk(tree.root_node)]


class RustRepository:
    def __init__(self, root: Path, files: dict[str, ParsedCode]):
        self.root = root
        self.files = files
        self._manifests: dict[Path, dict] = {}

    def _module_file(self, current: str, symbol, root: str) -> str | None:
        if symbol.metadata.get("rust_path_attribute") or symbol.metadata.get("rust_conditional"):
            return None
        path = Path(current)
        base = path.parent if current == root or path.name == "mod.rs" else path.with_suffix("")
        stem = base.joinpath(*symbol.scope_chain, symbol.simple_name)
        matches = [
            p
            for p in (stem.with_suffix(".rs").as_posix(), (stem / "mod.rs").as_posix())
            if p in self.files
        ]
        return matches[0] if len(matches) == 1 else None

    def _members(self, root: str) -> set[str]:
        members: set[str] = set()
        pending = [root]
        while pending:
            current = pending.pop()
            if current in members or current not in self.files:
                continue
            members.add(current)
            for symbol in self.files[current].symbols:
                if symbol.symbol_type == CodeSymbolType.MODULE and not symbol.metadata.get(
                    "rust_inline_module"
                ):
                    target = self._module_file(current, symbol, root)
                    if target:
                        pending.append(target)
        return members

    def _manifest(self, directory: Path) -> dict:
        if directory not in self._manifests:
            try:
                self._manifests[directory] = tomllib.loads((directory / "Cargo.toml").read_text())
            except (OSError, ValueError):
                self._manifests[directory] = {}
        return self._manifests[directory]

    def resolve(self, importer: str, parts: tuple[str, ...], name: str) -> RustBinding | None:
        source = self.root / importer
        package = source.parent
        while package != self.root and not (package / "Cargo.toml").is_file():
            package = package.parent
        manifest = self._manifest(package)
        if not manifest or not parts:
            return None
        library = (package / manifest.get("lib", {}).get("path", "src/lib.rs")).resolve()
        binaries = [
            (package / entry["path"]).resolve()
            for entry in manifest.get("bin", [])
            if "path" in entry
        ]
        if not library.is_relative_to(self.root) or any(
            not binary.is_relative_to(self.root) for binary in binaries
        ):
            return None
        if (
            source.name == "main.rs"
            or "bin" in source.relative_to(package).parts
            or source in binaries
        ):
            return None
        library_file = library.relative_to(self.root).as_posix()
        if importer not in self._members(library_file):
            return None
        binary_files = [*binaries, package / "src/main.rs"]
        binary_files.extend(
            self.root / file
            for file in self.files
            if (self.root / file).parent == package / "src/bin"
            or (
                (self.root / file).name == "main.rs"
                and (self.root / file).parent.parent == package / "src/bin"
            )
        )
        if any(
            importer in self._members(binary.relative_to(self.root).as_posix())
            for binary in binary_files
        ):
            return None
        head, *tail = parts
        if head in {"crate", "self", "super"}:
            if head != "crate":
                try:
                    relative = source.relative_to(library.parent)
                except ValueError:
                    return None
                scope = list(relative.with_suffix("").parts)
                if scope[-1] in {"lib", "main", "mod"}:
                    scope.pop()
                if head == "super":
                    if not scope:
                        return None
                    scope.pop()
                    while tail and tail[0] == "super":
                        if not scope:
                            return None
                        scope.pop()
                        tail.pop(0)
                tail = scope + tail
        else:
            dependencies = [
                value
                for key, value in manifest.get("dependencies", {}).items()
                if key.replace("-", "_") == head
            ]
            dependency = dependencies[0] if len(dependencies) == 1 else {}
            if not isinstance(dependency, dict) or "path" not in dependency:
                return None
            package = (package / dependency["path"]).resolve()
            if not package.is_relative_to(self.root):
                return None
            dependency_manifest = self._manifest(package)
            library = (
                package / dependency_manifest.get("lib", {}).get("path", "src/lib.rs")
            ).resolve()

        # Only declared modules belong to a crate; files alone are not evidence.
        if not library.is_relative_to(self.root):
            return None
        current = library.relative_to(self.root).as_posix()
        scope: tuple[str, ...] = ()
        for index, part in enumerate(tail):
            parsed = self.files.get(current)
            if parsed is None:
                return None
            modules = [
                symbol
                for symbol in parsed.symbols
                if symbol.symbol_type == CodeSymbolType.MODULE
                and tuple(symbol.scope_chain) == scope
                and symbol.simple_name == part
            ]
            if len(modules) > 1:
                return None
            if not modules:
                symbol_path = scope + tuple(tail[index:])
                if any(
                    (*s.scope_chain, s.simple_name) == symbol_path
                    and not s.metadata.get("rust_conditional")
                    for s in parsed.symbols
                ):
                    return RustBinding(current, symbol_path, name)
                return None
            if modules[0].metadata.get("rust_conditional"):
                return None
            if modules[0].metadata.get("rust_inline_module"):
                scope += (part,)
                continue
            if modules[0].metadata.get("rust_path_attribute"):
                return None
            target = self._module_file(
                current, modules[0], library.relative_to(self.root).as_posix()
            )
            if target is None:
                return None
            current, scope = target, ()
        return RustBinding(current, scope, name) if current in self.files else None
