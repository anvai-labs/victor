"""Static JNI declaration/export evidence, separate from body similarity.

JNI naming follows the JNI specification's UTF-16 escaping and parameter
descriptors. Dynamic RegisterNatives tables and runtime library loading are
not resolved here. Ambiguous export names are reported, never guessed.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any, Sequence


@dataclass(frozen=True)
class BindingFact:
    file: str
    symbol: str
    line: int
    language: str
    kind: str
    binding_name: str
    argument_descriptor: str | None = None


def jni_escape(value: str) -> str:
    """Escape UTF-16 code units according to JNI's native name mapping."""
    encoded = value.encode("utf-16-be")
    escaped = []
    for i in range(0, len(encoded), 2):
        unit = int.from_bytes(encoded[i : i + 2], "big")
        char = chr(unit)
        if "a" <= char <= "z" or "A" <= char <= "Z" or "0" <= char <= "9":
            escaped.append(char)
        else:
            escaped.append({"/": "_", "_": "_1", ";": "_2", "[": "_3"}.get(char, f"_0{unit:04x}"))
    return "".join(escaped)


def collect_binding_facts(
    path: str,
    data: bytes,
    root: Any,
    language: str,
    occurrences: Mapping[tuple[int, int], tuple[int, str]] | None = None,
) -> list[BindingFact]:
    """Collect native Java methods and conventional JNI C/C++ definitions."""
    if language not in {"java", "c", "cpp"}:
        return []
    from victor.core.graph_rag.language_parsers import jni_exports

    export_names = jni_exports(data) if language in {"c", "cpp"} else set()
    starts = [0, *(i + 1 for i, value in enumerate(data) if value == 10)]

    def text(node: Any) -> str:
        return data[node.start_byte : node.end_byte].decode("utf-8")

    def resolved_type(node: Any) -> str | None:
        """SCIP Java type descriptors retain binary package/nested-class identity."""
        targets = set()
        pending = [node]
        while pending and occurrences is not None:
            part = pending.pop()
            row = bisect_right(starts, part.start_byte) - 1
            hit = occurrences.get((row + 1, part.start_byte - starts[row]))
            if hit is not None and hit[0] == part.end_byte - part.start_byte:
                pieces = hit[1].split(" ", 4)
                if (
                    len(pieces) == 5
                    and pieces[0] == "scip-java"
                    and pieces[4].endswith("#")
                    and re.fullmatch(r"[\w/$#]+", pieces[4])
                ):
                    targets.add(pieces[4].rstrip("#").replace("#", "$"))
            pending.extend(part.named_children)
        return next(iter(targets)) if len(targets) == 1 else None

    package = ""
    imports: dict[str, str] = {}
    for child in root.named_children:
        if child.type == "package_declaration":
            package = text(child).removeprefix("package").rstrip(";").strip().replace(".", "/")
        elif child.type == "import_declaration":
            value = text(child).removeprefix("import").rstrip(";").strip()
            if not value.startswith("static ") and not value.endswith(".*"):
                imports[value.rsplit(".", 1)[-1]] = value.replace(".", "/")

    def descriptor(parameters: Any) -> str | None:
        result = []
        for parameter in parameters.named_children:
            if parameter.type != "formal_parameter":
                return None
            type_node = parameter.child_by_field_name("type")
            if type_node is None:
                return None
            value = text(type_node).replace(" ", "")
            dimensions = value.count("[]")
            extra = parameter.child_by_field_name("dimensions")
            if extra is not None:
                dimensions += text(extra).count("[]")
            value = value.replace("[]", "")
            primitive = {
                "boolean": "Z",
                "byte": "B",
                "char": "C",
                "short": "S",
                "int": "I",
                "long": "J",
                "float": "F",
                "double": "D",
            }.get(value)
            if primitive is not None:
                result.append("[" * dimensions + primitive)
                continue
            target = resolved_type(type_node)
            if target is None:
                # These explicit platform names are unambiguous. Other binary
                # reference descriptors require compiler evidence; dots alone
                # cannot distinguish packages from nested classes.
                qualified = imports.get(value, value.replace(".", "/"))
                if qualified in {
                    "java/lang/String",
                    "java/lang/Object",
                    "java/lang/Class",
                    "java/lang/Throwable",
                }:
                    target = qualified
            if target is None:
                return None
            result.append("[" * dimensions + "L" + target + ";")
        return "".join(result)

    facts = []

    def visit(node: Any, owners: tuple[str, ...] = ()) -> None:
        if node.type in {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        }:
            name = node.child_by_field_name("name")
            if name is not None:
                owners += (text(name),)
        if language == "java" and node.type == "method_declaration" and owners:
            modifiers = next((c for c in node.named_children if c.type == "modifiers"), None)
            name = node.child_by_field_name("name")
            parameters = node.child_by_field_name("parameters")
            if (
                modifiers is not None
                and any(child.type == "native" for child in modifiers.children)
                and name is not None
                and parameters is not None
                and node.child_by_field_name("body") is None
            ):
                owner = ((package + "/") if package else "") + "$".join(owners)
                symbol = text(name)
                facts.append(
                    BindingFact(
                        path,
                        symbol,
                        bisect_right(starts, name.start_byte),
                        language,
                        "jni_declaration",
                        "Java_" + jni_escape(owner) + "_" + jni_escape(symbol),
                        descriptor(parameters),
                    )
                )
        elif language in {"c", "cpp"} and node.type == "function_definition":
            declarator = node.child_by_field_name("declarator")
            while declarator is not None and declarator.type != "identifier":
                declarator = declarator.child_by_field_name("declarator")
            if declarator is not None:
                name = text(declarator)
                if (name, declarator.start_byte) in export_names:
                    facts.append(
                        BindingFact(
                            path,
                            name,
                            bisect_right(starts, declarator.start_byte),
                            language,
                            "jni_export",
                            name,
                        )
                    )
        for child in node.named_children:
            visit(child, owners)

    visit(root)
    return facts


def jni_relationships(facts: Sequence[BindingFact]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Link unique static exports; overloaded declarations require descriptors."""
    exports: dict[str, list[BindingFact]] = defaultdict(list)
    declarations: dict[str, list[BindingFact]] = defaultdict(list)
    for fact in facts:
        (exports if fact.kind == "jni_export" else declarations)[fact.binding_name].append(fact)
    links = []
    stats = {
        "ffi_jni_declarations": sum(map(len, declarations.values())),
        "ffi_jni_exports": sum(map(len, exports.values())),
        "ffi_ambiguous_bindings": 0,
        "ffi_unresolved_bindings": 0,
    }
    for short, methods in declarations.items():
        if len(methods) > 1 and short in exports:
            # The VM checks short names first. An overloaded short export
            # prevents attributing any long export to a specific runtime call.
            stats["ffi_ambiguous_bindings"] += len(methods)
            continue
        for method in methods:
            matches = exports.get(short, []) if len(methods) == 1 else []
            if not matches and method.argument_descriptor is not None:
                matches = exports.get(short + "__" + jni_escape(method.argument_descriptor), [])
            if len(matches) != 1:
                stats[
                    (
                        "ffi_ambiguous_bindings"
                        if len(matches) > 1 or (len(methods) > 1 and short in exports)
                        else "ffi_unresolved_bindings"
                    )
                ] += 1
                continue
            target = matches[0]
            links.append(
                {
                    "relationship": "declares_native_binding",
                    "bridge": "JNI",
                    "evidence": "jni_declaration_export",
                    "confidence": "static_binding_candidate",
                    "equivalence_verified": False,
                    "runtime_verified": False,
                    "abi_verified": False,
                    "source": {
                        "file": method.file,
                        "symbol": method.symbol,
                        "line": method.line,
                        "language": method.language,
                        "has_body": False,
                    },
                    "target": {
                        "file": target.file,
                        "symbol": target.symbol,
                        "line": target.line,
                        "language": target.language,
                    },
                }
            )
    return links, stats
