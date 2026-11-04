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

from victor.core.graph_rag.jni_contracts import (
    JavaSignature,
    NativeSignature,
    compare_jni_declarations,
)


@dataclass(frozen=True)
class BindingFact:
    file: str
    symbol: str
    line: int
    language: str
    kind: str
    binding_name: str
    argument_descriptor: str | None = None
    java_signature: JavaSignature | None = None
    native_signature: NativeSignature | None = None


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
    for child in root.named_children:
        if child.type == "package_declaration":
            package = text(child).removeprefix("package").rstrip(";").strip().replace(".", "/")

    def type_descriptor(type_node: Any, extra: Any = None) -> str | None:
        if type_node is None:
            return None
        value = text(type_node).replace(" ", "")
        dimensions = value.count("[]")
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
            "void": "V",
        }.get(value)
        if primitive is not None:
            return "[" * dimensions + primitive
        target = resolved_type(type_node)
        # Imports and even qualified-looking names can be shadowed by member
        # types or type parameters. Require fresh compiler identity for every
        # reference descriptor rather than guessing Java name resolution.
        if target is None:
            return None
        return "[" * dimensions + "L" + target + ";"

    def java_signature(node: Any, parameters: Any, modifiers: Any) -> JavaSignature:
        descriptors = tuple(
            (
                type_descriptor(p.child_by_field_name("type"), p.child_by_field_name("dimensions"))
                if p.type == "formal_parameter"
                else None
            )
            for p in parameters.named_children
            if p.type not in {"receiver_parameter", "line_comment", "block_comment"}
        )
        return JavaSignature(
            descriptors,
            type_descriptor(
                node.child_by_field_name("type"), node.child_by_field_name("dimensions")
            ),
            any(c.type == "static" for c in modifiers.children),
            # Annotation effects (including nonstandard native conventions) are
            # not resolved. Even harmless annotations conservatively abstain.
            not any(
                c.type in {"annotation", "marker_annotation"} for c in modifiers.named_children
            ),
        )

    def native_signature(node: Any) -> NativeSignature:
        def spelling(
            type_node: Any, declarator: Any, *, function_result: bool = False
        ) -> str | None:
            if type_node is None or type_node.type not in {"type_identifier", "primitive_type"}:
                return None
            value = text(type_node)
            while declarator is not None and declarator.type == "pointer_declarator":
                if any(c.type == "type_qualifier" for c in declarator.named_children):
                    return None
                value += "*"
                declarator = declarator.child_by_field_name("declarator")
            allowed = {"identifier", "function_declarator"} if function_result else {"identifier"}
            if declarator is not None and declarator.type not in allowed:
                return None
            return value

        outer = node.child_by_field_name("declarator")
        function = outer
        while function is not None and function.type == "pointer_declarator":
            function = function.child_by_field_name("declarator")
        parameters = function.child_by_field_name("parameters") if function is not None else None
        result = spelling(node.child_by_field_name("type"), outer, function_result=True)
        if parameters is None or any(c.type == "..." for c in parameters.children):
            return NativeSignature(None, result)
        arguments = []
        for parameter in parameters.named_children:
            if parameter.type in {"comment", "line_comment", "block_comment"}:
                continue
            if parameter.type != "parameter_declaration":
                return NativeSignature(None, result)
            if any(c.type == "type_qualifier" for c in parameter.named_children):
                arguments.append(None)
            else:
                arguments.append(
                    spelling(
                        parameter.child_by_field_name("type"),
                        parameter.child_by_field_name("declarator"),
                    )
                )
        if arguments == ["void"]:
            arguments = []
        return NativeSignature(tuple(arguments), result)

    facts = []

    def visit(node: Any, owners: tuple[str, ...] = (), owner_known: bool = True) -> None:
        if node.type in {"object_creation_expression", "enum_constant"}:
            owner_known = False
        if node.type in {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        }:
            # Local/anonymous binary names contain compiler-assigned components.
            # Lexical nesting alone cannot identify their JNI export owner.
            if node.parent is not None and node.parent.type not in {
                "program",
                "class_body",
                "interface_body",
                "enum_body",
                "enum_body_declarations",
            }:
                owner_known = False
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
                signature = java_signature(node, parameters, modifiers)
                facts.append(
                    BindingFact(
                        path,
                        symbol,
                        bisect_right(starts, name.start_byte),
                        language,
                        "jni_declaration" if owner_known else "jni_unresolved_declaration",
                        (
                            "Java_" + jni_escape(owner) + "_" + jni_escape(symbol)
                            if owner_known
                            else ""
                        ),
                        (
                            "".join(p or "" for p in signature.parameters)
                            if None not in signature.parameters
                            else None
                        ),
                        java_signature=signature,
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
                            native_signature=native_signature(node),
                        )
                    )
        for child in node.named_children:
            visit(child, owners, owner_known)

    visit(root)
    return facts


def jni_relationships(facts: Sequence[BindingFact]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Link unique static exports; overloaded declarations require descriptors."""
    exports: dict[str, list[BindingFact]] = defaultdict(list)
    declarations: dict[str, list[BindingFact]] = defaultdict(list)
    unresolved_owners = 0
    for fact in facts:
        if fact.kind == "jni_unresolved_declaration":
            unresolved_owners += 1
        elif fact.kind == "jni_export":
            exports[fact.binding_name].append(fact)
        elif fact.kind == "jni_declaration":
            declarations[fact.binding_name].append(fact)
    links = []
    stats = {
        "ffi_jni_declarations": sum(map(len, declarations.values())) + unresolved_owners,
        "ffi_jni_exports": sum(map(len, exports.values())),
        "ffi_ambiguous_bindings": 0,
        "ffi_unresolved_bindings": unresolved_owners,
        "ffi_jni_unresolved_owners": unresolved_owners,
        "ffi_jni_matching_declarations": 0,
        "ffi_jni_conflicting_declarations": 0,
        "ffi_jni_unknown_declarations": 0,
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
            contract = compare_jni_declarations(method.java_signature, target.native_signature)
            stat = {
                "matching_declaration": "ffi_jni_matching_declarations",
                "conflicting_declaration": "ffi_jni_conflicting_declarations",
                "unknown": "ffi_jni_unknown_declarations",
            }[contract["status"]]
            stats[stat] += 1
            links.append(
                {
                    "relationship": "declares_native_binding",
                    "bridge": "JNI",
                    "evidence": "jni_declaration_export",
                    "confidence": "static_binding_candidate",
                    "equivalence_verified": False,
                    "runtime_verified": False,
                    "abi_verified": False,
                    "declared_contract": contract,
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
