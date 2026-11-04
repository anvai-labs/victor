"""Conservative source declaration checks, assuming standard JNI typedefs.

These spellings do not resolve preprocessing, aliases, calling conventions,
linkage or runtime object classes. A match is not ABI verification. References
with different spellings remain unknown: C aliases JNI reference types whereas
C++ models a hierarchy. Platform C integer widths are never inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JavaSignature:
    parameters: tuple[str | None, ...]
    result: str | None
    static: bool
    standard_convention: bool = True


@dataclass(frozen=True)
class NativeSignature:
    parameters: tuple[str | None, ...] | None
    result: str | None


_PRIMITIVES = {
    "Z": "jboolean",
    "B": "jbyte",
    "C": "jchar",
    "S": "jshort",
    "I": "jint",
    "J": "jlong",
    "F": "jfloat",
    "D": "jdouble",
    "V": "void",
}


def _expected(descriptor: str | None) -> str | None:
    if descriptor is None:
        return None
    if descriptor in _PRIMITIVES:
        return _PRIMITIVES[descriptor]
    if descriptor.startswith("["):
        if descriptor[1:] in _PRIMITIVES and descriptor != "[V":
            return _PRIMITIVES[descriptor[1:]] + "Array"
        if descriptor.startswith(("[[", "[L")):
            return "jobjectArray"
        return None
    return {
        "Ljava/lang/Object;": "jobject",
        "Ljava/lang/String;": "jstring",
        "Ljava/lang/Class;": "jclass",
        "Ljava/lang/Throwable;": "jthrowable",
    }.get(descriptor)


def compare_jni_declarations(
    java: JavaSignature | None, native: NativeSignature | None
) -> dict[str, Any]:
    """Compare known source pieces; preserve conflicts alongside unknown pieces."""
    conflicts: list[str] = []
    unknowns: list[str] = []
    if java is None or native is None:
        unknowns.append("signature_unavailable")
    elif not java.standard_convention:
        unknowns.append("annotated_convention_unqualified")
    else:
        expected = ("JNIEnv*", "jclass" if java.static else "jobject") + tuple(
            _expected(value) for value in java.parameters
        )

        def check(label: str, want: str | None, actual: str | None) -> None:
            if want is None or actual is None:
                unknowns.append(label)
            elif want != actual:
                if want in _PRIMITIVES.values() and actual in _PRIMITIVES.values():
                    conflicts.append(label)
                else:
                    unknowns.append(label)

        if native.parameters is None:
            unknowns.append("native_parameters_unqualified")
        else:
            if len(native.parameters) != len(expected):
                conflicts.append("parameter_count")
            for index, (want, actual) in enumerate(zip(expected, native.parameters)):
                check(f"parameter_{index}", want, actual)
        check("return", _expected(java.result), native.result)
    return {
        "schema": "jni-declared-contract-v1",
        "status": (
            "conflicting_declaration"
            if conflicts
            else "unknown" if unknowns else "matching_declaration"
        ),
        "assumptions": ["standard_jni_typedef_spellings", "source_declarations_only"],
        "conflicts": conflicts,
        "unknowns": unknowns,
        "abi_verified": False,
        "runtime_verified": False,
    }
