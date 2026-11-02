"""Declared JNI contracts do not establish compiled ABI or runtime identity."""

import pytest

from victor.core.graph_rag.jni_contracts import (
    JavaSignature,
    NativeSignature,
    compare_jni_declarations,
)


def compare(
    parameters=("I",),
    result="I",
    native=("JNIEnv*", "jobject", "jint"),
    native_result="jint",
    static=False,
    standard=True,
):
    return compare_jni_declarations(
        JavaSignature(parameters, result, static, standard),
        NativeSignature(native, native_result),
    )


def test_standard_primitive_declarations_match_without_abi_claim():
    result = compare()
    assert result["status"] == "matching_declaration"
    assert result["schema"] == "jni-declared-contract-v1"
    assert result["abi_verified"] is False
    assert result["runtime_verified"] is False


@pytest.mark.parametrize(
    "parameters,result,native,native_result",
    [
        (("I",), "I", ("JNIEnv*", "jobject", "jlong"), "jint"),
        (("I",), "I", ("JNIEnv*", "jobject", "jint"), "jlong"),
        (("I",), "I", ("JNIEnv*", "jobject"), "jint"),
        (("I",), "I", ("JNIEnv*", "jobject", "jint", "jint"), "jint"),
    ],
)
def test_known_primitive_or_arity_conflict_is_retained(parameters, result, native, native_result):
    report = compare(parameters, result, native, native_result)
    assert report["status"] == "conflicting_declaration"
    assert report["conflicts"]


@pytest.mark.parametrize(
    "native",
    [
        ("JNIEnv*", "jobject", "int"),
        ("JNIEnv*", "jobject", "project_integer"),
        ("JNIEnv*", "jobject", None),
    ],
)
def test_machine_width_and_aliases_are_not_guessed(native):
    assert compare(native=native)["status"] == "unknown"


def test_static_receiver_uses_class_spelling_but_reference_alias_is_unknown():
    assert (
        compare(static=True, native=("JNIEnv*", "jclass", "jint"))["status"]
        == "matching_declaration"
    )
    assert compare(static=True)["status"] == "unknown"


@pytest.mark.parametrize(
    "descriptor,spelling",
    [
        ("[I", "jintArray"),
        ("[[I", "jobjectArray"),
        ("[Ljava/lang/String;", "jobjectArray"),
        ("Ljava/lang/String;", "jstring"),
        ("Ljava/lang/Class;", "jclass"),
        ("Ljava/lang/Throwable;", "jthrowable"),
    ],
)
def test_reference_spellings_only_match_exact_expected_form(descriptor, spelling):
    assert (
        compare(parameters=(descriptor,), native=("JNIEnv*", "jobject", spelling))["status"]
        == "matching_declaration"
    )
    assert (
        compare(parameters=(descriptor,), native=("JNIEnv*", "jobject", "jobject"))["status"]
        == "unknown"
    )


def test_partial_unknowns_do_not_hide_independent_primitive_conflict():
    report = compare(parameters=(None, "J"), native=("JNIEnv*", "jobject", "jobject", "jint"))
    assert report["status"] == "conflicting_declaration"
    assert report["unknowns"] and report["conflicts"]


def test_unqualified_calling_convention_and_unavailable_signature_abstain():
    assert compare(standard=False, native=())["status"] == "unknown"
    assert compare_jni_declarations(None, None)["status"] == "unknown"
    assert (
        compare_jni_declarations(JavaSignature((), "V", False), NativeSignature(None, "void"))[
            "status"
        ]
        == "unknown"
    )


def test_void_return_and_empty_java_parameters():
    assert (
        compare(parameters=(), result="V", native=("JNIEnv*", "jobject"), native_result="void")[
            "status"
        ]
        == "matching_declaration"
    )
