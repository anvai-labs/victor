"""Cache compatibility includes producer generation and explicit custom scope."""

import pytest

from victor.core.graph_rag import body_provenance


def profile(**kwargs):
    return body_provenance.producer_profile("a" * 64, component_hash="b" * 64, **kwargs)


def test_runtime_library_upgrade_invalidates_cache_compatibility(monkeypatch):
    monkeypatch.setattr(body_provenance, "library_versions", lambda: {"torch": "old"})
    before = profile()
    monkeypatch.setattr(body_provenance, "library_versions", lambda: {"torch": "new"})
    after = profile()
    assert before.identity != after.identity
    assert before.manifest["libraries"] == {"torch": "old"}


def test_unchanged_runtime_and_component_are_repeatable():
    assert profile().identity == profile().identity
    assert (
        profile().identity
        != body_provenance.producer_profile("a" * 64, component_hash="c" * 64).identity
    )


def test_undeclared_custom_producers_are_isolated_by_encoder_instance():
    first = profile(instance_nonce="first")
    second = profile(instance_nonce="second")
    assert first.identity != second.identity
    assert first.reuse_scope == "encoder_instance"


def test_declared_custom_producer_can_reuse_across_encoder_instances():
    assert (
        profile(custom_id="test/encoder-v1").identity
        == profile(custom_id="test/encoder-v1").identity
    )
    assert (
        profile(custom_id="test/encoder-v1").identity
        != profile(custom_id="test/encoder-v2").identity
    )


def test_returned_manifest_cannot_mutate_profile_identity():
    producer = profile()
    original = producer.manifest
    producer.manifest["libraries"]["torch"] = "tampered"
    assert producer.manifest == original


@pytest.mark.parametrize("raw", ["{}", "null", "[]", "not-json", '{"schema":"other"}'])
def test_direct_profile_constructor_rejects_unvalidated_metadata(raw):
    with pytest.raises(ValueError):
        body_provenance.ProducerProfile(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unknown"),
        ("cache_reuse_scope", "unknown"),
        ("libraries", []),
        ("libraries", {"torch": 2}),
        ("python", ""),
    ],
)
def test_direct_profile_constructor_validates_fields(field, value):
    import json

    manifest = profile().manifest
    manifest[field] = value
    with pytest.raises(ValueError):
        body_provenance.ProducerProfile(json.dumps(manifest, sort_keys=True, separators=(",", ":")))


def test_direct_profile_constructor_rejects_noncanonical_json():
    import json

    with pytest.raises(ValueError, match="canonical"):
        body_provenance.ProducerProfile(json.dumps(profile().manifest, indent=2))


def test_callable_fingerprint_is_stable_after_interpreter_warmup():
    def function(value):
        return tuple(x + 3 for x in value), frozenset({1, 2})

    before = body_provenance.callable_code_fingerprint(function)
    for _ in range(2000):
        function((1, 2))
    assert body_provenance.callable_code_fingerprint(function) == before


def test_callable_fingerprint_retains_wrapped_implementation_and_nested_constants():
    from functools import wraps

    def original(value):
        return value + 1

    @wraps(original)
    def wrapped(value):
        return -original(value)

    assert body_provenance.callable_code_fingerprint(
        wrapped
    ) != body_provenance.callable_code_fingerprint(original)
    namespace = {}
    exec(compile("def f():\n return lambda x: x+2\n", "one.py", "exec"), namespace)
    before = body_provenance.callable_code_fingerprint(namespace["f"])
    exec(compile("def f():\n return lambda x: x+3\n", "one.py", "exec"), namespace)
    assert body_provenance.callable_code_fingerprint(namespace["f"]) != before


@pytest.mark.parametrize(
    "parameters",
    [
        {"component_hash": "invalid"},
        {"custom_id": ""},
        {"custom_id": "x" * 257},
        {"instance_nonce": ""},
        {"custom_id": "test\ninvalid"},
    ],
)
def test_invalid_producer_declarations_reject(parameters):
    values = {"component_hash": "b" * 64, **parameters}
    with pytest.raises(ValueError):
        body_provenance.producer_profile("a" * 64, **values)
