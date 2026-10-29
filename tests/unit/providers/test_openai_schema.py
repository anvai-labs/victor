"""Wire projection must retain the caller's stronger local response contract."""

import copy
import json
from unittest.mock import patch

import pytest
from jsonschema import ValidationError

from victor.providers.openai_schema import prepare_openai_response_schema


def selection_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["section", "choices"],
        "properties": {
            "section": {"const": "mechanism"},
            "choices": {
                "type": "array",
                "uniqueItems": True,
                "maxItems": 4,
                "items": {"enum": ["A", "B"]},
            },
        },
    }


def test_projection_preserves_canonical_constraints_and_isolates_copies():
    schema = selection_schema()
    original = copy.deepcopy(schema)
    contract = prepare_openai_response_schema(schema)
    wire = contract.wire_schema
    assert schema == original == contract.canonical_schema
    assert "uniqueItems" not in wire["properties"]["choices"]
    assert wire["properties"]["choices"]["items"]["type"] == "string"
    assert wire["properties"]["section"]["type"] == "string"
    contract.validate({"section": "mechanism", "choices": ["A", "B"]})
    with pytest.raises(ValidationError):
        contract.validate({"section": "mechanism", "choices": ["A", "A"]})
    schema["properties"].clear()
    wire["properties"].clear()
    assert contract.canonical_schema == original
    assert contract.wire_schema["properties"]
    assert contract.canonical_sha256 != contract.wire_sha256
    assert "/properties/choices/uniqueItems" in contract.local_constraint_paths


@pytest.mark.parametrize(
    "response",
    [
        {"section": "wrong", "choices": ["A"]},
        {"section": "mechanism", "choices": ["unknown"]},
        {"section": "mechanism", "choices": ["A"], "extra": True},
    ],
)
def test_canonical_validation_rejects_other_contract_violations(response):
    with pytest.raises(ValidationError):
        prepare_openai_response_schema(selection_schema()).validate(response)


def test_property_names_and_literal_data_are_not_schema_keywords():
    schema = {
        "type": "object",
        "properties": {
            "uniqueItems": {"type": "boolean"},
            "literal": {"const": "uniqueItems"},
        },
        "required": ["uniqueItems", "literal"],
        "additionalProperties": False,
    }
    contract = prepare_openai_response_schema(schema)
    assert "uniqueItems" in contract.wire_schema["properties"]
    assert contract.wire_schema["properties"]["literal"]["const"] == "uniqueItems"
    assert not contract.local_constraint_paths
    contract.validate({"uniqueItems": True, "literal": "uniqueItems"})


def test_nested_local_definitions_and_combinators_keep_full_validation():
    schema = {
        "type": "object",
        "properties": {"choices": {"$ref": "#/$defs/choices"}},
        "required": ["choices"],
        "additionalProperties": False,
        "$defs": {
            "choices": {
                "anyOf": [
                    {"type": "null"},
                    {"type": "array", "items": {"enum": [1, 2]}, "uniqueItems": True},
                ]
            }
        },
    }
    contract = prepare_openai_response_schema(schema)
    assert contract.wire_schema["$defs"]["choices"]["anyOf"][1]["items"]["type"] == "integer"
    contract.validate({"choices": [1, 2]})
    with pytest.raises(ValidationError):
        contract.validate({"choices": [1, 1]})


@pytest.mark.parametrize("reference", ["https://example.invalid/schema", "other.json"])
def test_external_schema_references_are_refused_without_network(reference):
    with patch("socket.socket", side_effect=AssertionError("No network schema resolution")):
        with pytest.raises(ValueError, match="local"):
            prepare_openai_response_schema({"type": "object", "$ref": reference})


def test_json_pointer_escapes_property_names():
    schema = {"type": "object", "properties": {"a/b~c": {"type": "array", "uniqueItems": True}}}
    contract = prepare_openai_response_schema(schema)
    assert contract.local_constraint_paths == ("/properties/a~1b~0c/uniqueItems",)


def test_unspecified_primitive_type_and_invalid_schema_are_not_silently_widened():
    with pytest.raises(ValueError):
        prepare_openai_response_schema({"type": "object", "properties": {"x": {"const": {"a": 1}}}})
    with pytest.raises(ValueError):
        prepare_openai_response_schema({"type": "object", "properties": {"x": {"type": "wrong"}}})
    with pytest.raises(ValueError):
        prepare_openai_response_schema({"type": "object", "description": float("nan")})
    with pytest.raises(ValueError):
        prepare_openai_response_schema({"type": "object", "properties": {1: {"type": "string"}}})
    with pytest.raises(ValueError):
        prepare_openai_response_schema({"type": "object", "properties": {"x": {"enum": []}}})
    json.dumps(prepare_openai_response_schema(selection_schema()).wire_schema, allow_nan=False)


def test_integral_float_remains_valid_for_inferred_integer_const_and_enum():
    from jsonschema import Draft202012Validator

    for constraint in ({"const": 1}, {"enum": [1, 2]}):
        schema = {"type": "object", "properties": {"x": constraint}}
        contract = prepare_openai_response_schema(schema)
        contract.validate({"x": 1.0})
        assert Draft202012Validator(contract.wire_schema).is_valid({"x": 1.0})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), (1, 2)])
def test_response_validation_refuses_non_json_python_values(value):
    contract = prepare_openai_response_schema({"type": "object", "properties": {"x": {}}})
    with pytest.raises(ValueError, match="JSON"):
        contract.validate({"x": value})


def test_other_dialects_cannot_silently_change_canonical_validation_or_reference_rules():
    for schema in (
        {"type": "object", "$schema": "http://json-schema.org/draft-07/schema#"},
        {
            "type": "object",
            "properties": {"x": {"$schema": "http://json-schema.org/draft-07/schema#"}},
        },
    ):
        with pytest.raises(ValueError, match="dialect"):
            prepare_openai_response_schema(schema)
