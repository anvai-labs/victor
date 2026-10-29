# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit wire projection with mandatory caller-side canonical validation.

OpenAI strict output accepts a JSON Schema subset. This helper handles the
observed unsupported ``uniqueItems`` constraint and missing primitive enum/const
types. It does not promise to translate every JSON Schema feature. Providers do
not apply it implicitly: callers must retain the contract and validate completed
JSON before accepting it. No network, request retry or provider choice occurs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator, SchemaError

__all__ = ["OpenAIResponseSchema", "prepare_openai_response_schema"]


@dataclass(frozen=True)
class OpenAIResponseSchema:
    """Immutable serialized schemas; returned dictionaries are independent copies.

    ``wire_schema`` is a generation aid, not the full response contract. Call
    ``validate`` on the completed JSON, including after consuming a stream.
    ``local_constraint_paths`` identifies constraints omitted only on the wire.
    """

    _canonical_json: str
    _wire_json: str
    local_constraint_paths: tuple[str, ...]

    @property
    def canonical_schema(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)  # type: ignore[no-any-return]

    @property
    def wire_schema(self) -> dict[str, Any]:
        return json.loads(self._wire_json)  # type: ignore[no-any-return]

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self._canonical_json.encode()).hexdigest()

    @property
    def wire_sha256(self) -> str:
        return hashlib.sha256(self._wire_json.encode()).hexdigest()

    def validate(self, content: Any) -> None:
        """Reject non-JSON values, then enforce every original Draft 2020-12 constraint."""
        try:
            parsed = json.loads(_serialize(content))
            if parsed != content:
                raise ValueError("Response contains non-JSON values or non-string object keys")
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(
                "Response must contain finite JSON values and string object keys"
            ) from exc
        Draft202012Validator(self.canonical_schema).validate(parsed)


def _serialize(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _primitive_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    raise ValueError("Missing enum/const types can only be inferred for JSON primitives")


def prepare_openai_response_schema(schema: dict[str, Any]) -> OpenAIResponseSchema:
    """Compile an explicit limited wire projection and retain the full contract.

    Input must be a bounded object schema with local-only references. Enum and
    const literals are data; dictionaries under those keywords are never walked
    as schemas. Unknown wire restrictions remain visible to the provider rather
    than being silently removed. Callers own request/response storage and must
    validate the completed response before acceptance. No tool schema is changed.
    """
    if not isinstance(schema, dict):
        raise ValueError("Response schema must be an object")
    try:
        canonical = _serialize(schema)
    except (SchemaError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Response schema must be valid finite JSON Schema") from exc
    if len(canonical.encode()) > 250_000:
        raise ValueError("Response schema exceeds the 250000-byte projection limit")
    wire = json.loads(canonical)
    if wire != schema:
        raise ValueError("Response schema must use JSON values and string object keys")
    if wire.get("type") != "object":
        raise ValueError("OpenAI response projection requires an object root")
    try:
        Draft202012Validator.check_schema(wire)
    except (SchemaError, RecursionError) as exc:
        raise ValueError("Response schema must be valid Draft 2020-12 JSON Schema") from exc
    omitted: list[str] = []

    def pointer(path: tuple[str, ...]) -> str:
        return "/" + "/".join(p.replace("~", "~0").replace("/", "~1") for p in path)

    def walk(node: Any, path: tuple[str, ...] = ()) -> None:
        if not isinstance(node, dict):
            return
        if len(path) > 100:
            raise ValueError("Response schema exceeds the projection depth limit")
        if node.get("$schema") not in (
            None,
            "https://json-schema.org/draft/2020-12/schema",
            "https://json-schema.org/draft/2020-12/schema#",
        ):
            raise ValueError("Response projection requires the Draft 2020-12 dialect")
        if "$id" in node:
            raise ValueError("Response projection requires local schemas without $id rebasing")
        for key in ("$ref", "$dynamicRef"):
            if key in node and not node[key].startswith("#"):
                raise ValueError("Response projection permits local schema references only")
        if "uniqueItems" in node:
            if node.pop("uniqueItems"):
                omitted.append(pointer(path + ("uniqueItems",)))
        if "type" not in node and ("const" in node or "enum" in node):
            values = [node["const"]] if "const" in node else node["enum"]
            types = {_primitive_type(value) for value in values}
            if not types:
                raise ValueError("An empty enum cannot be projected to an OpenAI primitive type")
            if "number" in types:
                types.discard("integer")
            node["type"] = next(iter(types)) if len(types) == 1 else sorted(types)
        for key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
            for name, child in node.get(key, {}).items():
                walk(child, path + (key, name))
        for key in (
            "items",
            "additionalProperties",
            "contains",
            "not",
            "if",
            "then",
            "else",
            "propertyNames",
            "unevaluatedItems",
            "unevaluatedProperties",
        ):
            walk(node.get(key), path + (key,))
        for key in ("anyOf", "allOf", "oneOf", "prefixItems"):
            for index, child in enumerate(node.get(key, [])):
                walk(child, path + (key, str(index)))

    walk(wire)
    return OpenAIResponseSchema(canonical, _serialize(wire), tuple(omitted))
