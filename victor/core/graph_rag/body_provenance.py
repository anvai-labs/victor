"""Model-input identity and producer generation are separate cache contracts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import re
import sys
from types import CodeType
from typing import Any


def _code_value(value: Any) -> Any:
    if isinstance(value, CodeType):
        return {
            "bytecode": value.co_code.hex(),
            "constants": [_code_value(item) for item in value.co_consts],
            "names": value.co_names,
            "varnames": value.co_varnames,
            "freevars": value.co_freevars,
            "cellvars": value.co_cellvars,
            "args": [value.co_argcount, value.co_posonlyargcount, value.co_kwonlyargcount],
            "flags": value.co_flags,
            "exceptiontable": value.co_exceptiontable.hex(),
        }
    if type(value) in (tuple, frozenset):
        items = [_code_value(item) for item in value]
        if type(value) is frozenset:
            items.sort(key=lambda item: json.dumps(item, sort_keys=True))
        return [type(value).__name__, items]
    if type(value) in (type(None), bool, int, float, complex, str, bytes, type(Ellipsis)):
        return [type(value).__name__, repr(value)]
    raise TypeError("Unsupported callable constant in body producer")


def callable_code_fingerprint(value: Any) -> str:
    """Hash actual bytecode, without unwrap, object-reference flags or locations."""
    return hashlib.sha256(
        json.dumps(_code_value(value.__code__), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def library_versions() -> dict[str, str | None]:
    """Read installation metadata without importing/loading the local model stack."""
    result: dict[str, str | None] = {}
    for package in (
        "numpy",
        "sentence-transformers",
        "torch",
        "transformers",
        "tokenizers",
        "victor-codegraph",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


@dataclass(frozen=True)
class ProducerProfile:
    canonical_json: str

    def __post_init__(self) -> None:
        try:
            manifest = json.loads(self.canonical_json)
            expected_keys = {
                "schema",
                "input_profile",
                "component_sha256",
                "libraries",
                "python",
                "platform",
                "machine",
                "cache_reuse_scope",
                "custom_id",
                "instance_nonce",
            }
            if not isinstance(manifest, dict) or set(manifest) != expected_keys:
                raise ValueError("Invalid producer profile fields")
            if manifest["schema"] != "body-vector-producer-v1":
                raise ValueError("Invalid producer profile schema")
            _validate_declarations(
                manifest["input_profile"],
                manifest["component_sha256"],
                manifest["custom_id"],
                manifest["instance_nonce"],
            )
            scope = (
                "encoder_instance"
                if manifest["instance_nonce"] is not None
                else (
                    "declared_producer" if manifest["custom_id"] is not None else "runtime_producer"
                )
            )
            if manifest["cache_reuse_scope"] != scope:
                raise ValueError("Invalid producer reuse scope")
            if not isinstance(manifest["libraries"], dict) or any(
                not isinstance(k, str) or not isinstance(v, (str, type(None)))
                for k, v in manifest["libraries"].items()
            ):
                raise ValueError("Invalid producer library versions")
            if any(
                not isinstance(manifest[k], str) or not manifest[k]
                for k in ("python", "platform", "machine")
            ):
                raise ValueError("Invalid producer runtime metadata")
            if json.dumps(manifest, sort_keys=True, separators=(",", ":")) != self.canonical_json:
                raise ValueError("Producer profile must be canonical JSON")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid producer profile JSON") from exc

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    @property
    def manifest(self) -> dict[str, Any]:
        # Return detached metadata: callers cannot alter a recorded identity.
        return dict(json.loads(self.canonical_json))

    @property
    def reuse_scope(self) -> str:
        return str(self.manifest["cache_reuse_scope"])


def _validate_declarations(
    input_identity: str, component_hash: str, custom_id: str | None, instance_nonce: str | None
) -> None:
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in (input_identity, component_hash)
    ):
        raise ValueError("Producer input/component identities must be SHA-256 hex")
    for value in (custom_id, instance_nonce):
        if value is not None and (
            not isinstance(value, str)
            or not 1 <= len(value) <= 256
            or any(ord(c) < 32 for c in value)
        ):
            raise ValueError("Producer declarations must be nonempty bounded strings")


def producer_profile(
    input_identity: str,
    *,
    component_hash: str,
    custom_id: str | None = None,
    instance_nonce: str | None = None,
) -> ProducerProfile:
    """Bind runtime/code generation and scope a caller-supplied model producer.

    A custom declaration is provenance supplied by its caller, not authenticated
    model quality. Without one, a factory's cache identity is instance-scoped.
    """
    _validate_declarations(input_identity, component_hash, custom_id, instance_nonce)
    descriptor = {
        "schema": "body-vector-producer-v1",
        "input_profile": input_identity,
        "component_sha256": component_hash,
        "libraries": library_versions(),
        "python": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "cache_reuse_scope": (
            "encoder_instance"
            if instance_nonce is not None
            else "declared_producer" if custom_id is not None else "runtime_producer"
        ),
        "custom_id": custom_id,
        "instance_nonce": instance_nonce,
    }
    return ProducerProfile(json.dumps(descriptor, sort_keys=True, separators=(",", ":")))
