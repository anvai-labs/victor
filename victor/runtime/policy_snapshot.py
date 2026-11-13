"""Immutable policy snapshot admitted for a run.

FEP-0044 Phase 1 records only the non-secret sandbox policy needed to detect a
reload between admission and dispatch. The snapshot is deliberately small: it
is not a second policy engine and does not authorize anything by itself.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


class PolicySnapshotError(RuntimeError):
    """A run-policy snapshot could not be established or verified."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class RunPolicySnapshot:
    """Typed immutable copy of the admitted sandbox policy.

    ``digest`` binds every policy field. Persisted metadata is safe to store in
    the existing paused-run record because it contains no credentials or paths
    beyond the operator-configured mount allow-list.
    """

    schema_version: int
    effective_policy: str
    sandbox_enabled: bool
    filesystem_mode: str
    namespace_restrictions: bool
    network_isolation: bool
    allowed_mounts: tuple[str, ...]
    digest: str

    @classmethod
    def from_settings(cls, settings: Any) -> RunPolicySnapshot:
        """Copy a successfully loaded settings object into immutable values.

        Any settings read or type failure raises before a run is admitted. A
        successful read with ``sandbox=None`` is the existing explicit disabled
        policy and remains valid.
        """
        try:
            if settings is None:
                raise ValueError("settings object is required")
            sandbox = getattr(settings, "sandbox", None)
            sandbox_enabled = (
                False if sandbox is None else getattr(sandbox, "sandbox_enabled", False)
            )
            policy = cls._effective_policy(sandbox)
            filesystem_mode = (
                "workspace-only"
                if sandbox is None
                else getattr(sandbox, "sandbox_filesystem_mode", "workspace-only")
            )
            namespace_restrictions = (
                True
                if sandbox is None
                else getattr(sandbox, "sandbox_namespace_restrictions", True)
            )
            network_isolation = (
                False if sandbox is None else getattr(sandbox, "sandbox_network_isolation", False)
            )
            raw_mounts = [] if sandbox is None else getattr(sandbox, "sandbox_allowed_mounts", [])
            if type(sandbox_enabled) is not bool:
                raise ValueError("sandbox_enabled must be boolean")
            if policy not in {"disabled", "preferred", "required"}:
                raise ValueError(f"unknown sandbox policy: {policy!r}")
            if not isinstance(filesystem_mode, str) or not filesystem_mode:
                raise ValueError("sandbox_filesystem_mode must be a nonempty string")
            if type(namespace_restrictions) is not bool:
                raise ValueError("sandbox_namespace_restrictions must be boolean")
            if type(network_isolation) is not bool:
                raise ValueError("sandbox_network_isolation must be boolean")
            if not isinstance(raw_mounts, (list, tuple)):
                raise ValueError("sandbox_allowed_mounts must be a list or tuple")
            allowed_mounts = tuple(raw_mounts)
            if any(not isinstance(mount, str) or not mount for mount in allowed_mounts):
                raise ValueError("sandbox_allowed_mounts entries must be nonempty strings")
        except PolicySnapshotError:
            raise
        except Exception as exc:
            raise PolicySnapshotError(
                "settings_read_failed",
                f"Run policy could not be established ({exc}); refusing execution.",
            ) from exc

        values = {
            "schema_version": 1,
            "effective_policy": policy,
            "sandbox_enabled": sandbox_enabled,
            "filesystem_mode": filesystem_mode,
            "namespace_restrictions": namespace_restrictions,
            "network_isolation": network_isolation,
            "allowed_mounts": allowed_mounts,
        }
        return cls(**values, digest=cls._digest(values))

    @classmethod
    def from_metadata(cls, metadata: Any) -> RunPolicySnapshot:
        """Verify and restore persisted snapshot metadata."""
        required = {
            "schema_version",
            "effective_policy",
            "sandbox_enabled",
            "filesystem_mode",
            "namespace_restrictions",
            "network_isolation",
            "allowed_mounts",
            "digest",
        }
        try:
            if not isinstance(metadata, Mapping) or set(metadata) != required:
                raise ValueError("snapshot fields do not match schema version 1")
            if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
                raise ValueError("unsupported snapshot schema version")
            raw_mounts = metadata["allowed_mounts"]
            if not isinstance(raw_mounts, list):
                raise ValueError("allowed_mounts must be a list")
            if metadata["effective_policy"] not in {"disabled", "preferred", "required"}:
                raise ValueError("unknown effective policy")
            if type(metadata["sandbox_enabled"]) is not bool:
                raise ValueError("sandbox_enabled must be boolean")
            if not isinstance(metadata["filesystem_mode"], str) or not metadata["filesystem_mode"]:
                raise ValueError("filesystem_mode must be a nonempty string")
            if type(metadata["namespace_restrictions"]) is not bool:
                raise ValueError("namespace_restrictions must be boolean")
            if type(metadata["network_isolation"]) is not bool:
                raise ValueError("network_isolation must be boolean")
            if any(not isinstance(mount, str) or not mount for mount in raw_mounts):
                raise ValueError("allowed_mounts entries must be nonempty strings")
            values = {
                "schema_version": metadata["schema_version"],
                "effective_policy": metadata["effective_policy"],
                "sandbox_enabled": metadata["sandbox_enabled"],
                "filesystem_mode": metadata["filesystem_mode"],
                "namespace_restrictions": metadata["namespace_restrictions"],
                "network_isolation": metadata["network_isolation"],
                "allowed_mounts": tuple(raw_mounts),
            }
            expected_digest = cls._digest(values)
            digest = metadata["digest"]
            if not isinstance(digest, str) or not hmac.compare_digest(digest, expected_digest):
                raise ValueError("snapshot digest does not match policy fields")
        except PolicySnapshotError:
            raise
        except Exception as exc:
            raise PolicySnapshotError(
                "metadata_invalid",
                f"Run-policy snapshot is invalid or tampered ({exc}).",
            ) from exc

        return cls(**values, digest=metadata["digest"])

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable immutable copy for paused-run storage."""
        values = asdict(self)
        values["allowed_mounts"] = list(self.allowed_mounts)
        return values

    @staticmethod
    def _effective_policy(sandbox: Any) -> str:
        if sandbox is None:
            return "disabled"
        explicit = getattr(sandbox, "sandbox_policy", None)
        if explicit in {"disabled", "preferred", "required"}:
            return explicit
        if bool(getattr(sandbox, "sandbox_enabled", False)):
            return "preferred"
        return "disabled"

    @staticmethod
    def _digest(values: Mapping[str, Any]) -> str:
        canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
