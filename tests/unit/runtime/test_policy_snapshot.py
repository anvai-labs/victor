"""Immutable run-policy snapshot contract (FEP-0044 Phase 1)."""

from types import SimpleNamespace
import hashlib
import json

import pytest

from victor.config.sandbox_settings import SandboxSettings
from victor.runtime.policy_snapshot import (
    PolicySnapshotError,
    RunPolicySnapshot,
)


def test_snapshot_copies_policy_values_and_ignores_later_mutation() -> None:
    sandbox = SandboxSettings(
        sandbox_policy="required",
        sandbox_network_isolation=True,
        sandbox_allowed_mounts=["/data"],
    )
    settings = SimpleNamespace(sandbox=sandbox)

    snapshot = RunPolicySnapshot.from_settings(settings)
    sandbox.sandbox_policy = "disabled"
    sandbox.sandbox_network_isolation = False
    sandbox.sandbox_allowed_mounts.append("/secret")

    assert snapshot.effective_policy == "required"
    assert snapshot.network_isolation is True
    assert snapshot.allowed_mounts == ("/data",)


def test_snapshot_metadata_round_trips_and_binds_digest() -> None:
    snapshot = RunPolicySnapshot.from_settings(
        SimpleNamespace(sandbox=SandboxSettings(sandbox_policy="required"))
    )

    restored = RunPolicySnapshot.from_metadata(snapshot.to_metadata())

    assert restored == snapshot
    assert restored.digest == snapshot.digest
    assert restored.to_metadata() == snapshot.to_metadata()


def test_changed_policy_changes_digest_and_equality() -> None:
    admitted = RunPolicySnapshot.from_settings(
        SimpleNamespace(sandbox=SandboxSettings(sandbox_policy="disabled"))
    )
    reloaded = RunPolicySnapshot.from_settings(
        SimpleNamespace(sandbox=SandboxSettings(sandbox_policy="required"))
    )

    assert admitted != reloaded
    assert admitted.digest != reloaded.digest


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"schema_version": 1},
        {"schema_version": 1, "digest": "0" * 64},
        {
            "schema_version": 1,
            "effective_policy": "required",
            "sandbox_enabled": False,
            "filesystem_mode": "workspace-only",
            "namespace_restrictions": True,
            "network_isolation": False,
            "allowed_mounts": [],
            "digest": "not-a-digest",
        },
    ],
)
def test_malformed_or_tampered_metadata_refuses(metadata: dict) -> None:
    with pytest.raises(PolicySnapshotError):
        RunPolicySnapshot.from_metadata(metadata)


def test_valid_digest_does_not_excuse_wrong_metadata_types() -> None:
    metadata = RunPolicySnapshot.from_settings(
        SimpleNamespace(sandbox=SandboxSettings(sandbox_policy="required"))
    ).to_metadata()
    metadata["sandbox_enabled"] = 1
    digest_values = {
        key: (tuple(value) if key == "allowed_mounts" else value)
        for key, value in metadata.items()
        if key != "digest"
    }
    metadata["digest"] = hashlib.sha256(
        json.dumps(digest_values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    with pytest.raises(PolicySnapshotError):
        RunPolicySnapshot.from_metadata(metadata)


def test_unreadable_or_malformed_settings_refuse_snapshot() -> None:
    class Exploding:
        @property
        def sandbox(self):
            raise OSError("settings unavailable")

    with pytest.raises(PolicySnapshotError):
        RunPolicySnapshot.from_settings(Exploding())

    with pytest.raises(PolicySnapshotError):
        RunPolicySnapshot.from_settings(None)
