"""The provider message vocabulary kernel (self-analysis round 2, W4).

Contract pinned here mirrors ``victor.core.shared_types``: the kernel is
pure data, importable with NO provider machinery, and the original
``victor.providers.base`` path re-exports the SAME objects (compat) so
mock.patch targets and isinstance checks stay stable.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

from victor.core import provider_messages
from victor.providers import base as provider_base

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_kernel_imports_without_provider_machinery():
    """A fresh interpreter importing ONLY the kernel must load no
    victor.providers module — the runtime speaks messages without
    importing provider infrastructure (the point of the kernel)."""
    code = (
        "import sys; "
        "import victor.core.provider_messages; "
        "loaded = [m for m in sys.modules if m.startswith('victor.providers')]; "
        "assert not loaded, loaded"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_compat_reexports_are_the_same_objects():
    for name in (
        "Message",
        "ToolDefinition",
        "CompletionResponse",
        "StreamChunk",
        "CacheCostModel",
        "ProviderError",
        "ProviderNotFoundError",
        "ModelNotFoundError",
        "ProviderTimeoutError",
        "ProviderRateLimitError",
        "ProviderAuthError",
        "ProviderConnectionError",
        "ProviderInvalidResponseError",
    ):
        assert getattr(provider_base, name) is getattr(provider_messages, name), name


def test_patch_targets_stay_stable():
    """mock.patch through the legacy path keeps working (the compat name is
    a module attribute as before) and restores the shared object after."""
    with mock.patch("victor.providers.base.Message") as patched:
        assert provider_base.Message is patched  # legacy-path patching works
    assert provider_base.Message is provider_messages.Message  # restored


def test_vocabulary_fields_unchanged():
    """The kernel models keep the exact field surface (field-for-field
    move, no behavior change)."""
    assert set(provider_messages.Message.model_fields) == {
        "role",
        "content",
        "name",
        "tool_calls",
        "tool_call_id",
        "images",
        "metadata",
    }
    assert set(provider_messages.CompletionResponse.model_fields) == {
        "content",
        "role",
        "tool_calls",
        "stop_reason",
        "usage",
        "model",
        "raw_response",
        "metadata",
    }
    assert set(provider_messages.StreamChunk.model_fields) == {
        "content",
        "tool_calls",
        "stop_reason",
        "is_final",
        "metadata",
        "usage",
    }
    assert set(provider_messages.ToolDefinition.model_fields) == {
        "name",
        "description",
        "parameters",
        "schema_level",
    }
    # metadata stays excluded from serialization (never sent to providers).
    message = provider_messages.Message(role="user", content="hi", metadata={"src": "test"})
    dumped = message.model_dump()
    assert "metadata" not in dumped
    assert dumped["role"] == "user" and dumped["content"] == "hi"
    assert message.to_dict() == dumped
