"""A failed safety extension must not silently remove coding restrictions."""

from unittest.mock import MagicMock, patch

import pytest


def test_coding_safety_provider_propagates_extension_failures():
    from victor_coding.protocols import CodingSafetyProvider

    with patch(
        "victor_coding.protocols.CodingSafetyExtension", side_effect=RuntimeError("init failed")
    ):
        with pytest.raises(RuntimeError, match="init failed"):
            CodingSafetyProvider()
    extension = MagicMock()
    with patch("victor_coding.protocols.CodingSafetyExtension", return_value=extension):
        provider = CodingSafetyProvider()
    for method in ("get_bash_patterns", "get_file_patterns", "get_tool_restrictions"):
        getattr(extension, method).side_effect = RuntimeError("lookup failed")
        with pytest.raises(RuntimeError, match="lookup failed"):
            getattr(provider, method)()
