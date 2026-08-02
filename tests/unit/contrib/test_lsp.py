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

"""Unit tests for victor.contrib.lsp package."""

import pytest
from pathlib import Path

from victor.contrib.lsp import BasicLSPClient
from victor.framework.vertical_protocols import CompletionItemKind, Location


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Create a sample file for LSP testing."""
    return tmp_path / "test.py"


class TestBasicLSPClient:
    """Test BasicLSPClient implementation."""

    def test_client_info(self) -> None:
        """Test client metadata retrieval."""
        lsp = BasicLSPClient()
        info = lsp.get_server_info()

        assert info["name"] == "BasicLSPClient"
        assert info["version"] == "1.0.0"
        assert "capabilities" in info
        assert "stub_only" in info["capabilities"]
        assert "note" in info["info"]
        assert "victor-coding" in info["note"]

    @pytest.mark.asyncio
    async def test_start_server_returns_false(self, sample_file: Path) -> None:
        """Test that start_server returns False (stub implementation)."""
        lsp = BasicLSPClient()
        result = await lsp.start_server(
            language="python",
            file_path=sample_file,
        )

        assert result is False  # Stub implementation returns False

    @pytest.mark.asyncio
    async def test_stop_server_returns_true(self, sample_file: Path) -> None:
        """Test that stop_server returns True (no server to stop)."""
        lsp = BasicLSPClient()
        result = await lsp.stop_server(file_path=sample_file)

        assert result is True  # No server to stop, so returns True

    @pytest.mark.asyncio
    async def test_get_completions_returns_empty(self, sample_file: Path) -> None:
        """Test that get_completions returns empty list."""
        lsp = BasicLSPClient()
        result = await lsp.get_completions(
            file_path=sample_file,
            line=5,
            character=10,
        )

        assert result == []  # Stub returns empty list

    @pytest.mark.asyncio
    async def test_get_definition_returns_none(self, sample_file: Path) -> None:
        """Test that get_definition returns None."""
        lsp = BasicLSPClient()
        result = await lsp.get_definition(
            file_path=sample_file,
            line=5,
            character=10,
        )

        assert result is None  # Stub returns None

    @pytest.mark.asyncio
    async def test_get_hover_returns_none(self, sample_file: Path) -> None:
        """Test that get_hover returns None."""
        lsp = BasicLSPClient()
        result = await lsp.get_hover(
            file_path=sample_file,
            line=5,
            character=10,
        )

        assert result is None  # Stub returns None

    @pytest.mark.asyncio
    async def test_get_diagnostics_returns_empty(self, sample_file: Path) -> None:
        """Test that get_diagnostics returns empty list."""
        lsp = BasicLSPClient()
        result = await lsp.get_diagnostics(file_path=sample_file)

        assert result == []  # Stub returns empty list

    def test_stub_only_capability(self) -> None:
        """Test that capabilities reflect stub-only nature."""
        lsp = BasicLSPClient()
        info = lsp.get_server_info()

        # All operations should return empty/None results
        assert "stub_only" in info["capabilities"]


class TestLSPProtocolCompliance:
    """Test that BasicLSPClient complies with LanguageServerProtocol."""

    @pytest.mark.asyncio
    async def test_all_methods_callable(self, sample_file: Path) -> None:
        """Test that all protocol methods are callable."""
        lsp = BasicLSPClient()

        # All methods should be callable without errors
        await lsp.start_server("python", sample_file)
        await lsp.stop_server(sample_file)
        await lsp.get_completions(sample_file, 0, 0)
        await lsp.get_definition(sample_file, 0, 0)
        await lsp.get_hover(sample_file, 0, 0)
        await lsp.get_diagnostics(sample_file)
        lsp.get_server_info()

    @pytest.mark.asyncio
    async def test_returns_expected_types(self, sample_file: Path) -> None:
        """Test that methods return expected types."""
        lsp = BasicLSPClient()

        # start_server should return bool
        start_result = await lsp.start_server("python", sample_file)
        assert isinstance(start_result, bool)

        # stop_server should return bool
        stop_result = await lsp.stop_server(sample_file)
        assert isinstance(stop_result, bool)

        # get_completions should return list
        completions = await lsp.get_completions(sample_file, 0, 0)
        assert isinstance(completions, list)

        # get_definition should return Location or None
        definition = await lsp.get_definition(sample_file, 0, 0)
        assert definition is None or isinstance(definition, Location)

        # get_hover should return HoverInfo or None
        hover = await lsp.get_hover(sample_file, 0, 0)
        assert hover is None or hasattr(hover, "contents")

        # get_diagnostics should return list
        diagnostics = await lsp.get_diagnostics(sample_file)
        assert isinstance(diagnostics, list)

        # get_server_info should return dict
        info = lsp.get_server_info()
        assert isinstance(info, dict)
