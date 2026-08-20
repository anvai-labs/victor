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

"""Tests for unified LSP tool module.

These tests require victor-coding package to be installed and are
marked as integration tests.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from victor.tools.lsp_tool import lsp, KIND_NAMES

# Check if LSP capability is available
try:
    from victor.core.capability_registry import CapabilityRegistry
    from victor.framework.vertical_protocols import LSPManagerProtocol

    LSP_AVAILABLE = CapabilityRegistry.get_instance().get(LSPManagerProtocol) is not None
except Exception:
    LSP_AVAILABLE = False

# Mark all tests in this module as integration tests (require victor-coding)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not LSP_AVAILABLE,
        reason="LSP capability not available. Requires victor-coding with LSP support",
    ),
]


@dataclass
class MockCompletionItem:
    """Mock completion item."""

    label: str
    kind: int
    detail: str = None
    insert_text: str = None


@dataclass
class MockHover:
    """Mock hover result."""

    contents: str


@dataclass
class MockLSPStatus:
    """Mock LSP status."""

    server_name: str
    running: bool
    initialized: bool
    open_documents: int
    capabilities: list


class TestLspUnknownAction:
    """Tests for unknown action handling."""

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        """Test unknown action returns error."""
        result = await lsp(action="invalid_action")
        assert result["success"] is False
        assert "Unknown action" in result["error"]


class TestLspStatus:
    """Tests for lsp status action."""

    @pytest.mark.asyncio
    async def test_status_success(self):
        """Test successful status check."""
        mock_manager = MagicMock()
        mock_manager.get_status.return_value = {
            "python": MockLSPStatus(
                server_name="pylsp",
                running=True,
                initialized=True,
                open_documents=3,
                capabilities=["completion", "hover"],
            )
        }
        mock_manager.get_available_servers.return_value = [
            {"language": "python", "name": "pylsp", "installed": True, "running": True},
            {
                "language": "typescript",
                "name": "tsserver",
                "installed": True,
                "running": False,
            },
        ]

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            result = await lsp(action="status")

            assert result["success"] is True
            assert "servers" in result
            assert "available" in result
            assert len(result["available"]) == 2


class TestLspStart:
    """Tests for lsp start action."""

    @pytest.mark.asyncio
    async def test_start_missing_language(self):
        """Test start with missing language parameter."""
        result = await lsp(action="start")
        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_start_unknown_language(self):
        """Test start with unknown language."""
        mock_manager = MagicMock()

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            with patch("victor_coding.lsp.config.LANGUAGE_SERVERS", {}):
                result = await lsp(action="start", language="unknown_lang")

                assert result["success"] is False
                assert "Unknown language" in result["error"]

    @pytest.mark.asyncio
    async def test_start_success(self):
        """Test successful server start."""
        mock_manager = MagicMock()
        mock_manager.start_server = AsyncMock(return_value=True)

        mock_config = MagicMock()
        mock_config.name = "Python LSP"
        mock_config.install_command = "pip install python-lsp-server"

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            with patch("victor_coding.lsp.config.LANGUAGE_SERVERS", {"python": mock_config}):
                result = await lsp(action="start", language="python")

                assert result["success"] is True
                assert "Started" in result["message"]
                mock_manager.start_server.assert_called_once_with("python")

    @pytest.mark.asyncio
    async def test_start_failure(self):
        """Test server start failure."""
        mock_manager = MagicMock()
        mock_manager.start_server = AsyncMock(return_value=False)

        mock_config = MagicMock()
        mock_config.name = "Python LSP"
        mock_config.install_command = "pip install python-lsp-server"

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            with patch("victor_coding.lsp.config.LANGUAGE_SERVERS", {"python": mock_config}):
                result = await lsp(action="start", language="python")

                assert result["success"] is False
                assert "Failed to start" in result["error"]


class TestLspStop:
    """Tests for lsp stop action."""

    @pytest.mark.asyncio
    async def test_stop_missing_language(self):
        """Test stop with missing language parameter."""
        result = await lsp(action="stop")
        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_stop_success(self):
        """Test successful server stop."""
        mock_manager = MagicMock()
        mock_manager.stop_server = AsyncMock()

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            result = await lsp(action="stop", language="python")

            assert result["success"] is True
            assert "Stopped" in result["message"]
            mock_manager.stop_server.assert_called_once_with("python")


class TestLspCompletions:
    """Tests for lsp completions action."""

    @pytest.mark.asyncio
    async def test_completions_missing_file_path(self):
        """Test completions with missing file_path."""
        result = await lsp(action="completions", line=10, character=5)
        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_completions_missing_line(self):
        """Test completions with missing line."""
        result = await lsp(action="completions", file_path="test.py", character=5)
        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_completions_missing_character(self):
        """Test completions with missing character."""
        result = await lsp(action="completions", file_path="test.py", line=10)
        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_completions_success(self):
        """Test successful completions."""
        mock_manager = MagicMock()
        mock_manager.open_document = AsyncMock(return_value=True)
        mock_manager.get_completions = AsyncMock(
            return_value=[
                MockCompletionItem(label="print", kind=3, detail="builtin"),
                MockCompletionItem(label="len", kind=3, detail="builtin"),
            ]
        )

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            result = await lsp(
                action="completions",
                file_path="test.py",
                line=10,
                character=5,
            )

            assert result["success"] is True
            assert result["count"] == 2
            assert len(result["completions"]) == 2
            assert result["completions"][0]["label"] == "print"

    @pytest.mark.asyncio
    async def test_completions_empty(self):
        """Test completions with no results."""
        mock_manager = MagicMock()
        mock_manager.open_document = AsyncMock(return_value=True)
        mock_manager.get_completions = AsyncMock(return_value=[])

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            result = await lsp(
                action="completions",
                file_path="test.py",
                line=10,
                character=5,
            )

            assert result["success"] is False
            assert result["count"] == 0


class TestLspHover:
    """Tests for lsp hover action."""

    @pytest.mark.asyncio
    async def test_hover_missing_params(self):
        """Test hover with missing parameters."""
        result = await lsp(action="hover")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_hover_success(self):
        """Test successful hover."""
        mock_manager = MagicMock()
        mock_manager.open_document = AsyncMock(return_value=True)
        mock_manager.get_hover = AsyncMock(
            return_value=MockHover(contents="def print(*args) -> None")
        )

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            result = await lsp(
                action="hover",
                file_path="test.py",
                line=10,
                character=5,
            )

            assert result["success"] is True
            assert "print" in result["contents"]

    @pytest.mark.asyncio
    async def test_hover_no_info(self):
        """Test hover with no information."""
        mock_manager = MagicMock()
        mock_manager.open_document = AsyncMock(return_value=True)
        mock_manager.get_hover = AsyncMock(return_value=None)

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            result = await lsp(
                action="hover",
                file_path="test.py",
                line=10,
                character=5,
            )

            assert result["success"] is False
            assert "No hover information" in result["message"]


class TestLspDefinition:
    """Tests for lsp definition action."""

    @pytest.mark.asyncio
    async def test_definition_missing_params(self):
        """Test definition with missing parameters."""
        result = await lsp(action="definition")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_definition_success(self):
        """Test successful go to definition."""
        mock_manager = MagicMock()
        mock_manager.open_document = AsyncMock(return_value=True)
        mock_manager.get_definition = AsyncMock(
            return_value=[{"uri": "file:///utils.py", "line": 15, "character": 0}]
        )

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            result = await lsp(
                action="definition",
                file_path="test.py",
                line=10,
                character=5,
            )

            assert result["success"] is True
            assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_definition_not_found(self):
        """Test definition not found."""
        mock_manager = MagicMock()
        mock_manager.open_document = AsyncMock(return_value=True)
        mock_manager.get_definition = AsyncMock(return_value=[])

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            result = await lsp(
                action="definition",
                file_path="test.py",
                line=10,
                character=5,
            )

            assert result["success"] is False
            assert result["count"] == 0


class TestLspReferences:
    """Tests for lsp references action."""

    @pytest.mark.asyncio
    async def test_references_missing_params(self):
        """Test references with missing parameters."""
        result = await lsp(action="references")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_references_success(self):
        """Test successful find references."""
        mock_manager = MagicMock()
        mock_manager.open_document = AsyncMock(return_value=True)
        mock_manager.get_references = AsyncMock(
            return_value=[
                {"uri": "file:///a.py", "line": 10},
                {"uri": "file:///b.py", "line": 20},
                {"uri": "file:///c.py", "line": 30},
            ]
        )

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            result = await lsp(
                action="references",
                file_path="test.py",
                line=10,
                character=5,
            )

            assert result["success"] is True
            assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_references_not_found(self):
        """Test references not found."""
        mock_manager = MagicMock()
        mock_manager.open_document = AsyncMock(return_value=True)
        mock_manager.get_references = AsyncMock(return_value=[])

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            result = await lsp(
                action="references",
                file_path="test.py",
                line=10,
                character=5,
            )

            assert result["success"] is False
            assert result["count"] == 0


class TestLspDiagnostics:
    """Tests for lsp diagnostics action."""

    @pytest.mark.asyncio
    async def test_diagnostics_missing_file_path(self):
        """Test diagnostics with missing file_path."""
        result = await lsp(action="diagnostics")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_diagnostics_success(self):
        """Test successful diagnostics."""
        mock_manager = MagicMock()
        mock_manager.open_document = AsyncMock(return_value=True)
        mock_manager.get_diagnostics.return_value = [
            {"severity": "error", "message": "Undefined variable"},
            {"severity": "warning", "message": "Unused import"},
            {"severity": "info", "message": "Hint"},
        ]

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await lsp(action="diagnostics", file_path="test.py")

                assert result["success"] is True
                assert result["errors"] == 1
                assert result["warnings"] == 1
                assert result["info"] == 1


class TestLspOpen:
    """Tests for lsp open action."""

    @pytest.mark.asyncio
    async def test_open_missing_file_path(self):
        """Test open with missing file_path."""
        result = await lsp(action="open")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_open_success(self):
        """Test successful open."""
        mock_manager = MagicMock()
        mock_manager.open_document = AsyncMock(return_value=True)

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            result = await lsp(action="open", file_path="test.py")

            assert result["success"] is True
            assert "Opened" in result["message"]

    @pytest.mark.asyncio
    async def test_open_no_server(self):
        """Test open when no server available."""
        mock_manager = MagicMock()
        mock_manager.open_document = AsyncMock(return_value=False)

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            result = await lsp(action="open", file_path="test.py")

            assert result["success"] is False
            assert "Could not open" in result["error"]


class TestLspClose:
    """Tests for lsp close action."""

    @pytest.mark.asyncio
    async def test_close_missing_file_path(self):
        """Test close with missing file_path."""
        result = await lsp(action="close")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_close_success(self):
        """Test successful close."""
        mock_manager = MagicMock()
        mock_manager.close_document = MagicMock()

        with patch("victor_coding.lsp.manager.get_lsp_manager", return_value=mock_manager):
            result = await lsp(action="close", file_path="test.py")

            assert result["success"] is True
            assert "Closed" in result["message"]


class TestKindNames:
    """Tests for completion kind names."""

    def test_kind_names_contains_common_types(self):
        """Test that KIND_NAMES contains common completion types."""
        assert KIND_NAMES[1] == "Text"
        assert KIND_NAMES[2] == "Method"
        assert KIND_NAMES[3] == "Function"
        assert KIND_NAMES[6] == "Variable"
        assert KIND_NAMES[7] == "Class"

    def test_kind_names_complete(self):
        """Test KIND_NAMES has entries for 1-25."""
        for i in range(1, 26):
            assert i in KIND_NAMES
