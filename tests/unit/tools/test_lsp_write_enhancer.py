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

"""Tests for LSP Write Enhancer.

Tests LSP validation and formatting for multiple supported languages:
- C/C++ (clangd, clang-format)
- Python (pyright/pylsp, black)
- Rust (rust-analyzer, rustfmt)
- JavaScript/TypeScript (tsserver, prettier)
- Go (gopls, gofmt)
- Config files (JSON, YAML, TOML)
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.tools.lsp_write_enhancer import (
    Diagnostic,
    DiagnosticSeverity,
    LSPWriteEnhancer,
    WriteResult,
)


@pytest.fixture(autouse=True)
def reset_enhancer():
    """Reset LSP write enhancer between tests."""
    global _enhancer
    import victor.tools.lsp_write_enhancer as enhancer_module

    enhancer_module._enhancer = None
    yield
    enhancer_module._enhancer = None


class TestDiagnostic:
    """Tests for Diagnostic dataclass."""

    def test_diagnostic_creation(self):
        """Test creating a diagnostic."""
        diagnostic = Diagnostic(
            line=10,
            column=5,
            severity=DiagnosticSeverity.ERROR,
            message="Undefined variable 'x'",
            code="E0602",
            source="pylint",
        )

        assert diagnostic.line == 10
        assert diagnostic.column == 5
        assert diagnostic.severity == DiagnosticSeverity.ERROR
        assert diagnostic.message == "Undefined variable 'x'"
        assert diagnostic.code == "E0602"
        assert diagnostic.source == "pylint"

    def test_diagnostic_without_optional_fields(self):
        """Test creating diagnostic without optional fields."""
        diagnostic = Diagnostic(
            line=1,
            column=0,
            severity=DiagnosticSeverity.WARNING,
            message="Unused import",
        )

        assert diagnostic.code is None
        assert diagnostic.source is None

    def test_severity_is_enum_not_string(self):
        """Regression: severity is a DiagnosticSeverity Enum, so ``== "error"`` is False.

        write()'s simple-mode lsp_suffix previously compared ``d.severity == "error"``
        (a string), which was always False and silently reported "validation passed" even
        for error-level diagnostics. Counting MUST compare against the enum members.
        """
        error = Diagnostic(line=1, column=0, severity=DiagnosticSeverity.ERROR, message="x")
        warning = Diagnostic(line=1, column=0, severity=DiagnosticSeverity.WARNING, message="y")
        # The buggy comparison (string) must be False for both...
        assert error.severity != "error"
        assert warning.severity != "warning"
        # ...while the enum comparison correctly classifies them.
        diagnostics = [error, warning]
        errors = sum(1 for d in diagnostics if d.severity == DiagnosticSeverity.ERROR)
        warnings = sum(1 for d in diagnostics if d.severity == DiagnosticSeverity.WARNING)
        assert errors == 1
        assert warnings == 1


class TestWriteResult:
    """Tests for WriteResult dataclass."""

    def test_write_result_creation(self):
        """Test creating a write result."""
        result = WriteResult(
            success=True,
            path="test.py",
            formatted=True,
            validated=True,
        )

        assert result.success is True
        assert result.path == "test.py"
        assert result.formatted is True
        assert result.validated is True

    def test_has_errors(self):
        """Test checking for errors."""
        result = WriteResult(success=False, path="test.py")

        # No diagnostics
        assert result.has_errors is False

        # Add error diagnostic
        result.diagnostics.append(
            Diagnostic(line=1, column=0, severity=DiagnosticSeverity.ERROR, message="Error")
        )
        assert result.has_errors is True

    def test_has_warnings(self):
        """Test checking for warnings."""
        result = WriteResult(success=True, path="test.py")

        # No diagnostics
        assert result.has_warnings is False

        # Add warning diagnostic
        result.diagnostics.append(
            Diagnostic(line=1, column=0, severity=DiagnosticSeverity.WARNING, message="Warning")
        )
        assert result.has_warnings is True

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = WriteResult(
            success=True,
            path="test.py",
            formatted=True,
            validated=True,
        )
        result.diagnostics = [
            Diagnostic(
                line=1,
                column=0,
                severity=DiagnosticSeverity.ERROR,
                message="Error",
                code="E001",
            ),
            Diagnostic(
                line=2,
                column=0,
                severity=DiagnosticSeverity.WARNING,
                message="Warning",
            ),
        ]

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["path"] == "test.py"
        assert result_dict["formatted"] is True
        assert result_dict["validated"] is True
        assert len(result_dict["diagnostics"]) == 2
        assert result_dict["summary"]["total_diagnostics"] == 2
        assert result_dict["summary"]["errors"] == 1
        assert result_dict["summary"]["warnings"] == 1


class TestLSPWriteEnhancer:
    """Tests for LSPWriteEnhancer class."""

    def test_init_default_workspace(self):
        """Test initialization with default workspace."""
        enhancer = LSPWriteEnhancer()

        assert enhancer._workspace_root == str(Path.cwd())
        assert enhancer._lsp_pool is None
        assert enhancer._language_registry is None

    def test_init_custom_workspace(self):
        """Test initialization with custom workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enhancer = LSPWriteEnhancer(workspace_root=tmpdir)

            assert enhancer._workspace_root == tmpdir

    def test_get_lsp_pool_lazy_initialization(self):
        """Test LSP pool is lazily initialized."""
        enhancer = LSPWriteEnhancer()

        # Pool should be None initially
        assert enhancer._lsp_pool is None

        # The pool is created when first accessed via _get_lsp_pool
        # We can't easily test the actual pool creation without mocking,
        # but we can verify the singleton behavior
        from victor.tools.lsp_write_enhancer import get_write_enhancer

        enhancer2 = get_write_enhancer()
        assert enhancer2 is not None

        # Multiple calls should return the same instance (if workspace root is same)
        enhancer3 = get_write_enhancer()
        assert enhancer2 is enhancer3

    def test_get_language_registry_lazy_initialization(self):
        """Test language registry is lazily initialized."""
        enhancer = LSPWriteEnhancer()

        # Registry should be None initially
        assert enhancer._language_registry is None

        registry = enhancer._get_language_registry()
        assert registry is not None
        assert enhancer._language_registry is not None

    @pytest.mark.asyncio
    async def test_format_with_formatter_no_language(self):
        """Test formatting when language is not detected."""
        enhancer = LSPWriteEnhancer()

        # Unknown file extension
        content, formatter_name = await enhancer.format_with_formatter(
            "test.unknown_ext", "some content"
        )

        # Should return content unchanged
        assert content == "some content"
        assert formatter_name is None

    @pytest.mark.asyncio
    async def test_format_with_formatter_unavailable_formatter(self):
        """Test formatting when formatter is not available."""
        enhancer = LSPWriteEnhancer()

        # Python file but formatter might not be installed
        content, formatter_name = await enhancer.format_with_formatter(
            "test.py", "def hello(): pass"
        )

        # Should return content (possibly formatted or unchanged)
        assert isinstance(content, str)
        # Formatter name depends on system, so we don't assert it


class TestLSPFormatterPolicyBoundary:
    """Formatter fallback launches follow the admitted sandbox policy."""

    @staticmethod
    def _install_policy(monkeypatch, policy: str, backend_available: bool) -> None:
        from victor.config.sandbox_settings import SandboxSettings
        from victor.tools.sandbox import backends

        settings = SimpleNamespace(
            sandbox=SandboxSettings(sandbox_enabled=True, sandbox_policy=policy)
        )
        monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)
        monkeypatch.setattr(backends.platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            backends.shutil,
            "which",
            lambda name: "/usr/bin/bwrap" if backend_available and name == "bwrap" else None,
        )

    @staticmethod
    def _enhancer_with_formatter(tmp_path):
        enhancer = LSPWriteEnhancer(workspace_root=str(tmp_path))
        enhancer._use_thread_formatter = False
        formatter = SimpleNamespace(name="fakefmt", command=["fakefmt"])
        plugin = SimpleNamespace(get_formatter=lambda _root: formatter)
        enhancer._language_registry = SimpleNamespace(
            detect_language=lambda path: "python", get=lambda _language: plugin
        )
        return enhancer

    @staticmethod
    def _trap_launch(monkeypatch, launched: list[list[str]]) -> None:
        import victor.tools.lsp_write_enhancer as lsp_module

        async def _communicate() -> tuple[bytes, bytes]:
            return b"", b""

        async def _create_subprocess_exec(*argv: str, **_kwargs):
            launched.append([str(value) for value in argv])
            return SimpleNamespace(returncode=0, communicate=_communicate)

        monkeypatch.setattr(lsp_module.asyncio, "create_subprocess_exec", _create_subprocess_exec)

    async def test_required_available_wraps_formatter_launch(self, monkeypatch, tmp_path):
        self._install_policy(monkeypatch, "required", backend_available=True)
        launched: list[list[str]] = []
        self._trap_launch(monkeypatch, launched)

        content, formatter_name = await self._enhancer_with_formatter(
            tmp_path
        ).format_with_formatter("example.py", "x = 1\n")

        assert content == "x = 1\n"
        assert formatter_name == "fakefmt"
        assert launched and launched[0][0] == "bwrap"
        assert "fakefmt" in launched[0]
        assert Path(launched[0][-1]).is_relative_to(tmp_path)

    async def test_required_unavailable_skips_formatter_launch(self, monkeypatch, tmp_path):
        self._install_policy(monkeypatch, "required", backend_available=False)
        launched: list[list[str]] = []
        self._trap_launch(monkeypatch, launched)

        content, formatter_name = await self._enhancer_with_formatter(
            tmp_path
        ).format_with_formatter("example.py", "x = 1\n")

        assert content == "x = 1\n"
        assert formatter_name is None
        assert launched == []

    async def test_unreadable_policy_skips_formatter_launch(self, monkeypatch, tmp_path):
        def _unavailable():
            raise RuntimeError("settings unavailable")

        monkeypatch.setattr("victor.config.settings.get_settings", _unavailable)
        launched: list[list[str]] = []
        self._trap_launch(monkeypatch, launched)

        content, formatter_name = await self._enhancer_with_formatter(
            tmp_path
        ).format_with_formatter("example.py", "x = 1\n")

        assert content == "x = 1\n"
        assert formatter_name is None
        assert launched == []

    @pytest.mark.parametrize("policy", ["disabled", "preferred"])
    async def test_non_required_policy_keeps_formatter_raw(self, monkeypatch, tmp_path, policy):
        self._install_policy(monkeypatch, policy, backend_available=True)
        launched: list[list[str]] = []
        self._trap_launch(monkeypatch, launched)

        await self._enhancer_with_formatter(tmp_path).format_with_formatter("example.py", "x = 1\n")

        assert launched and launched[0][0] == "fakefmt"

    async def test_admitted_snapshot_overrides_reloaded_required_settings(
        self, monkeypatch, tmp_path
    ):
        import victor.config.settings as settings_module
        from victor.runtime.policy_snapshot import RunPolicySnapshot

        self._install_policy(monkeypatch, "disabled", backend_available=True)
        admitted = RunPolicySnapshot.from_settings(settings_module.get_settings())
        self._install_policy(monkeypatch, "required", backend_available=False)
        launched: list[list[str]] = []
        self._trap_launch(monkeypatch, launched)

        content, formatter_name = await self._enhancer_with_formatter(
            tmp_path
        ).format_with_formatter("example.py", "x = 1\n", run_policy_snapshot=admitted)

        assert content == "x = 1\n"
        assert formatter_name == "fakefmt"
        assert launched and launched[0][0] == "fakefmt"

    async def test_write_with_lsp_passes_framework_snapshot_to_formatter(self, tmp_path):
        from unittest.mock import AsyncMock

        admitted_policy = object()
        enhancer = self._enhancer_with_formatter(tmp_path)
        enhancer.format_with_formatter = AsyncMock(return_value=("x = 1\n", "fakefmt"))

        await enhancer.write_with_lsp(
            "example.py",
            "x = 1\n",
            validate=False,
            format_code=True,
            write=False,
            _exec_ctx={"run_policy_snapshot": admitted_policy},
        )

        enhancer.format_with_formatter.assert_awaited_once_with(
            "example.py", "x = 1\n", run_policy_snapshot=admitted_policy
        )

    @pytest.mark.parametrize("backend_available", [True, False])
    async def test_filesystem_write_keeps_admitted_formatter_policy(
        self, monkeypatch, tmp_path, backend_available
    ):
        import victor.config.settings as settings_module
        from victor.runtime.policy_snapshot import RunPolicySnapshot
        import victor.tools.filesystem as filesystem_module
        import victor.tools.lsp_write_enhancer as lsp_module

        self._install_policy(monkeypatch, "required", backend_available)
        admitted = RunPolicySnapshot.from_settings(settings_module.get_settings())
        self._install_policy(monkeypatch, "disabled", backend_available)

        enhancer = self._enhancer_with_formatter(tmp_path)
        enhancer.validate_with_lsp = AsyncMock(return_value=[])
        original_format = enhancer.format_with_formatter
        captured_format_kwargs = {}

        async def _capturing_format(*args, **kwargs):
            captured_format_kwargs.update(kwargs)
            return await original_format(*args, **kwargs)

        enhancer.format_with_formatter = _capturing_format
        lsp_module._enhancer = enhancer
        launched: list[list[str]] = []
        self._trap_launch(monkeypatch, launched)

        result = await filesystem_module.write(
            str(tmp_path / "example.py"),
            "x = 1\n",
            format_code=True,
            dry_run=True,
            _exec_ctx={"run_policy_snapshot": admitted},
        )

        assert result["success"] is True
        assert captured_format_kwargs["run_policy_snapshot"] is admitted
        if backend_available:
            assert launched and launched[0][0] == "bwrap"
            assert "fakefmt" in launched[0]
            assert Path(launched[0][-1]).is_relative_to(tmp_path)
        else:
            assert launched == []

    @pytest.mark.asyncio
    async def test_validate_with_lsp_mocked(self):
        """Test LSP validation with mocked LSP pool."""
        enhancer = LSPWriteEnhancer()

        # Mock LSP pool and diagnostics
        mock_pool = AsyncMock()
        mock_pool._path_to_uri = MagicMock(return_value="file://test.py")
        mock_pool.open_document = MagicMock()
        mock_pool.get_diagnostics = MagicMock(
            return_value=[
                {
                    "line": 1,
                    "column": 0,
                    "severity": "error",
                    "message": "Syntax error",
                    "code": "E001",
                }
            ]
        )
        mock_pool.close_document = MagicMock()

        enhancer._lsp_pool = mock_pool

        diagnostics = await enhancer.validate_with_lsp("test.py", "invalid python code")

        assert len(diagnostics) == 1
        assert diagnostics[0].line == 1
        assert diagnostics[0].severity == DiagnosticSeverity.ERROR
        assert diagnostics[0].message == "Syntax error"

    @pytest.mark.asyncio
    async def test_write_with_lsp_dry_run(self):
        """Test dry-run mode (validate and format without writing)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enhancer = LSPWriteEnhancer(workspace_root=tmpdir)

            # Mock methods to avoid actual LSP calls
            enhancer.format_with_formatter = AsyncMock(return_value=("formatted content", "black"))
            enhancer.validate_with_lsp = AsyncMock(return_value=[])

            result = await enhancer.write_with_lsp(
                path="test.py",
                content="original content",
                validate=True,
                format_code=True,
                write=False,  # Dry run
            )

            assert result.success is True  # No errors
            assert result.path == "test.py"
            assert result.formatted is True
            assert result.formatter_used == "black"
            assert result.validated is True
            assert result.written_content == "formatted content"
            assert len(result.diagnostics) == 0

            # File should not exist (dry run)
            assert not Path(tmpdir, "test.py").exists()

    @pytest.mark.asyncio
    async def test_write_with_lsp_with_errors(self):
        """Test write with validation errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enhancer = LSPWriteEnhancer(workspace_root=tmpdir)

            # Mock validation to return errors
            enhancer.format_with_formatter = AsyncMock(return_value=("formatted content", "black"))
            enhancer.validate_with_lsp = AsyncMock(
                return_value=[
                    Diagnostic(
                        line=1,
                        column=0,
                        severity=DiagnosticSeverity.ERROR,
                        message="Syntax error",
                    )
                ]
            )

            result = await enhancer.write_with_lsp(
                path="test.py",
                content="invalid code",
                validate=True,
                format_code=True,
                write=True,
            )

            assert result.success is False
            assert result.has_errors is True
            assert result.error == "File has errors - not written"

            # File should not exist (errors prevented write)
            assert not Path(tmpdir, "test.py").exists()

    @pytest.mark.asyncio
    async def test_write_with_lsp_success(self):
        """Test successful write with LSP enhancement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enhancer = LSPWriteEnhancer(workspace_root=tmpdir)

            # Mock methods
            enhancer.format_with_formatter = AsyncMock(return_value=("formatted content", "black"))
            enhancer.validate_with_lsp = AsyncMock(return_value=[])

            # Use full path in tmpdir
            test_file = Path(tmpdir) / "test.py"

            result = await enhancer.write_with_lsp(
                path=str(test_file),
                content="original content",
                validate=True,
                format_code=True,
                write=True,
            )

            assert result.success is True
            assert result.has_errors is False
            assert result.formatted is True
            assert result.validated is True
            assert result.formatter_used == "black"

            # File should exist with formatted content
            assert test_file.exists()
            assert test_file.read_text() == "formatted content"
