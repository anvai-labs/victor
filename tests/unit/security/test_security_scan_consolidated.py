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

"""Tests for consolidated security_scan tool."""

import pytest
from pathlib import Path
import tempfile

from victor.tools.security_scanner_tool import scan


@pytest.mark.asyncio
async def test_security_scan_secrets():
    """Test secrets scanning functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file with a secret
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text(
            'api_key = "sk-1234567890abcdef1234567890abcdef"\n' 'password = "SuperSecret123"\n'
        )

        # Run secrets scan
        result = await scan(path=tmpdir, scan_types=["secrets"], file_pattern="*.py")

        assert result["success"] is True
        assert "secrets" in result["results"]
        assert result["results"]["secrets"]["count"] >= 2
        assert result["total_issues"] >= 2
        assert "formatted_report" in result


@pytest.mark.asyncio
async def test_security_scan_config():
    """Test configuration scanning functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file with config issues
        test_file = Path(tmpdir) / "config.py"
        test_file.write_text(
            "DEBUG = True\n" 'API_URL = "http://insecure-api.com"\n' 'SERVER_IP = "192.168.1.100"\n'
        )

        # Run config scan
        result = await scan(path=tmpdir, scan_types=["config"], file_pattern="*.py")

        assert result["success"] is True
        assert "config" in result["results"]
        assert result["results"]["config"]["count"] >= 2
        assert "formatted_report" in result


@pytest.mark.asyncio
async def test_security_scan_dependencies():
    """Test dependency scanning functionality.

    Note: Dependency scanning requires dependency_scan=True and pip-audit installed.
    Without pip-audit, it will return an error in the dependencies result.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a requirements file with vulnerable packages
        req_file = Path(tmpdir) / "requirements.txt"
        req_file.write_text("django==2.2.0\n" "flask==1.0.0\n" "requests==2.25.0\n")

        # Run dependency scan with dependency_scan=True (requires pip-audit)
        result = await scan(
            path=tmpdir,
            scan_types=["dependencies"],
            requirements_file=str(req_file),
            dependency_scan=True,
        )

        assert result["success"] is True
        assert "dependencies" in result["results"]
        # May have error if pip-audit not installed, or packages_checked if it is
        assert (
            "error" in result["results"]["dependencies"]
            or "packages_checked" in result["results"]["dependencies"]
        )
        assert "formatted_report" in result


@pytest.mark.asyncio
async def test_security_scan_all():
    """Test comprehensive security scan.

    Note: 'all' expands to secrets, dependencies, config but dependency scanning
    requires dependency_scan=True flag to actually run.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file with multiple issues
        test_file = Path(tmpdir) / "app.py"
        test_file.write_text(
            "DEBUG = True\n"
            'api_key = "sk-1234567890abcdef1234567890abcdef"\n'
            'password = "SuperSecret123"\n'
            'API_URL = "http://insecure-api.com"\n'
        )

        # Create requirements file
        req_file = Path(tmpdir) / "requirements.txt"
        req_file.write_text("django==2.2.0\n")

        # Run all scans (dependency_scan=True to include dependency scanning)
        result = await scan(
            path=tmpdir,
            scan_types=["all"],
            requirements_file=str(req_file),
            dependency_scan=True,
        )

        assert result["success"] is True
        assert "secrets" in result["results"]
        assert "config" in result["results"]
        assert "dependencies" in result["results"]
        # At least 4 issues from secrets and config (api_key, password, DEBUG, http URL)
        assert result["total_issues"] >= 4
        assert "issues_by_severity" in result
        assert "formatted_report" in result


@pytest.mark.asyncio
async def test_security_scan_severity_threshold():
    """Test severity threshold filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file with mixed severity issues
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text(
            'api_key = "sk-1234567890abcdef1234567890abcdef"\n'  # high
            "DEBUG = True\n"  # medium
            'SERVER_IP = "192.168.1.100"\n'  # medium
        )

        # Scan with high threshold
        result = await scan(path=tmpdir, scan_types=["all"], severity_threshold="high")

        assert result["success"] is True
        # Should only report high severity issues
        assert result["issues_by_severity"]["high"] >= 1


@pytest.mark.asyncio
async def test_security_scan_single_file():
    """Test scanning a single file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text('api_key = "sk-1234567890abcdef1234567890abcdef"\n')

        # Scan single file
        result = await scan(path=str(test_file), scan_types=["secrets"])

        assert result["success"] is True
        assert result["results"]["secrets"]["files_scanned"] == 1


@pytest.mark.asyncio
async def test_security_scan_nonexistent_path():
    """Test scanning nonexistent path."""
    result = await scan(path="/nonexistent/path", scan_types=["all"])

    assert result["success"] is False
    assert "error" in result
    assert "not found" in result["error"].lower()


@pytest.mark.asyncio
async def test_security_scan_no_issues():
    """Test scanning clean code with no issues."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create clean test file
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text("def hello():\n" '    return "Hello, World!"\n')

        # Run all scans
        result = await scan(path=tmpdir, scan_types=["secrets", "config"])

        assert result["success"] is True
        assert result["total_issues"] == 0
        assert "No critical security issues" in result["formatted_report"]
