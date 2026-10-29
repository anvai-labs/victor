"""Unit tests for SecurityFormatter."""

from victor.tools.formatters.security import SecurityFormatter


class TestSecurityFormatter:
    """Test SecurityFormatter."""

    def test_validate_input_valid(self):
        """Test validate_input() with valid data."""
        formatter = SecurityFormatter()

        assert formatter.validate_input({"vulnerabilities": []}) is True
        assert formatter.validate_input({"findings": []}) is True
        assert formatter.validate_input({"issues": []}) is True
        assert formatter.validate_input({"summary": {}}) is True

    def test_validate_input_invalid(self):
        """Test validate_input() with invalid data."""
        formatter = SecurityFormatter()

        assert formatter.validate_input({}) is False
        assert formatter.validate_input(None) is False

    def test_format_no_findings(self):
        """Test formatting with no security findings."""
        formatter = SecurityFormatter()
        data = {
            "vulnerabilities": [],
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[green bold]✓ No security issues found[/]" in result.content
        assert result.summary == "No issues"

    def test_format_critical_severity(self):
        """Test formatting critical severity finding."""
        formatter = SecurityFormatter()
        data = {
            "vulnerabilities": [
                {
                    "severity": "critical",
                    "title": "Remote Code Execution",
                    "cve_id": "CVE-2024-1234",
                    "file": "src/vulnerable.py",
                    "line": 42,
                    "remediation": "Update to version 2.0",
                }
            ],
            "summary": {
                "critical": 1,
                "high": 0,
                "medium": 0,
                "low": 0,
            },
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[red bold]‼[/]" in result.content
        assert "[red bold]CRITICAL[/]" in result.content
        assert "Remote Code Execution" in result.content
        assert "CVE-2024-1234" in result.content
        assert "src/vulnerable.py:42" in result.content

    def test_format_high_severity(self):
        """Test formatting high severity finding."""
        formatter = SecurityFormatter()
        data = {
            "vulnerabilities": [
                {
                    "severity": "high",
                    "title": "SQL Injection",
                    "cve_id": "CVE-2024-5678",
                    "file": "src/database.py",
                }
            ],
            "summary": {
                "critical": 0,
                "high": 1,
                "medium": 0,
                "low": 0,
            },
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[orange1 bold]⚠[/]" in result.content
        assert "[orange1 bold]HIGH[/]" in result.content

    def test_format_medium_severity(self):
        """Test formatting medium severity finding."""
        formatter = SecurityFormatter()
        data = {
            "vulnerabilities": [
                {
                    "severity": "medium",
                    "title": "Cross-Site Scripting",
                    "file": "src/web.py",
                }
            ],
            "summary": {
                "critical": 0,
                "high": 0,
                "medium": 1,
                "low": 0,
            },
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[yellow bold]⚡[/]" in result.content
        assert "[yellow bold]MEDIUM[/]" in result.content

    def test_format_low_severity(self):
        """Test formatting low severity finding."""
        formatter = SecurityFormatter()
        data = {
            "vulnerabilities": [
                {
                    "severity": "low",
                    "title": "Information Disclosure",
                    "file": "src/utils.py",
                }
            ],
            "summary": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 1,
            },
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[blue bold]⚐[/]" in result.content
        assert "[blue bold]LOW[/]" in result.content

    def test_format_unknown_severity(self):
        """Test formatting unknown severity finding."""
        formatter = SecurityFormatter()
        data = {
            "vulnerabilities": [
                {
                    "severity": "unknown",
                    "title": "Unknown issue",
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[white bold]•[/]" in result.content
        assert "[white bold]UNKNOWN[/]" in result.content

    def test_format_summary_breakdown(self):
        """Test formatting severity breakdown summary."""
        formatter = SecurityFormatter()
        data = {
            "vulnerabilities": [
                {
                    "severity": "critical",
                    "title": "Critical issue",
                },
                {
                    "severity": "high",
                    "title": "High issue",
                },
            ],
            "summary": {
                "critical": 1,
                "high": 1,
                "medium": 0,
                "low": 0,
            },
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        # Summary breakdown shown when there are findings
        assert "Security Scan Results:" in result.content
        assert "[red bold]1 critical[/]" in result.content
        assert "[orange1]1 high[/]" in result.content  # Summary line doesn't have bold on the count
        assert "[orange1 bold]HIGH[/]" in result.content  # But the detailed finding does

    def test_format_with_findings(self):
        """Test formatting with actual findings."""
        formatter = SecurityFormatter()
        data = {
            "vulnerabilities": [
                {
                    "severity": "high",
                    "title": "Buffer Overflow",
                    "cve_id": "CVE-2024-9999",
                    "file": "src/buffer.c",
                    "line": 100,
                    "remediation": "Add bounds checking",
                }
            ],
            "summary": {
                "critical": 0,
                "high": 1,
                "medium": 0,
                "low": 0,
            },
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "Findings:" in result.content
        assert "Buffer Overflow" in result.content
        assert result.summary == "1 findings"

    def test_format_max_findings(self):
        """Test max_findings parameter limits output."""
        formatter = SecurityFormatter()
        data = {
            "vulnerabilities": [{"severity": "low", "title": f"Issue {i}"} for i in range(25)],
            "summary": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 25,
            },
        }

        result = formatter.format(data, max_findings=20)

        assert result.contains_markup is True
        assert "... and 5 more findings" in result.content

    def test_format_missing_cve_id(self):
        """Test formatting finding without CVE ID."""
        formatter = SecurityFormatter()
        data = {
            "vulnerabilities": [
                {
                    "severity": "medium",
                    "title": "Missing Validation",
                    "file": "src/input.py",
                    # Missing cve_id
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "Missing Validation" in result.content

    def test_format_missing_file_path(self):
        """Test formatting finding without file path."""
        formatter = SecurityFormatter()
        data = {
            "vulnerabilities": [
                {
                    "severity": "low",
                    "title": "General Issue",
                    # Missing file and line
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "General Issue" in result.content

    def test_format_with_remediation(self):
        """Test formatting with remediation suggestion."""
        formatter = SecurityFormatter()
        data = {
            "vulnerabilities": [
                {
                    "severity": "high",
                    "title": "Outdated Library",
                    "remediation": "Update to version 3.0.0 or later",
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "Fix:" in result.content
        assert "Update to version 3.0.0" in result.content

    def test_format_alternate_keys(self):
        """Test that alternate keys (findings, issues) work."""
        formatter = SecurityFormatter()

        # Test 'findings' key
        result1 = formatter.format({"findings": [{"severity": "low", "title": "Test"}]})
        assert result1.contains_markup is True

        # Test 'issues' key
        result2 = formatter.format({"issues": [{"severity": "low", "title": "Test"}]})
        assert result2.contains_markup is True

    def test_format_missing_optional_fields(self):
        """Test formatting with all optional fields missing."""
        formatter = SecurityFormatter()
        data = {
            "vulnerabilities": [
                {
                    "severity": "medium",
                    # Missing title, cve_id, file, line, remediation
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[yellow bold]MEDIUM[/]" in result.content
