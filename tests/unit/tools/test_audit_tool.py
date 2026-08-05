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

"""Tests for Audit Tool."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from datetime import datetime

import victor.tools.audit_tool as audit_tool_module
from victor.tools.audit_tool import AuditTool
from victor.agent.presentation import NullPresentationAdapter
from victor.tools.base import CostTier
from victor.security.audit import (
    AuditEvent,
    AuditEventType,
    AuditReport,
    ComplianceFramework,
    ComplianceRule,
    ComplianceViolation,
    Severity,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def use_null_presentation_adapter():
    """Use NullPresentationAdapter for predictable test output."""
    audit_tool_module._presentation = NullPresentationAdapter()
    yield
    audit_tool_module._presentation = None


@pytest.fixture
def tool():
    """Create audit tool instance."""
    return AuditTool()


@pytest.fixture
def sample_event():
    """Create sample audit event."""
    return AuditEvent(
        event_id="evt_001",
        event_type=AuditEventType.FILE_READ,
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
        severity=Severity.INFO,
        actor="system",
        action="Read file src/main.py",
        resource="src/main.py",
    )


@pytest.fixture
def sample_rule():
    """Create sample compliance rule."""
    return ComplianceRule(
        rule_id="soc2_001",
        framework=ComplianceFramework.SOC2,
        name="Access Logging",
        description="All file access must be logged",
        event_types=[AuditEventType.FILE_READ, AuditEventType.FILE_WRITE],
    )


@pytest.fixture
def sample_violation(sample_rule, sample_event):
    """Create sample compliance violation."""
    return ComplianceViolation(
        violation_id="viol_001",
        rule=sample_rule,
        event=sample_event,
        violation_type="missing_field",
        message="Missing required field: ip_address",
    )


@pytest.fixture
def sample_report(sample_violation):
    """Create sample audit report."""
    return AuditReport(
        report_id="rpt_001",
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 31),
        framework=ComplianceFramework.SOC2,
        total_events=150,
        events_by_type={"file_read": 100, "tool_execution": 50},
        events_by_severity={"info": 120, "warning": 25, "error": 5},
        violations=[sample_violation],
    )


@pytest.fixture
def sample_summary():
    """Create sample audit summary."""
    return {
        "period_days": 7,
        "total_events": 250,
        "compliance_status": "compliant",
        "violations": 0,
        "events_by_type": {
            "file_read": 100,
            "tool_execution": 80,
            "file_write": 50,
            "code_generation": 20,
        },
        "events_by_severity": {
            "info": 200,
            "warning": 40,
            "error": 10,
        },
    }


# =============================================================================
# TOOL PROPERTIES TESTS
# =============================================================================


class TestAuditToolProperties:
    """Tests for tool properties and metadata."""

    def test_tool_name(self, tool):
        """Test tool name."""
        assert tool.name == "audit"

    def test_tool_description_contains_frameworks(self, tool):
        """Test description mentions compliance frameworks."""
        assert "SOC 2" in tool.description
        assert "GDPR" in tool.description
        assert "HIPAA" in tool.description
        assert "PCI DSS" in tool.description
        assert "ISO 27001" in tool.description

    def test_cost_tier(self, tool):
        """Test cost tier is FREE."""
        assert tool.cost_tier == CostTier.FREE

    def test_parameters_schema(self, tool):
        """Test parameters schema structure."""
        assert tool.parameters["type"] == "object"
        assert "action" in tool.parameters["properties"]
        assert "framework" in tool.parameters["properties"]
        assert "days" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["action"]

    def test_action_enum_values(self, tool):
        """Test action enum has all expected values."""
        actions = tool.parameters["properties"]["action"]["enum"]
        assert "summary" in actions
        assert "report" in actions
        assert "query" in actions
        assert "compliance" in actions
        assert "export" in actions

    def test_framework_enum_values(self, tool):
        """Test framework enum has expected values."""
        frameworks = tool.parameters["properties"]["framework"]["enum"]
        assert "soc2" in frameworks
        assert "gdpr" in frameworks
        assert "hipaa" in frameworks
        assert "pci_dss" in frameworks
        assert "iso_27001" in frameworks

    def test_metadata_category(self, tool):
        """Test metadata category."""
        assert tool.metadata.category == "audit"

    def test_metadata_keywords(self, tool):
        """Test metadata keywords."""
        keywords = tool.metadata.keywords
        assert "audit" in keywords
        assert "compliance" in keywords
        assert "soc2" in keywords


# =============================================================================
# SUMMARY ACTION TESTS
# =============================================================================


class TestSummaryAction:
    """Tests for the summary action."""

    @pytest.mark.asyncio
    async def test_summary_basic(self, tool, sample_summary):
        """Test basic summary generation."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.get_summary.return_value = sample_summary
            MockManager.get_instance.return_value = mock_instance

            result = await tool.execute(action="summary")

            assert result.success is True
            assert "Audit Activity Summary" in result.output
            assert "Total Events:" in result.output
            mock_instance.get_summary.assert_called_once_with(days=7)

    @pytest.mark.asyncio
    async def test_summary_custom_days(self, tool, sample_summary):
        """Test summary with custom days."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.get_summary.return_value = sample_summary
            MockManager.get_instance.return_value = mock_instance

            await tool.execute(action="summary", days=30)

            mock_instance.get_summary.assert_called_once_with(days=30)

    @pytest.mark.asyncio
    async def test_summary_with_violations(self, tool):
        """Test summary output with violations."""
        summary = {
            "period_days": 7,
            "total_events": 100,
            "compliance_status": "non_compliant",
            "violations": 5,
            "events_by_type": {},
            "events_by_severity": {},
        }

        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.get_summary.return_value = summary
            MockManager.get_instance.return_value = mock_instance

            result = await tool.execute(action="summary")

            assert result.success is True
            assert "!" in result.output  # warning icon
            assert "**Violations:** 5" in result.output


# =============================================================================
# REPORT ACTION TESTS
# =============================================================================


class TestReportAction:
    """Tests for the report action."""

    @pytest.mark.asyncio
    async def test_report_basic(self, tool, sample_report):
        """Test basic report generation."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.generate_report.return_value = sample_report
            MockManager.get_instance.return_value = mock_instance

            result = await tool.execute(action="report")

            assert result.success is True
            assert "Audit Report" in result.output
            mock_instance.generate_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_report_with_framework(self, tool, sample_report):
        """Test report with specific framework."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.generate_report.return_value = sample_report
            MockManager.get_instance.return_value = mock_instance

            result = await tool.execute(action="report", framework="soc2")

            assert result.success is True
            call_args = mock_instance.generate_report.call_args
            assert call_args[1]["framework"] == ComplianceFramework.SOC2

    @pytest.mark.asyncio
    async def test_report_includes_violations(self, tool, sample_report):
        """Test report includes violations in output."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.generate_report.return_value = sample_report
            MockManager.get_instance.return_value = mock_instance

            result = await tool.execute(action="report")

            assert "Violations" in result.output
            assert "!" in result.output  # warning icon

    @pytest.mark.asyncio
    async def test_report_no_violations(self, tool):
        """Test report with no violations."""
        report = AuditReport(
            report_id="rpt_001",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
            framework=None,
            total_events=100,
            violations=[],
        )

        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.generate_report.return_value = report
            MockManager.get_instance.return_value = mock_instance

            result = await tool.execute(action="report")

            assert "+ None" in result.output  # success icon + None


# =============================================================================
# QUERY ACTION TESTS
# =============================================================================


class TestQueryAction:
    """Tests for the query action."""

    @pytest.mark.asyncio
    async def test_query_basic(self, tool, sample_event):
        """Test basic event query."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.query_events.return_value = [sample_event]
            MockManager.get_instance.return_value = mock_instance

            result = await tool.execute(action="query")

            assert result.success is True
            assert "Audit Events" in result.output
            assert result.metadata["count"] == 1

    @pytest.mark.asyncio
    async def test_query_no_events(self, tool):
        """Test query with no results."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.query_events.return_value = []
            MockManager.get_instance.return_value = mock_instance

            result = await tool.execute(action="query")

            assert result.success is True
            assert "No audit events found" in result.output

    @pytest.mark.asyncio
    async def test_query_with_event_type(self, tool, sample_event):
        """Test query with event type filter."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.query_events.return_value = [sample_event]
            MockManager.get_instance.return_value = mock_instance

            result = await tool.execute(action="query", event_type="file_read")

            assert result.success is True
            call_args = mock_instance.query_events.call_args
            assert call_args[1]["event_types"] == [AuditEventType.FILE_READ]

    @pytest.mark.asyncio
    async def test_query_invalid_event_type(self, tool):
        """Test query with invalid event type."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            MockManager.get_instance.return_value = AsyncMock()

            result = await tool.execute(action="query", event_type="invalid_type")

            assert result.success is False
            assert "Invalid event type" in result.output

    @pytest.mark.asyncio
    async def test_query_with_limit(self, tool, sample_event):
        """Test query with custom limit."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.query_events.return_value = [sample_event]
            MockManager.get_instance.return_value = mock_instance

            await tool.execute(action="query", limit=100)

            call_args = mock_instance.query_events.call_args
            assert call_args[1]["limit"] == 100


# =============================================================================
# COMPLIANCE ACTION TESTS
# =============================================================================


class TestComplianceAction:
    """Tests for the compliance action."""

    @pytest.mark.asyncio
    async def test_compliance_compliant(self, tool):
        """Test compliance check when compliant."""
        status = {
            "framework": "soc2",
            "compliant": True,
            "total_events": 1000,
            "report_period_days": 30,
            "violations": 0,
            "violation_details": [],
        }

        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.check_compliance.return_value = status
            MockManager.get_instance.return_value = mock_instance

            result = await tool.execute(action="compliance", framework="soc2")

            assert result.success is True
            assert "+" in result.output  # success icon
            assert "Compliant" in result.output

    @pytest.mark.asyncio
    async def test_compliance_non_compliant(self, tool):
        """Test compliance check when non-compliant."""
        status = {
            "framework": "hipaa",
            "compliant": False,
            "total_events": 500,
            "report_period_days": 30,
            "violations": 3,
            "violation_details": [
                {
                    "violation_type": "missing_encryption",
                    "message": "PHI not encrypted",
                },
            ],
        }

        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.check_compliance.return_value = status
            MockManager.get_instance.return_value = mock_instance

            result = await tool.execute(action="compliance", framework="hipaa")

            assert result.success is True
            assert "x" in result.output  # error icon
            assert "Non-Compliant" in result.output
            assert "Violations" in result.output

    @pytest.mark.asyncio
    async def test_compliance_missing_framework(self, tool):
        """Test compliance check without framework."""
        result = await tool.execute(action="compliance")

        assert result.success is False
        assert "framework is required" in result.output


# =============================================================================
# EXPORT ACTION TESTS
# =============================================================================


class TestExportAction:
    """Tests for the export action."""

    @pytest.mark.asyncio
    async def test_export_json(self, tool):
        """Test export in JSON format."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.export_audit_log.return_value = Path("/tmp/audit_log.json")
            MockManager.get_instance.return_value = mock_instance

            result = await tool.execute(action="export")

            assert result.success is True
            assert "Exported" in result.output
            assert result.metadata["path"] == "/tmp/audit_log.json"
            call_args = mock_instance.export_audit_log.call_args
            assert call_args[1]["format"] == "json"

    @pytest.mark.asyncio
    async def test_export_csv(self, tool):
        """Test export in CSV format."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.export_audit_log.return_value = Path("/tmp/audit_log.csv")
            MockManager.get_instance.return_value = mock_instance

            result = await tool.execute(action="export", format="csv")

            assert result.success is True
            call_args = mock_instance.export_audit_log.call_args
            assert call_args[1]["format"] == "csv"

    @pytest.mark.asyncio
    async def test_export_custom_days(self, tool):
        """Test export with custom days."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.export_audit_log.return_value = Path("/tmp/audit.json")
            MockManager.get_instance.return_value = mock_instance

            await tool.execute(action="export", days=90)

            call_args = mock_instance.export_audit_log.call_args
            assert call_args[1]["days"] == 90


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        """Test handling unknown action."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            MockManager.get_instance.return_value = AsyncMock()

            result = await tool.execute(action="invalid_action")

            assert result.success is False
            assert "Unknown action" in result.output

    @pytest.mark.asyncio
    async def test_exception_handling(self, tool):
        """Test exception handling in execute."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.get_summary.side_effect = Exception("Database error")
            MockManager.get_instance.return_value = mock_instance

            result = await tool.execute(action="summary")

            assert result.success is False
            assert "Audit operation failed" in result.output
            assert "Database error" in result.error


# =============================================================================
# FORMATTING TESTS
# =============================================================================


class TestFormatMethods:
    """Tests for formatting helper methods."""

    def test_format_summary_basic(self, tool, sample_summary):
        """Test formatting basic summary."""
        output = tool._format_summary(sample_summary)

        assert "Audit Activity Summary" in output
        assert "Last 7 days" in output
        assert "250" in output
        assert "+" in output  # compliant status (text alternative for success)

    def test_format_summary_with_violations(self, tool):
        """Test formatting summary with violations."""
        summary = {
            "period_days": 7,
            "total_events": 100,
            "compliance_status": "non_compliant",
            "violations": 5,
            "events_by_type": {"file_read": 50},
            "events_by_severity": {"info": 80, "warning": 15, "error": 5},
        }
        output = tool._format_summary(summary)

        assert "!" in output  # warning icon (text alternative)
        assert "**Violations:** 5" in output

    def test_format_summary_with_severity(self, tool, sample_summary):
        """Test formatting summary includes severity breakdown."""
        output = tool._format_summary(sample_summary)

        assert "Events by Severity" in output
        assert "info" in output

    def test_format_report_with_framework(self, tool, sample_report):
        """Test formatting report with framework."""
        output = tool._format_report(sample_report)

        assert "Audit Report" in output
        assert "SOC2" in output
        assert "150" in output

    def test_format_report_many_violations(self, tool, sample_rule, sample_event):
        """Test formatting report with many violations."""
        violations = [
            ComplianceViolation(
                violation_id=f"viol_{i}",
                rule=sample_rule,
                event=sample_event,
                violation_type="test",
                message=f"Violation {i}",
            )
            for i in range(10)
        ]
        report = AuditReport(
            report_id="rpt_001",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
            framework=None,
            total_events=100,
            violations=violations,
        )
        output = tool._format_report(report)

        assert "... and 5 more" in output

    def test_format_events_basic(self, tool, sample_event):
        """Test formatting events."""
        output = tool._format_events([sample_event])

        assert "Audit Events" in output
        assert "file_read" in output
        assert "[I]" in output  # info severity (text alternative for level_info)

    def test_format_events_empty(self, tool):
        """Test formatting empty events list."""
        output = tool._format_events([])
        assert "No audit events found" in output

    def test_format_events_many(self, tool):
        """Test formatting many events truncates."""
        events = [
            AuditEvent(
                event_id=f"evt_{i}",
                event_type=AuditEventType.FILE_READ,
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                severity=Severity.INFO,
                actor="system",
                action=f"Action {i}",
            )
            for i in range(30)
        ]
        output = tool._format_events(events)

        assert "... and 10 more" in output

    def test_format_compliance_compliant(self, tool):
        """Test formatting compliance status when compliant."""
        status = {
            "framework": "gdpr",
            "compliant": True,
            "total_events": 500,
            "report_period_days": 30,
            "violations": 0,
            "violation_details": [],
        }
        output = tool._format_compliance(status)

        assert "+" in output  # success icon
        assert "Compliant" in output
        assert "GDPR" in output

    def test_format_compliance_non_compliant(self, tool):
        """Test formatting compliance status when non-compliant."""
        status = {
            "framework": "hipaa",
            "compliant": False,
            "total_events": 500,
            "report_period_days": 30,
            "violations": 2,
            "violation_details": [
                {"violation_type": "encryption", "message": "Data not encrypted"},
            ],
        }
        output = tool._format_compliance(status)

        assert "x" in output  # error icon
        assert "Non-Compliant" in output
        assert "Violations" in output
        assert "encryption" in output


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestIntegrationStyle:
    """Integration-style tests combining multiple aspects."""

    @pytest.mark.asyncio
    async def test_full_audit_workflow(self, tool, sample_summary, sample_report):
        """Test a full audit workflow."""
        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.get_summary.return_value = sample_summary
            mock_instance.generate_report.return_value = sample_report
            mock_instance.check_compliance.return_value = {
                "framework": "soc2",
                "compliant": True,
                "total_events": 1000,
                "report_period_days": 30,
                "violations": 0,
                "violation_details": [],
            }
            MockManager.get_instance.return_value = mock_instance

            # Step 1: Get summary
            summary_result = await tool.execute(action="summary")
            assert summary_result.success is True

            # Step 2: Generate report
            report_result = await tool.execute(action="report", framework="soc2")
            assert report_result.success is True
            assert report_result.metadata is not None

            # Step 3: Check compliance
            compliance_result = await tool.execute(action="compliance", framework="soc2")
            assert compliance_result.success is True

    @pytest.mark.asyncio
    async def test_gdpr_compliance_check(self, tool):
        """Test GDPR compliance check workflow."""
        status = {
            "framework": "gdpr",
            "compliant": True,
            "total_events": 2000,
            "report_period_days": 30,
            "violations": 0,
            "violation_details": [],
        }

        with patch("victor.tools.audit_tool.AuditManager") as MockManager:
            mock_instance = AsyncMock()
            mock_instance.check_compliance.return_value = status
            MockManager.get_instance.return_value = mock_instance

            result = await tool.execute(action="compliance", framework="gdpr")

            assert result.success is True
            assert "GDPR" in result.output
            call_args = mock_instance.check_compliance.call_args
            assert call_args[0][0] == ComplianceFramework.GDPR
