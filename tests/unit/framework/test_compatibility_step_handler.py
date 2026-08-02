# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for CompatibilityStepHandler — version gating in pipeline."""

from unittest.mock import MagicMock, patch
from typing import Any, Dict

import pytest


class FakeContext:
    """Minimal VerticalContext stub."""

    def __init__(self):
        self.capability_configs: Dict[str, Any] = {}

    def set_capability_config(self, name: str, config: Any) -> None:
        self.capability_configs[name] = config


class FakeResult:
    """Minimal IntegrationResult stub."""

    def __init__(self):
        self.warnings: list = []
        self.errors: list = []

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)


class TestCompatibilityStepHandler:
    def _make_handler(self):
        from victor.framework.step_handlers import CompatibilityStepHandler

        return CompatibilityStepHandler()

    def test_handler_name_and_order(self):
        handler = self._make_handler()
        assert handler.name == "compatibility"
        assert handler.order == 1

    def test_compatible_vertical_passes(self):
        handler = self._make_handler()
        context = FakeContext()
        result = FakeResult()

        mock_report = MagicMock()
        mock_report.compatible = True
        mock_report.warnings = []
        mock_report.to_dict.return_value = {"compatible": True}
        mock_report.vertical_name = "test"
        mock_report.vertical_version = "1.0.0"
        mock_report.framework_version = "0.7.0"

        mock_gate = MagicMock()
        mock_gate.assess_manifest.return_value = mock_report

        with patch(
            "victor.framework.step_handlers.CompatibilityStepHandler._do_apply"
        ) as mock_apply:
            # Call directly to test the internal logic
            pass

        # Test via direct call with mocked gate
        with (
            patch("victor.core.verticals.compatibility_gate.VerticalCompatibilityGate") as MockGate,
            patch(
                "victor.core.verticals.manifest_contract.get_or_create_vertical_manifest"
            ) as mock_manifest,
        ):
            MockGate.return_value = mock_gate
            mock_manifest.return_value = MagicMock()

            handler._do_apply(MagicMock(), MagicMock(), context, result)

        assert "compatibility_report" in context.capability_configs
        assert len(result.warnings) == 0

    def test_incompatible_vertical_raises(self):
        from victor.core.verticals.compatibility_gate import (
            VerticalCompatibilityError,
            VerticalCompatibilityReport,
        )

        handler = self._make_handler()
        context = FakeContext()
        result = FakeResult()

        report = VerticalCompatibilityReport(
            vertical_name="old-vertical",
            vertical_version="0.1.0",
            framework_version="0.7.0",
            errors=["Incompatible framework version: 0.7.0 does not meet >=0.9.0"],
        )

        mock_gate = MagicMock()
        mock_gate.assess_manifest.return_value = report

        with (
            patch("victor.core.verticals.compatibility_gate.VerticalCompatibilityGate") as MockGate,
            patch(
                "victor.core.verticals.manifest_contract.get_or_create_vertical_manifest"
            ) as mock_manifest,
        ):
            MockGate.return_value = mock_gate
            mock_manifest.return_value = MagicMock()

            with pytest.raises(VerticalCompatibilityError) as exc_info:
                handler._do_apply(MagicMock(), MagicMock(), context, result)

            assert "old-vertical" in str(exc_info.value)
            assert "0.7.0" in str(exc_info.value)

    def test_apply_marks_incompatible_vertical_as_error_in_non_strict_mode(self):
        from victor.core.verticals.compatibility_gate import VerticalCompatibilityReport

        handler = self._make_handler()
        context = FakeContext()

        from victor.framework.vertical_integration import IntegrationResult

        result = IntegrationResult()
        report = VerticalCompatibilityReport(
            vertical_name="old-vertical",
            vertical_version="0.1.0",
            framework_version="0.7.0",
            errors=["min_framework_version >=0.8.0 not met"],
        )

        mock_gate = MagicMock()
        mock_gate.assess_manifest.return_value = report

        with (
            patch("victor.core.verticals.compatibility_gate.VerticalCompatibilityGate") as MockGate,
            patch(
                "victor.core.verticals.manifest_contract.get_or_create_vertical_manifest"
            ) as mock_manifest,
        ):
            MockGate.return_value = mock_gate
            mock_manifest.return_value = MagicMock()

            handler.apply(MagicMock(), MagicMock(), context, result, strict_mode=False)

        assert result.success is False
        assert result.validation_status == "failed"
        assert result.errors
        assert result.warnings == []
        assert result.step_status["compatibility"]["status"] == "error"

    def test_warnings_recorded_but_dont_block(self):
        handler = self._make_handler()
        context = FakeContext()
        result = FakeResult()

        mock_report = MagicMock()
        mock_report.compatible = True
        mock_report.warnings = ["SDK version mismatch (minor)"]
        mock_report.to_dict.return_value = {
            "compatible": True,
            "warnings": ["SDK version mismatch"],
        }
        mock_report.vertical_name = "test"
        mock_report.vertical_version = "1.0.0"
        mock_report.framework_version = "0.7.0"

        mock_gate = MagicMock()
        mock_gate.assess_manifest.return_value = mock_report

        with (
            patch("victor.core.verticals.compatibility_gate.VerticalCompatibilityGate") as MockGate,
            patch(
                "victor.core.verticals.manifest_contract.get_or_create_vertical_manifest"
            ) as mock_manifest,
        ):
            MockGate.return_value = mock_gate
            mock_manifest.return_value = MagicMock()

            handler._do_apply(MagicMock(), MagicMock(), context, result)

        assert len(result.warnings) == 1
        assert "SDK version mismatch" in result.warnings[0]

    def test_no_manifest_skips_gracefully(self):
        handler = self._make_handler()
        context = FakeContext()
        result = FakeResult()

        with patch(
            "victor.core.verticals.manifest_contract.get_or_create_vertical_manifest",
            side_effect=ImportError("no manifest"),
        ):
            handler._do_apply(MagicMock(), MagicMock(), context, result)

        assert "compatibility_report" not in context.capability_configs
        assert len(result.warnings) == 0


class TestVerticalCompatibilityError:
    def test_error_message_format(self):
        from victor.core.verticals.compatibility_gate import (
            VerticalCompatibilityError,
            VerticalCompatibilityReport,
        )

        report = VerticalCompatibilityReport(
            vertical_name="victor-coding",
            vertical_version="0.5.0",
            framework_version="0.7.0",
            errors=["min_framework_version >=0.8.0 not met"],
        )

        err = VerticalCompatibilityError(report)
        assert "victor-coding" in str(err)
        assert "0.5.0" in str(err)
        assert "0.7.0" in str(err)
        assert "min_framework_version" in str(err)
        assert isinstance(err, ValueError)  # backward compat

    def test_report_attached_to_error(self):
        from victor.core.verticals.compatibility_gate import (
            VerticalCompatibilityError,
            VerticalCompatibilityReport,
        )

        report = VerticalCompatibilityReport(
            vertical_name="test",
            vertical_version="1.0",
            framework_version="0.7.0",
            errors=["bad version"],
        )
        err = VerticalCompatibilityError(report)
        assert err.report is report


class TestCompatibilityInRegistry:
    def test_compatibility_handler_registered_in_default(self):
        from victor.framework.step_handlers import StepHandlerRegistry

        registry = StepHandlerRegistry.default()
        handlers = registry.get_ordered_handlers()
        handler_names = [h.name for h in handlers]
        assert "compatibility" in handler_names

    def test_compatibility_runs_first(self):
        from victor.framework.step_handlers import StepHandlerRegistry

        registry = StepHandlerRegistry.default()
        handlers = registry.get_ordered_handlers()
        assert handlers[0].name == "compatibility"

    def test_pipeline_stops_after_compatibility_failure_in_non_strict_mode(self):
        from victor.core.verticals.compatibility_gate import VerticalCompatibilityReport
        from victor.framework.step_handlers import CompatibilityStepHandler
        from victor.framework.vertical_integration import (
            IntegrationResult,
            VerticalIntegrationPipeline,
        )

        class _NextHandler:
            name = "next"
            order = 10

            def __init__(self) -> None:
                self.called = False

            def apply(self, *args, **kwargs) -> None:
                self.called = True

        class _Registry:
            def __init__(self, next_handler):
                self.next_handler = next_handler

            def get_ordered_handlers(self):
                return [CompatibilityStepHandler(), self.next_handler]

        class _Vertical:
            name = "incompatible"
            version = "0.1.0"

        next_handler = _NextHandler()
        pipeline = VerticalIntegrationPipeline(
            strict_mode=False,
            step_registry=_Registry(next_handler),
        )
        context = FakeContext()
        result = IntegrationResult()
        report = VerticalCompatibilityReport(
            vertical_name="incompatible",
            vertical_version="0.1.0",
            framework_version="0.7.0",
            errors=["api_version unsupported"],
        )
        mock_gate = MagicMock()
        mock_gate.assess_manifest.return_value = report

        with (
            patch("victor.core.verticals.compatibility_gate.VerticalCompatibilityGate") as MockGate,
            patch(
                "victor.core.verticals.manifest_contract.get_or_create_vertical_manifest"
            ) as mock_manifest,
        ):
            MockGate.return_value = mock_gate
            mock_manifest.return_value = MagicMock()

            pipeline._apply_with_step_handlers(MagicMock(), _Vertical, context, result)

        assert result.success is False
        assert next_handler.called is False
