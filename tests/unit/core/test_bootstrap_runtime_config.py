# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for bootstrap runtime policy wiring."""

from unittest.mock import MagicMock, patch

from victor.config.settings import Settings
from victor.core.bootstrap import (
    _configure_extension_loader_runtime,
    _register_embedding_services,
)
from victor.core.container import EmbeddingServiceProtocol, ServiceContainer


class TestBootstrapRuntimeConfig:
    """Tests for bootstrap-time extension-loader runtime settings application."""

    def test_configure_extension_loader_runtime_applies_pressure_and_starts_reporter(
        self,
    ):
        """Enabled reporter config should apply thresholds and start reporter."""
        settings = Settings(
            extension_loader_warn_queue_threshold=10,
            extension_loader_error_queue_threshold=20,
            extension_loader_warn_in_flight_threshold=3,
            extension_loader_error_in_flight_threshold=5,
            extension_loader_pressure_cooldown_seconds=2.5,
            extension_loader_emit_pressure_events=True,
            extension_loader_metrics_reporter_enabled=True,
            extension_loader_metrics_reporter_interval_seconds=7.0,
            extension_loader_metrics_reporter_topic="vertical.extensions.metrics.custom",
            extension_loader_metrics_reporter_reset_after_emit=True,
        )

        with patch(
            "victor.core.verticals.extension_loader.VerticalExtensionLoader.configure_extension_loader_pressure"
        ) as mock_configure:
            with patch(
                "victor.core.verticals.extension_loader.start_extension_loader_metrics_reporter"
            ) as mock_start:
                with patch(
                    "victor.core.verticals.extension_loader.stop_extension_loader_metrics_reporter"
                ) as mock_stop:
                    _configure_extension_loader_runtime(settings)

        mock_configure.assert_called_once_with(
            warn_queue_threshold=10,
            error_queue_threshold=20,
            warn_in_flight_threshold=3,
            error_in_flight_threshold=5,
            cooldown_seconds=2.5,
            emit_events=True,
        )
        mock_start.assert_called_once_with(
            interval_seconds=7.0,
            topic="vertical.extensions.metrics.custom",
            source="BootstrapExtensionLoaderMetricsReporter",
            reset_after_emit=True,
        )
        mock_stop.assert_not_called()

    def test_configure_extension_loader_runtime_stops_reporter_when_disabled(self):
        """Disabled reporter config should stop reporter singleton."""
        settings = Settings(extension_loader_metrics_reporter_enabled=False)

        with patch(
            "victor.core.verticals.extension_loader.VerticalExtensionLoader.configure_extension_loader_pressure"
        ) as mock_configure:
            with patch(
                "victor.core.verticals.extension_loader.start_extension_loader_metrics_reporter"
            ) as mock_start:
                with patch(
                    "victor.core.verticals.extension_loader.stop_extension_loader_metrics_reporter"
                ) as mock_stop:
                    _configure_extension_loader_runtime(settings)

        mock_configure.assert_called_once()
        mock_start.assert_not_called()
        mock_stop.assert_called_once_with(timeout=2.0)

    def test_register_embedding_services_uses_nested_embedding_settings(self):
        """Embedding bootstrap should read the canonical embedding settings group."""
        settings = Settings(embedding={"unified_embedding_model": "nested-bootstrap-model"})
        container = ServiceContainer()

        with patch(
            "victor.storage.embeddings.service.EmbeddingService.get_instance",
            return_value=MagicMock(),
        ) as get_instance:
            _register_embedding_services(container, settings)
            container.get(EmbeddingServiceProtocol)

        get_instance.assert_called_once_with(model_name="nested-bootstrap-model")

    def test_configure_extension_loader_runtime_coerces_invalid_values_to_defaults(
        self,
    ):
        """Non-typed/mock settings values should safely fall back to defaults."""
        settings = MagicMock()
        settings.extension_loader_warn_queue_threshold = MagicMock()
        settings.extension_loader_error_queue_threshold = MagicMock()
        settings.extension_loader_warn_in_flight_threshold = MagicMock()
        settings.extension_loader_error_in_flight_threshold = MagicMock()
        settings.extension_loader_pressure_cooldown_seconds = MagicMock()
        settings.extension_loader_emit_pressure_events = MagicMock()
        settings.extension_loader_metrics_reporter_enabled = MagicMock()
        settings.extension_loader_metrics_reporter_interval_seconds = MagicMock()
        settings.extension_loader_metrics_reporter_topic = MagicMock()
        settings.extension_loader_metrics_reporter_reset_after_emit = MagicMock()

        with patch(
            "victor.core.verticals.extension_loader.VerticalExtensionLoader.configure_extension_loader_pressure"
        ) as mock_configure:
            with patch(
                "victor.core.verticals.extension_loader.start_extension_loader_metrics_reporter"
            ) as mock_start:
                with patch(
                    "victor.core.verticals.extension_loader.stop_extension_loader_metrics_reporter"
                ) as mock_stop:
                    _configure_extension_loader_runtime(settings)

        mock_configure.assert_called_once_with(
            warn_queue_threshold=24,
            error_queue_threshold=32,
            warn_in_flight_threshold=6,
            error_in_flight_threshold=8,
            cooldown_seconds=5.0,
            emit_events=False,
        )
        mock_start.assert_not_called()
        mock_stop.assert_called_once_with(timeout=2.0)
