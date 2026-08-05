# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for Metric label uniqueness and timestamp validation — Wave N."""

import time

import pytest

from victor.framework.observability.metrics import Metric, MetricLabel, MetricType


class TestMetricLabelValidation:
    """Metric labels must have unique keys."""

    def test_metric_with_duplicate_label_keys_raises(self):
        """Metric with duplicate label keys should raise ValueError."""
        labels = (
            MetricLabel(key="env", value="prod"),
            MetricLabel(key="env", value="staging"),  # Duplicate key
        )

        with pytest.raises(ValueError, match="label keys must be unique"):
            Metric(
                name="test_metric",
                description="Test metric with duplicate labels",
                metric_type=MetricType.GAUGE,
                labels=labels,
            )

    def test_metric_with_unique_label_keys_passes(self):
        """Metric with unique label keys should succeed."""
        labels = (
            MetricLabel(key="env", value="prod"),
            MetricLabel(key="region", value="us-west"),
        )

        metric = Metric(
            name="test_metric",
            description="Test metric with unique labels",
            metric_type=MetricType.GAUGE,
            labels=labels,
        )

        assert len(metric.labels) == 2


class TestMetricTimestampValidation:
    """Metric timestamps should not be too far in the future."""

    def test_metric_timestamp_too_far_in_future_raises(self):
        """Metric with timestamp > 60 seconds in future should raise ValueError."""
        future_timestamp = time.time() + 120  # 2 minutes in future

        with pytest.raises(ValueError, match="timestamp is too far in the future"):
            Metric(
                name="test_metric",
                description="Test metric with future timestamp",
                metric_type=MetricType.GAUGE,
                timestamp=future_timestamp,
            )

    def test_metric_with_future_timestamp_within_tolerance_passes(self):
        """Metric with timestamp within 60 seconds tolerance should pass."""
        within_tolerance = time.time() + 45  # 45 seconds in future

        metric = Metric(
            name="test_metric",
            description="Test metric with acceptable future timestamp",
            metric_type=MetricType.GAUGE,
            timestamp=within_tolerance,
        )

        assert metric.timestamp == within_tolerance

    def test_metric_with_past_timestamp_passes(self):
        """Metric with past timestamp should always pass."""
        past_timestamp = time.time() - 3600  # 1 hour ago

        metric = Metric(
            name="test_metric",
            description="Test metric with past timestamp",
            metric_type=MetricType.COUNTER,
            timestamp=past_timestamp,
        )

        assert metric.timestamp == past_timestamp

    def test_metric_with_current_timestamp_passes(self):
        """Metric with current timestamp should pass."""
        now = time.time()

        metric = Metric(
            name="test_metric",
            description="Test metric with current timestamp",
            metric_type=MetricType.COUNTER,
            timestamp=now,
        )

        assert metric.timestamp == now


class TestMetricBasicValidation:
    """Existing Metric validation should still work."""

    def test_metric_name_empty_raises(self):
        """Metric with empty name should raise ValueError (existing validation)."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Metric(
                name="",
                description="Test metric",
                metric_type=MetricType.GAUGE,
            )

    def test_metric_with_valid_name_passes(self):
        """Metric with valid non-empty name should pass."""
        metric = Metric(
            name="valid_metric",
            description="Test metric",
            metric_type=MetricType.GAUGE,
        )

        assert metric.name == "valid_metric"
