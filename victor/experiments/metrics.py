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

"""Metrics aggregation and querying for experiment tracking.

This module provides functionality for aggregating metrics across runs,
querying metric history, and computing statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from victor.experiments.storage import IStorageBackend

logger = logging.getLogger(__name__)


@dataclass
class MetricStatistics:
    """Statistical summary for a metric.

    Attributes:
        metric_name: Name of the metric
        count: Number of data points
        mean: Mean value
        min: Minimum value
        max: Maximum value
        last: Last (most recent) value
        std: Standard deviation (optional)
    """

    metric_name: str
    count: int
    mean: float
    min: float
    max: float
    last: float
    std: Optional[float] = None


class MetricsAggregator:
    """Aggregator for computing metric statistics.

    This class provides methods for aggregating metrics across runs
    and computing statistical summaries.

    Args:
        storage: Storage backend for querying metrics

    Example:
        aggregator = MetricsAggregator(storage)
        stats = aggregator.get_metric_statistics(run_id, "accuracy")
        print(f"Mean accuracy: {stats.mean}")
    """

    def __init__(self, storage: IStorageBackend) -> None:
        """Initialize metrics aggregator.

        Args:
            storage: Storage backend
        """
        self._storage = storage

    def get_metric_statistics(self, run_id: str, metric_name: str) -> Optional[MetricStatistics]:
        """Get statistics for a specific metric.

        Args:
            run_id: Run ID
            metric_name: Metric name

        Returns:
            MetricStatistics if metrics found, None otherwise
        """
        metrics = self._storage.get_metric_history(run_id, metric_name)

        if not metrics:
            return None

        values = [m.value for m in metrics]

        # Compute statistics
        count = len(values)
        mean_val = sum(values) / count if count > 0 else 0.0
        min_val = min(values)
        max_val = max(values)
        last_val = values[-1]

        # Compute standard deviation
        std_val = None
        if count > 1:
            variance = sum((x - mean_val) ** 2 for x in values) / (count - 1)
            std_val = variance**0.5

        return MetricStatistics(
            metric_name=metric_name,
            count=count,
            mean=mean_val,
            min=min_val,
            max=max_val,
            last=last_val,
            std=std_val,
        )

    def aggregate_metrics_across_runs(
        self, run_ids: List[str], metric_name: str, aggregation: str = "mean"
    ) -> Optional[float]:
        """Aggregate a metric across multiple runs.

        Args:
            run_ids: List of run IDs
            metric_name: Metric name
            aggregation: Aggregation method (mean, min, max, last)

        Returns:
            Aggregated value, or None if no metrics found

        Raises:
            ValueError: If aggregation method is invalid
        """
        if aggregation not in ("mean", "min", "max", "last"):
            raise ValueError(f"Invalid aggregation: {aggregation}")

        all_values = []
        for run_id in run_ids:
            metrics = self._storage.get_metric_history(run_id, metric_name)
            if metrics:
                all_values.append([m.value for m in metrics])

        if not all_values:
            return None

        # Aggregate based on method
        if aggregation == "mean":
            # Average of last values from each run
            last_values = [values[-1] for values in all_values]
            return sum(last_values) / len(last_values)
        elif aggregation == "min":
            # Global minimum across all runs
            return min(min(values) for values in all_values)
        elif aggregation == "max":
            # Global maximum across all runs
            return max(max(values) for values in all_values)
        elif aggregation == "last":
            # Last value from the last run
            return all_values[-1][-1]

        return None

    def get_all_metrics_summary(self, run_id: str) -> Dict[str, MetricStatistics]:
        """Get statistics for all metrics in a run.

        Args:
            run_id: Run ID

        Returns:
            Dictionary mapping metric names to statistics
        """
        all_metrics = self._storage.get_metrics(run_id)

        # Group by metric name
        metrics_by_name: Dict[str, List] = {}
        for metric in all_metrics:
            if metric.key not in metrics_by_name:
                metrics_by_name[metric.key] = []
            metrics_by_name[metric.key].append(metric)

        # Compute statistics for each metric
        summary = {}
        for metric_name, metrics in metrics_by_name.items():
            values = [m.value for m in metrics]

            summary[metric_name] = MetricStatistics(
                metric_name=metric_name,
                count=len(values),
                mean=sum(values) / len(values) if values else 0.0,
                min=min(values) if values else 0.0,
                max=max(values) if values else 0.0,
                last=values[-1] if values else 0.0,
            )

        return summary

    def compare_runs(
        self, run_ids: List[str], metric_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare metrics across multiple runs.

        Args:
            run_ids: List of run IDs to compare
            metric_names: Optional list of metrics to compare (all if None)

        Returns:
            Dictionary mapping run_id -> metric_name -> value
        """
        if metric_names is None:
            # Get all unique metric names across runs
            metric_names = set()
            for run_id in run_ids:
                metrics = self._storage.get_metrics(run_id)
                metric_names.update(m.key for m in metrics)
            metric_names = list(metric_names)

        comparison = {}
        for run_id in run_ids:
            comparison[run_id] = {}
            for metric_name in metric_names:
                stats = self.get_metric_statistics(run_id, metric_name)
                if stats:
                    comparison[run_id][metric_name] = stats.last

        return comparison


def get_metrics_history(
    run_id: str, metric_name: str, storage: IStorageBackend
) -> List[Dict[str, Any]]:
    """Get the history of a metric as a list of dictionaries.

    Convenience function for easy access to metric history.

    Args:
        run_id: Run ID
        metric_name: Metric name
        storage: Storage backend

    Returns:
        List of dictionaries with timestamp, value, and step
    """
    metrics = storage.get_metric_history(run_id, metric_name)

    return [
        {
            "timestamp": m.timestamp.isoformat(),
            "value": m.value,
            "step": m.step,
        }
        for m in metrics
    ]


def get_best_run(
    run_ids: List[str],
    metric_name: str,
    storage: IStorageBackend,
    maximize: bool = True,
) -> Optional[str]:
    """Find the best run based on a metric.

    Args:
        run_ids: List of run IDs to compare
        metric_name: Metric name to compare
        storage: Storage backend
        maximize: If True, maximize metric; if False, minimize

    Returns:
        Run ID with best metric value, or None if no metrics found
    """
    aggregator = MetricsAggregator(storage)

    best_run_id = None
    best_value = None

    for run_id in run_ids:
        stats = aggregator.get_metric_statistics(run_id, metric_name)
        if stats and stats.last is not None:
            if best_value is None:
                best_value = stats.last
                best_run_id = run_id
            else:
                if maximize:
                    if stats.last > best_value:
                        best_value = stats.last
                        best_run_id = run_id
                else:
                    if stats.last < best_value:
                        best_value = stats.last
                        best_run_id = run_id

    return best_run_id
