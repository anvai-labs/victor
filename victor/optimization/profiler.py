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

"""Workflow profiling and bottleneck detection.

This module provides the WorkflowProfiler class for analyzing workflow
executions to detect performance bottlenecks and optimization opportunities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import defaultdict

import numpy as np

from victor.optimization.models import (
    Bottleneck,
    BottleneckSeverity,
    BottleneckType,
    NodeStatistics,
    OptimizationOpportunity,
    OptimizationStrategyType,
    WorkflowProfile,
)
from victor.experiments.tracking import ExperimentTracker

logger = logging.getLogger(__name__)


class WorkflowProfiler:
    """Profiles workflow execution for optimization opportunities.

    The profiler analyzes historical execution data to:
    1. Calculate per-node statistics (duration, tokens, cost)
    2. Identify bottlenecks (slow nodes, failing nodes, expensive tools)
    3. Generate optimization opportunities

    Example:
        profiler = WorkflowProfiler()

        # Profile from experiment tracker
        profile = await profiler.profile_workflow(
            workflow_id="my_workflow",
            experiment_tracker=tracker,
        )

        # View bottlenecks
        for bottleneck in profile.bottlenecks:
            print(f"{bottleneck.type}: {bottleneck.suggestion}")
    """

    def __init__(
        self,
        slow_threshold_multiplier: float = 3.0,
        success_rate_threshold: float = 0.8,
        expensive_tool_threshold: float = 0.1,
    ):
        """Initialize the workflow profiler.

        Args:
            slow_threshold_multiplier: Multiplier for median duration to consider a node slow
            success_rate_threshold: Minimum success rate threshold
            expensive_tool_threshold: Minimum cost contribution to be considered expensive
        """
        self.slow_threshold_multiplier = slow_threshold_multiplier
        self.success_rate_threshold = success_rate_threshold
        self.expensive_tool_threshold = expensive_tool_threshold

    async def profile_workflow(
        self,
        workflow_id: str,
        experiment_tracker: ExperimentTracker,
        min_executions: int = 3,
    ) -> Optional[WorkflowProfile]:
        """Profile a workflow from historical execution data.

        Args:
            workflow_id: Workflow to profile
            experiment_tracker: Experiment tracker instance
            min_executions: Minimum number of executions required

        Returns:
            WorkflowProfile if enough data available, None otherwise
        """
        logger.info(f"Profiling workflow: {workflow_id}")

        # Fetch historical executions from experiment tracker
        executions = await self._fetch_executions(
            workflow_id,
            experiment_tracker,
            min_executions,
        )

        if not executions:
            logger.warning(
                f"Insufficient execution data for workflow {workflow_id}. "
                f"Need at least {min_executions} executions."
            )
            return None

        logger.info(f"Analyzing {len(executions)} executions")

        # Calculate node statistics
        node_stats = self._calculate_node_statistics(executions)

        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(node_stats)

        # Generate optimization opportunities
        opportunities = self._generate_opportunities(bottlenecks, node_stats)

        # Calculate workflow-level aggregates
        total_duration = sum(
            max(node_stats[node_id].p99_duration for node_id in node_stats) for _ in executions
        ) / len(executions)

        total_cost = sum(stats.total_cost for stats in node_stats.values())

        total_tokens = sum(
            stats.avg_input_tokens + stats.avg_output_tokens for stats in node_stats.values()
        )

        success_rate = np.mean([exec.get("success", True) for exec in executions])

        profile = WorkflowProfile(
            workflow_id=workflow_id,
            node_stats=node_stats,
            bottlenecks=bottlenecks,
            opportunities=opportunities,
            total_duration=total_duration,
            total_cost=total_cost,
            total_tokens=total_tokens,
            success_rate=success_rate,
            num_executions=len(executions),
        )

        logger.info(
            f"Profile created: {len(bottlenecks)} bottlenecks, "
            f"{len(opportunities)} opportunities"
        )

        return profile

    async def _fetch_executions(
        self,
        workflow_id: str,
        tracker: ExperimentTracker,
        min_executions: int,
    ) -> List[Dict[str, Any]]:
        """Fetch historical executions from experiment tracker.

        Args:
            workflow_id: Workflow identifier
            tracker: Experiment tracker instance
            min_executions: Minimum executions needed

        Returns:
            List of execution records
        """
        # Query experiment tracker for workflow executions
        # This is a simplified implementation - in production, you'd query
        # the actual storage backend with proper filtering

        executions = []

        try:
            # Get experiment by name
            experiment = tracker.get_experiment(workflow_id)

            if experiment:
                # Get runs for this experiment
                runs = tracker.search_runs(
                    experiment_ids=[workflow_id],
                )

                for run in runs:
                    execution = {
                        "run_id": run.info.run_id,
                        "success": run.info.status == "FINISHED",
                        "duration": run.data.metrics.get("duration", 0),
                        "cost": run.data.params.get("cost", 0),
                        "node_metrics": run.data.params.get("node_metrics", {}),
                        "tool_metrics": run.data.params.get("tool_metrics", {}),
                    }
                    executions.append(execution)

        except Exception as e:
            logger.error(f"Error fetching executions: {e}")

        return executions[:min_executions] if executions else []

    def _calculate_node_statistics(
        self,
        executions: List[Dict[str, Any]],
    ) -> Dict[str, NodeStatistics]:
        """Calculate statistics for each node across executions.

        Args:
            executions: List of execution records

        Returns:
            Dictionary mapping node IDs to statistics
        """
        node_data = defaultdict(
            lambda: {
                "durations": [],
                "input_tokens": [],
                "output_tokens": [],
                "costs": [],
                "successes": [],
                "errors": defaultdict(int),
                "tool_calls": defaultdict(
                    lambda: {
                        "count": 0,
                        "cost": 0,
                    }
                ),
            }
        )

        # Aggregate data across executions
        for exec_data in executions:
            node_metrics = exec_data.get("node_metrics", {})

            for node_id, metrics in node_metrics.items():
                node_data[node_id]["durations"].append(metrics.get("duration", 0))
                node_data[node_id]["input_tokens"].append(metrics.get("input_tokens", 0))
                node_data[node_id]["output_tokens"].append(metrics.get("output_tokens", 0))
                node_data[node_id]["costs"].append(metrics.get("cost", 0))
                node_data[node_id]["successes"].append(metrics.get("success", True))

                # Track errors
                if not metrics.get("success", True):
                    error_type = metrics.get("error_type", "unknown")
                    node_data[node_id]["errors"][error_type] += 1

                # Track tool calls
                for tool_name, tool_stats in metrics.get("tools", {}).items():
                    node_data[node_id]["tool_calls"][tool_name]["count"] += tool_stats.get(
                        "count", 0
                    )
                    node_data[node_id]["tool_calls"][tool_name]["cost"] += tool_stats.get("cost", 0)

        # Calculate statistics
        node_stats = {}

        for node_id, data in node_data.items():
            durations = np.array(data["durations"])
            input_tokens = np.array(data["input_tokens"])
            output_tokens = np.array(data["output_tokens"])
            costs = np.array(data["costs"])
            successes = np.array(data["successes"], dtype=bool)

            # Token efficiency
            token_efficiency = 0.0
            if input_tokens.sum() > 0:
                token_efficiency = output_tokens.sum() / input_tokens.sum()

            stats = NodeStatistics(
                node_id=node_id,
                avg_duration=float(np.mean(durations)),
                p50_duration=float(np.percentile(durations, 50)),
                p95_duration=float(np.percentile(durations, 95)),
                p99_duration=float(np.percentile(durations, 99)),
                success_rate=float(np.mean(successes)),
                error_types=dict(data["errors"]),
                avg_input_tokens=int(np.mean(input_tokens)),
                avg_output_tokens=int(np.mean(output_tokens)),
                token_efficiency=token_efficiency,
                tool_calls=dict(data["tool_calls"]),
                total_cost=float(np.sum(costs)),
            )

            node_stats[node_id] = stats

        return node_stats

    def _detect_bottlenecks(
        self,
        node_stats: Dict[str, NodeStatistics],
    ) -> List[Bottleneck]:
        """Detect performance bottlenecks from node statistics.

        Args:
            node_stats: Dictionary of node statistics

        Returns:
            List of detected bottlenecks
        """
        bottlenecks = []

        if not node_stats:
            return bottlenecks

        # Calculate median duration
        median_duration = np.median([stats.avg_duration for stats in node_stats.values()])

        total_cost = sum(stats.total_cost for stats in node_stats.values())

        # Detect slow nodes
        for node_id, stats in node_stats.items():
            if stats.avg_duration > self.slow_threshold_multiplier * median_duration:
                bottlenecks.append(
                    Bottleneck(
                        type=BottleneckType.SLOW_NODE,
                        severity=BottleneckSeverity.HIGH,
                        node_id=node_id,
                        metric="duration",
                        value=stats.avg_duration,
                        threshold=self.slow_threshold_multiplier * median_duration,
                        suggestion=f"Node '{node_id}' is {stats.avg_duration / median_duration:.1f}x slower than median. "
                        f"Consider parallelization or breaking into smaller nodes.",
                        confidence=0.9,
                    )
                )

            # Check for dominant nodes
            if median_duration > 0:
                contribution = stats.avg_duration / sum(s.avg_duration for s in node_stats.values())
                if contribution > 0.20:
                    bottlenecks.append(
                        Bottleneck(
                            type=BottleneckType.DOMINANT_NODE,
                            severity=BottleneckSeverity.MEDIUM,
                            node_id=node_id,
                            metric="time_contribution",
                            value=contribution * 100,
                            threshold=20.0,
                            suggestion=f"Node '{node_id}' consumes {contribution*100:.1f}% of workflow time. "
                            f"Consider optimization or parallelization.",
                            confidence=0.8,
                        )
                    )

            # Detect failing nodes
            if stats.success_rate < self.success_rate_threshold:
                bottlenecks.append(
                    Bottleneck(
                        type=BottleneckType.UNRELIABLE_NODE,
                        severity=BottleneckSeverity.HIGH,
                        node_id=node_id,
                        metric="success_rate",
                        value=stats.success_rate * 100,
                        threshold=self.success_rate_threshold * 100,
                        suggestion=f"Node '{node_id}' success rate {stats.success_rate*100:.1f}% is below threshold. "
                        f"Review error handling or consider pruning.",
                        confidence=0.95,
                    )
                )

            # Detect expensive tools
            for tool_id, tool_data in stats.tool_calls.items():
                tool_cost = tool_data.get("cost", 0)
                if total_cost > 0 and tool_cost / total_cost > self.expensive_tool_threshold:
                    bottlenecks.append(
                        Bottleneck(
                            type=BottleneckType.EXPENSIVE_TOOL,
                            severity=BottleneckSeverity.MEDIUM,
                            tool_id=tool_id,
                            metric="cost_contribution",
                            value=tool_cost / total_cost * 100,
                            threshold=self.expensive_tool_threshold * 100,
                            suggestion=f"Tool '{tool_id}' in node '{node_id}' consumes {tool_cost/total_cost*100:.1f}% of cost. "
                            f"Consider cheaper alternative or caching.",
                            confidence=0.85,
                        )
                    )

            # Detect missing caching
            if stats.token_efficiency > 0.8:  # High consistency suggests deterministic
                # Check if caching is not enabled (simplified check)
                if not any("cache" in str(k).lower() for k in stats.tool_calls.keys()):
                    bottlenecks.append(
                        Bottleneck(
                            type=BottleneckType.MISSING_CACHING,
                            severity=BottleneckSeverity.LOW,
                            node_id=node_id,
                            metric="determinism",
                            value=stats.token_efficiency * 100,
                            threshold=80.0,
                            suggestion=f"Node '{node_id}' shows high token efficiency ({stats.token_efficiency:.2f}), "
                            f"suggesting deterministic operations. Consider enabling caching.",
                            confidence=0.7,
                        )
                    )

        return bottlenecks

    def _generate_opportunities(
        self,
        bottlenecks: List[Bottleneck],
        node_stats: Dict[str, NodeStatistics],
    ) -> List[OptimizationOpportunity]:
        """Generate optimization opportunities from bottlenecks.

        Args:
            bottlenecks: Detected bottlenecks
            node_stats: Node statistics

        Returns:
            List of optimization opportunities
        """
        opportunities = []

        for bottleneck in bottlenecks:
            if bottleneck.type == BottleneckType.UNRELIABLE_NODE:
                # Pruning opportunity
                if bottleneck.node_id:
                    stats = node_stats.get(bottleneck.node_id)
                    if stats:
                        opportunities.append(
                            OptimizationOpportunity(
                                strategy_type=OptimizationStrategyType.PRUNING,
                                target=bottleneck.node_id,
                                description=f"Remove consistently failing node '{bottleneck.node_id}' "
                                f"(success rate: {stats.success_rate:.1%})",
                                expected_improvement=1.0 - stats.success_rate,
                                risk_level=BottleneckSeverity.HIGH,
                                estimated_cost_reduction=stats.total_cost,
                                estimated_duration_reduction=stats.avg_duration,
                                confidence=0.6,
                            )
                        )

            elif bottleneck.type == BottleneckType.EXPENSIVE_TOOL:
                # Tool selection opportunity
                if bottleneck.tool_id:
                    cost_reduction = bottleneck.value * 0.5  # Estimate 50% reduction
                    opportunities.append(
                        OptimizationOpportunity(
                            strategy_type=OptimizationStrategyType.TOOL_SELECTION,
                            target=bottleneck.tool_id,
                            description=f"Substitute expensive tool '{bottleneck.tool_id}' with cheaper alternative",
                            expected_improvement=cost_reduction / 100,
                            risk_level=BottleneckSeverity.LOW,
                            estimated_cost_reduction=cost_reduction / 100,
                            confidence=0.7,
                        )
                    )

            elif bottleneck.type in [
                BottleneckType.SLOW_NODE,
                BottleneckType.DOMINANT_NODE,
            ]:
                # Parallelization opportunity
                if bottleneck.node_id:
                    # Estimate potential speedup from parallelization
                    # (simplified - actual analysis would check dependencies)
                    speedup = 2.0  # Assume 2x speedup potential
                    opportunities.append(
                        OptimizationOpportunity(
                            strategy_type=OptimizationStrategyType.PARALLELIZATION,
                            target=bottleneck.node_id,
                            description=f"Parallelize independent nodes including '{bottleneck.node_id}'",
                            expected_improvement=(speedup - 1) / speedup,
                            risk_level=BottleneckSeverity.MEDIUM,
                            estimated_duration_reduction=bottleneck.value
                            * (speedup - 1)
                            / speedup
                            / 100,
                            confidence=0.6,
                        )
                    )

            elif bottleneck.type == BottleneckType.MISSING_CACHING:
                # Caching opportunity
                if bottleneck.node_id:
                    stats = node_stats.get(bottleneck.node_id)
                    if stats:
                        hit_rate = 0.75  # Estimate 75% hit rate
                        speedup = 1 / (hit_rate * 0.01 + (1 - hit_rate))  # Cache is 100x faster
                        opportunities.append(
                            OptimizationOpportunity(
                                strategy_type=OptimizationStrategyType.CACHING,
                                target=bottleneck.node_id,
                                description=f"Enable caching for deterministic node '{bottleneck.node_id}' "
                                f"(estimated {hit_rate:.0%} hit rate, {speedup:.1f}x speedup)",
                                expected_improvement=(speedup - 1) / speedup,
                                risk_level=BottleneckSeverity.LOW,
                                estimated_duration_reduction=stats.avg_duration
                                * (speedup - 1)
                                / speedup,
                                estimated_cost_reduction=stats.total_cost * hit_rate,
                                confidence=0.8,
                            )
                        )

        # Sort by expected improvement
        opportunities.sort(key=lambda o: o.expected_improvement, reverse=True)

        return opportunities
