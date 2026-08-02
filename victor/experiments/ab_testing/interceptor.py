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

"""Workflow interceptor for A/B testing.

This module provides the interceptor that wraps workflow execution
to inject experiment configuration.
"""

import time
import uuid
from typing import Any, Callable, Dict, Optional

from victor.experiments.ab_testing.experiment import ABTestManager
from victor.experiments.ab_testing.models import ExperimentConfig


class WorkflowInterceptor:
    """Intercepts workflow execution for A/B testing.

    This class wraps workflow execution to:
    1. Allocate variants to users
    2. Apply variant-specific configuration
    3. Tag executions with experiment metadata
    4. Record metrics

    Usage:
        manager = ABTestManager()
        interceptor = WorkflowInterceptor(manager)

        # Execute workflow with experiment
        result = await interceptor.execute_with_experiment(
            workflow_func=engine.execute_yaml,
            experiment_id=exp_id,
            user_id=user_id,
            context={"query": "test"},
            yaml_path="workflow.yaml",
            initial_state={"input": "data"},
        )
    """

    def __init__(
        self,
        experiment_manager: ABTestManager,
    ):
        """Initialize workflow interceptor.

        Args:
            experiment_manager: A/B test manager instance
        """
        self.experiment_manager = experiment_manager

    async def execute_with_experiment(
        self,
        workflow_func: Callable,
        experiment_id: str,
        user_id: str,
        context: Optional[Dict] = None,
        *args,
        **kwargs,
    ) -> Any:
        """Execute workflow with A/B testing.

        Args:
            workflow_func: Workflow execution function
            experiment_id: Experiment identifier
            user_id: User identifier
            context: Optional context for allocation
            *args: Positional arguments for workflow
            **kwargs: Keyword arguments for workflow

        Returns:
            Workflow execution result

        Raises:
            ValueError: If experiment not found or not running
        """
        # Get experiment
        experiment = await self.experiment_manager.get_experiment(experiment_id)
        if not experiment:
            # No active experiment, execute normally
            return await workflow_func(*args, **kwargs)

        status = await self.experiment_manager.get_status(experiment_id)
        if not status or status.status != "running":
            # Experiment not running, execute normally
            return await workflow_func(*args, **kwargs)

        # Allocate variant
        variant_id = await self.experiment_manager.allocate_variant(experiment_id, user_id, context)

        # Apply variant configuration
        modified_kwargs = self._apply_variant_config(experiment, variant_id, kwargs)

        # Add experiment context
        execution_id = uuid.uuid4().hex
        experiment_context = {
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "user_id": user_id,
            "execution_id": execution_id,
        }

        # Inject experiment context into kwargs
        modified_kwargs["_experiment_context"] = experiment_context

        # Track start time
        start_time = time.time()

        # Execute workflow
        try:
            result = await workflow_func(*args, **modified_kwargs)

            # Record successful execution
            await self._record_execution(
                experiment_id=experiment_id,
                variant_id=variant_id,
                user_id=user_id,
                execution_id=execution_id,
                context=context,
                result=result,
                start_time=start_time,
                success=True,
                error=None,
            )

            return result

        except Exception as e:
            # Record failed execution
            await self._record_execution(
                experiment_id=experiment_id,
                variant_id=variant_id,
                user_id=user_id,
                execution_id=execution_id,
                context=context,
                result=None,
                start_time=start_time,
                success=False,
                error=str(e),
            )

            raise

    def _apply_variant_config(
        self,
        experiment: ExperimentConfig,
        variant_id: str,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply variant configuration to execution parameters.

        Args:
            experiment: Experiment configuration
            variant_id: Variant identifier
            kwargs: Original keyword arguments

        Returns:
            Modified keyword arguments
        """
        # Find variant
        variant = next((v for v in experiment.variants if v.variant_id == variant_id), None)
        if not variant:
            return kwargs

        # Apply parameter overrides
        modified = kwargs.copy()
        modified.update(variant.parameter_overrides)

        return modified

    async def _record_execution(
        self,
        experiment_id: str,
        variant_id: str,
        user_id: str,
        execution_id: str,
        context: Optional[Dict],
        result: Any,
        start_time: float,
        success: bool,
        error: Optional[str],
    ) -> None:
        """Record execution metrics.

        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            user_id: User identifier
            execution_id: Execution identifier
            context: Execution context
            result: Execution result
            start_time: Execution start time
            success: Whether execution succeeded
            error: Error message if failed
        """
        from victor.experiments.ab_testing.models import ExecutionMetrics

        # Calculate duration
        duration = time.time() - start_time

        # Extract metrics from result
        execution_time = duration
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        tool_calls_count = 0
        estimated_cost = 0.0

        if result is not None:
            # Try to extract metrics from result
            if hasattr(result, "duration_seconds"):
                execution_time = result.duration_seconds
            if hasattr(result, "prompt_tokens"):
                prompt_tokens = result.prompt_tokens
            if hasattr(result, "completion_tokens"):
                completion_tokens = result.completion_tokens
            if hasattr(result, "total_tokens"):
                total_tokens = result.total_tokens
            if hasattr(result, "tool_calls_count"):
                tool_calls_count = result.tool_calls_count
            if hasattr(result, "estimated_cost"):
                estimated_cost = result.estimated_cost

        # Create metrics
        metrics = ExecutionMetrics(
            execution_id=execution_id,
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            execution_time=execution_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls_count=tool_calls_count,
            success=success,
            error_message=error,
            estimated_cost=estimated_cost,
            timestamp=time.time(),
            workflow_name=context.get("workflow_name", "") if context else "",
            workflow_type=context.get("workflow_type", "") if context else "",
        )

        # Record metrics
        await self.experiment_manager.record_execution(metrics)


class ExperimentCompiledGraphWrapper:
    """Wraps CompiledGraph to inject experiment configuration.

    This wrapper intercepts CompiledGraph.invoke() and stream() calls to:
    1. Allocate variants
    2. Apply variant configuration
    3. Tag events with experiment metadata
    """

    def __init__(
        self,
        graph: Any,
        experiment_id: str,
        experiment_manager: ABTestManager,
    ):
        """Initialize wrapper.

        Args:
            graph: CompiledGraph instance
            experiment_id: Experiment identifier
            experiment_manager: A/B test manager instance
        """
        self.graph = graph
        self.experiment_id = experiment_id
        self.experiment_manager = experiment_manager

    async def invoke(
        self,
        initial_state: Dict[str, Any],
        user_id: str = "unknown",
        **kwargs,
    ) -> Any:
        """Invoke graph with experiment configuration.

        Args:
            initial_state: Initial state for graph
            user_id: User identifier
            **kwargs: Additional keyword arguments

        Returns:
            Graph execution result
        """
        # Get experiment
        experiment = await self.experiment_manager.get_experiment(self.experiment_id)
        if not experiment:
            # No active experiment, execute normally
            return await self.graph.invoke(initial_state, **kwargs)

        # Allocate variant
        variant_id = await self.experiment_manager.allocate_variant(
            self.experiment_id, user_id, initial_state
        )

        # Add experiment context
        execution_id = uuid.uuid4().hex
        experiment_context = {
            "experiment_id": self.experiment_id,
            "variant_id": variant_id,
            "user_id": user_id,
            "execution_id": execution_id,
        }

        # Inject into kwargs
        kwargs["_experiment_context"] = experiment_context

        # Execute graph
        start_time = time.time()
        try:
            result = await self.graph.invoke(initial_state, **kwargs)

            # Record execution
            await self._record_execution(
                experiment_id=self.experiment_id,
                variant_id=variant_id,
                user_id=user_id,
                execution_id=execution_id,
                context=initial_state,
                result=result,
                start_time=start_time,
                success=True,
                error=None,
            )

            return result

        except Exception as e:
            # Record failed execution
            await self._record_execution(
                experiment_id=self.experiment_id,
                variant_id=variant_id,
                user_id=user_id,
                execution_id=execution_id,
                context=initial_state,
                result=None,
                start_time=start_time,
                success=False,
                error=str(e),
            )
            raise

    async def stream(
        self,
        initial_state: Dict[str, Any],
        user_id: str = "unknown",
        **kwargs,
    ):
        """Stream graph execution with experiment configuration.

        Args:
            initial_state: Initial state for graph
            user_id: User identifier
            **kwargs: Additional keyword arguments

        Yields:
            Graph execution events tagged with experiment metadata
        """
        # Get experiment
        experiment = await self.experiment_manager.get_experiment(self.experiment_id)
        if not experiment:
            # No active experiment, stream normally
            async for event in self.graph.stream(initial_state, **kwargs):
                yield event
            return

        # Allocate variant
        variant_id = await self.experiment_manager.allocate_variant(
            self.experiment_id, user_id, initial_state
        )

        # Add experiment context
        execution_id = uuid.uuid4().hex
        experiment_context = {
            "experiment_id": self.experiment_id,
            "variant_id": variant_id,
            "user_id": user_id,
            "execution_id": execution_id,
        }

        # Inject into kwargs
        kwargs["_experiment_context"] = experiment_context

        # Stream graph
        start_time = time.time()
        success = True
        error = None

        try:
            async for event in self.graph.stream(initial_state, **kwargs):
                # Tag event with experiment metadata
                if isinstance(event, dict):
                    event.update(experiment_context)
                elif hasattr(event, "data"):
                    if hasattr(event.data, "update"):
                        event.data.update(experiment_context)
                    else:
                        event.data = {**event.data, **experiment_context}

                yield event

        except Exception as e:
            success = False
            error = str(e)
            raise

        finally:
            # Record execution
            await self._record_execution(
                experiment_id=self.experiment_id,
                variant_id=variant_id,
                user_id=user_id,
                execution_id=execution_id,
                context=initial_state,
                result=None,
                start_time=start_time,
                success=success,
                error=error,
            )

    async def _record_execution(
        self,
        experiment_id: str,
        variant_id: str,
        user_id: str,
        execution_id: str,
        context: Optional[Dict],
        result: Any,
        start_time: float,
        success: bool,
        error: Optional[str],
    ) -> None:
        """Record execution metrics.

        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            user_id: User identifier
            execution_id: Execution identifier
            context: Execution context
            result: Execution result
            start_time: Execution start time
            success: Whether execution succeeded
            error: Error message if failed
        """
        from victor.experiments.ab_testing.models import ExecutionMetrics

        # Calculate duration
        duration = time.time() - start_time

        # Create metrics
        metrics = ExecutionMetrics(
            execution_id=execution_id,
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            execution_time=duration,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            tool_calls_count=0,
            success=success,
            error_message=error,
            estimated_cost=0.0,
            timestamp=time.time(),
            workflow_name="",
            workflow_type="stategraph",
        )

        # Record metrics
        await self.experiment_manager.record_execution(metrics)
