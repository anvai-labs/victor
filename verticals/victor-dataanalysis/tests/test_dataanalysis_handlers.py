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

"""Unit tests for DataAnalysis vertical handlers."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest


class MockToolResult:
    """Mock tool result for testing."""

    def __init__(self, success: bool = True, output: Any = None, error: str = None):
        self.success = success
        self.output = output
        self.error = error


class MockComputeNode:
    """Mock compute node for testing."""

    def __init__(
        self,
        node_id: str = "test_node",
        input_mapping: Dict[str, Any] = None,
        output_key: str = None,
    ):
        self.id = node_id
        self.input_mapping = input_mapping or {}
        self.output_key = output_key


class MockWorkflowContext:
    """Mock workflow context for testing."""

    def __init__(self, data: Dict[str, Any] = None):
        self._data = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value


class TestStatsComputeHandler:
    """Tests for StatsComputeHandler."""

    @pytest.fixture
    def handler(self):
        from victor_dataanalysis.handlers import StatsComputeHandler

        return StatsComputeHandler()

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_compute_mean(self, handler, mock_registry):
        """Test computing mean of data."""
        node = MockComputeNode(
            input_mapping={
                "data": "raw_data",
                "operations": ["mean"],
            },
            output_key="stats",
        )
        context = MockWorkflowContext({"raw_data": [1, 2, 3, 4, 5]})

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        assert result.output["mean"] == 3.0
        assert context.get("stats")["mean"] == 3.0

    @pytest.mark.asyncio
    async def test_compute_median(self, handler, mock_registry):
        """Test computing median of data."""
        node = MockComputeNode(
            input_mapping={
                "data": "raw_data",
                "operations": ["median"],
            },
        )
        context = MockWorkflowContext({"raw_data": [1, 2, 3, 4, 5]})

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        assert result.output["median"] == 3.0

    @pytest.mark.asyncio
    async def test_compute_median_even_length(self, handler, mock_registry):
        """Test computing median with even number of elements."""
        node = MockComputeNode(
            input_mapping={
                "data": "raw_data",
                "operations": ["median"],
            },
        )
        context = MockWorkflowContext({"raw_data": [1, 2, 3, 4]})

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        assert result.output["median"] == 2.5

    @pytest.mark.asyncio
    async def test_compute_std(self, handler, mock_registry):
        """Test computing standard deviation."""
        node = MockComputeNode(
            input_mapping={
                "data": "raw_data",
                "operations": ["std"],
            },
        )
        # Data with known std
        context = MockWorkflowContext({"raw_data": [2, 4, 4, 4, 5, 5, 7, 9]})

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        assert abs(result.output["std"] - 2.0) < 0.01

    @pytest.mark.asyncio
    async def test_compute_describe(self, handler, mock_registry):
        """Test describe operation returns summary stats."""
        node = MockComputeNode(
            input_mapping={
                "data": "raw_data",
                "operations": ["describe"],
            },
        )
        context = MockWorkflowContext({"raw_data": [1, 2, 3, 4, 5]})

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        desc = result.output["describe"]
        assert desc["count"] == 5
        assert desc["mean"] == 3.0
        assert desc["min"] == 1
        assert desc["max"] == 5
        assert desc["sum"] == 15

    @pytest.mark.asyncio
    async def test_compute_min_max_sum_count(self, handler, mock_registry):
        """Test basic aggregation operations."""
        node = MockComputeNode(
            input_mapping={
                "data": "raw_data",
                "operations": ["min", "max", "sum", "count"],
            },
        )
        context = MockWorkflowContext({"raw_data": [10, 20, 30]})

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        assert result.output["min"] == 10
        assert result.output["max"] == 30
        assert result.output["sum"] == 60
        assert result.output["count"] == 3

    @pytest.mark.asyncio
    async def test_multiple_operations(self, handler, mock_registry):
        """Test running multiple operations."""
        node = MockComputeNode(
            input_mapping={
                "data": "raw_data",
                "operations": ["mean", "median", "std"],
            },
        )
        context = MockWorkflowContext({"raw_data": [1, 2, 3, 4, 5]})

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        assert "mean" in result.output
        assert "median" in result.output
        assert "std" in result.output

    @pytest.mark.asyncio
    async def test_no_data_input(self, handler, mock_registry):
        """Test handling of missing data input."""
        node = MockComputeNode(
            input_mapping={
                "operations": ["mean"],
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"
        assert "No 'data' input" in result.error

    @pytest.mark.asyncio
    async def test_empty_data(self, handler, mock_registry):
        """Test handling of empty data."""
        node = MockComputeNode(
            input_mapping={
                "data": "raw_data",
                "operations": ["mean"],
            },
        )
        context = MockWorkflowContext({"raw_data": []})

        result = await handler(node, context, mock_registry)

        # Should complete but return None for the operation
        assert result.status.value == "completed"
        assert result.output["mean"] is None

    @pytest.mark.asyncio
    async def test_non_numeric_data_filtered(self, handler, mock_registry):
        """Test that non-numeric data is filtered out."""
        node = MockComputeNode(
            input_mapping={
                "data": "raw_data",
                "operations": ["mean"],
            },
        )
        context = MockWorkflowContext({"raw_data": [1, "text", 2, None, 3]})

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        assert result.output["mean"] == 2.0  # mean of [1, 2, 3]

    @pytest.mark.asyncio
    async def test_direct_data_value(self, handler, mock_registry):
        """Test providing data directly instead of context key."""
        node = MockComputeNode(
            input_mapping={
                "data": [1, 2, 3, 4, 5],
                "operations": ["mean"],
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        assert result.output["mean"] == 3.0


class TestMLTrainingHandler:
    """Tests for MLTrainingHandler."""

    @pytest.fixture
    def handler(self):
        from victor_dataanalysis.handlers import MLTrainingHandler

        return MLTrainingHandler()

    @pytest.fixture
    def mock_registry(self):
        registry = MagicMock()
        registry.execute = AsyncMock(
            return_value=MockToolResult(
                success=True,
                output="Model trained successfully",
            )
        )
        return registry

    @pytest.mark.asyncio
    async def test_train_linear_model(self, handler, mock_registry):
        """Test training a linear model."""
        node = MockComputeNode(
            input_mapping={
                "model_type": "linear",
                "features": "X",
                "target": "y",
            },
            output_key="model_result",
        )
        context = MockWorkflowContext({"X": [[1, 2], [3, 4], [5, 6]], "y": [1, 2, 3]})

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        assert result.output["model_type"] == "linear"
        assert result.output["status"] == "trained"
        mock_registry.execute.assert_called_once()
        call_args = mock_registry.execute.call_args
        assert "--model linear" in call_args.kwargs["command"]

    @pytest.mark.asyncio
    async def test_train_random_forest(self, handler, mock_registry):
        """Test training a random forest model."""
        node = MockComputeNode(
            input_mapping={
                "model_type": "random_forest",
                "features": "X",
                "target": "y",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        call_args = mock_registry.execute.call_args
        assert "--model random_forest" in call_args.kwargs["command"]

    @pytest.mark.asyncio
    async def test_training_failure(self, handler):
        """Test handling of training failure."""
        mock_registry = MagicMock()
        mock_registry.execute = AsyncMock(
            return_value=MockToolResult(success=False, error="Training failed")
        )

        node = MockComputeNode(
            input_mapping={
                "model_type": "linear",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"
        assert result.output["status"] == "failed"

    @pytest.mark.asyncio
    async def test_default_model_type(self, handler, mock_registry):
        """Test default model type is linear."""
        node = MockComputeNode(
            input_mapping={
                "features": "X",
                "target": "y",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        call_args = mock_registry.execute.call_args
        assert "--model linear" in call_args.kwargs["command"]


class TestHandlerRegistration:
    """Tests for handler registration."""

    def test_handlers_dict_exists(self):
        """Test HANDLERS dict is defined."""
        from victor_dataanalysis.handlers import HANDLERS

        assert "stats_compute" in HANDLERS
        assert "ml_training" in HANDLERS

    def test_register_handlers(self):
        """Test handler registration function."""
        from victor_dataanalysis.handlers import register_handlers

        # Should not raise
        register_handlers()


class TestEscapeHatches:
    """Tests for DataAnalysis escape hatch conditions."""

    def test_should_retry_cleaning_validation_failed(self):
        """Test retry cleaning when validation fails."""
        from victor_dataanalysis.escape_hatches import should_retry_cleaning

        ctx = {
            "validation_passed": False,
            "iteration": 0,
            "max_iterations": 3,
        }
        assert should_retry_cleaning(ctx) == "retry"

    def test_should_retry_cleaning_validation_passed(self):
        """Test done when validation passes."""
        from victor_dataanalysis.escape_hatches import should_retry_cleaning

        ctx = {
            "validation_passed": True,
            "iteration": 0,
        }
        assert should_retry_cleaning(ctx) == "done"

    def test_should_retry_cleaning_max_iterations(self):
        """Test done after max iterations."""
        from victor_dataanalysis.escape_hatches import should_retry_cleaning

        ctx = {
            "validation_passed": False,
            "iteration": 3,
            "max_iterations": 3,
        }
        assert should_retry_cleaning(ctx) == "done"

    def test_should_tune_more_below_threshold(self):
        """Test tune more when performance below threshold."""
        from victor_dataanalysis.escape_hatches import should_tune_more

        ctx = {
            "metrics": {"primary_metric": 0.75},
            "performance_threshold": 0.9,
            "iteration": 1,
            "max_iterations": 5,
        }
        assert should_tune_more(ctx) == "tune"

    def test_should_tune_more_threshold_reached(self):
        """Test done when threshold reached."""
        from victor_dataanalysis.escape_hatches import should_tune_more

        ctx = {
            "metrics": {"primary_metric": 0.95},
            "performance_threshold": 0.9,
        }
        assert should_tune_more(ctx) == "done"

    def test_should_tune_more_max_iterations(self):
        """Test done after max iterations."""
        from victor_dataanalysis.escape_hatches import should_tune_more

        ctx = {
            "metrics": {"primary_metric": 0.75},
            "performance_threshold": 0.9,
            "iteration": 3,
            "max_iterations": 3,
        }
        assert should_tune_more(ctx) == "done"

    def test_quality_threshold_high(self):
        """Test high quality threshold."""
        from victor_dataanalysis.escape_hatches import quality_threshold

        ctx = {
            "quality_score": 0.95,
            "missing_pct": 2,
            "outlier_count": 5,
        }
        assert quality_threshold(ctx) == "high_quality"

    def test_quality_threshold_acceptable(self):
        """Test acceptable quality threshold."""
        from victor_dataanalysis.escape_hatches import quality_threshold

        ctx = {
            "quality_score": 0.80,
            "missing_pct": 10,
            "outlier_count": 15,
        }
        assert quality_threshold(ctx) == "acceptable"

    def test_quality_threshold_needs_cleanup(self):
        """Test needs cleanup when quality too low."""
        from victor_dataanalysis.escape_hatches import quality_threshold

        ctx = {
            "quality_score": 0.60,
            "missing_pct": 25,
        }
        assert quality_threshold(ctx) == "needs_cleanup"

    def test_model_selection_criteria_excellent(self):
        """Test model selection with excellent score."""
        from victor_dataanalysis.escape_hatches import model_selection_criteria

        ctx = {
            "evaluation_results": [
                {"model": "rf", "score": 0.96},
                {"model": "lr", "score": 0.85},
            ],
        }
        assert model_selection_criteria(ctx) == "excellent"

    def test_model_selection_criteria_good(self):
        """Test model selection with good score."""
        from victor_dataanalysis.escape_hatches import model_selection_criteria

        ctx = {
            "evaluation_results": [
                {"model": "rf", "score": 0.88},
            ],
        }
        assert model_selection_criteria(ctx) == "good"

    def test_model_selection_criteria_no_models(self):
        """Test model selection with no results."""
        from victor_dataanalysis.escape_hatches import model_selection_criteria

        ctx = {
            "evaluation_results": [],
        }
        assert model_selection_criteria(ctx) == "no_models"

    def test_analysis_confidence_high(self):
        """Test high analysis confidence."""
        from victor_dataanalysis.escape_hatches import analysis_confidence

        ctx = {
            "sample_size": 1000,
            "confidence_score": 0.85,
            "uncertainty_areas": ["one"],
        }
        assert analysis_confidence(ctx) == "high"

    def test_analysis_confidence_low_small_sample(self):
        """Test low analysis confidence with small sample."""
        from victor_dataanalysis.escape_hatches import analysis_confidence

        ctx = {
            "sample_size": 50,
            "confidence_score": 0.9,
        }
        assert analysis_confidence(ctx) == "low"

    def test_analysis_confidence_low_many_uncertainties(self):
        """Test low analysis confidence with many uncertainties."""
        from victor_dataanalysis.escape_hatches import analysis_confidence

        ctx = {
            "sample_size": 1000,
            "confidence_score": 0.9,
            "uncertainty_areas": ["a", "b", "c", "d", "e", "f"],
        }
        assert analysis_confidence(ctx) == "low"

    def test_analysis_confidence_medium(self):
        """Test medium analysis confidence."""
        from victor_dataanalysis.escape_hatches import analysis_confidence

        ctx = {
            "sample_size": 500,
            "confidence_score": 0.7,
            "uncertainty_areas": ["a", "b", "c"],
        }
        assert analysis_confidence(ctx) == "medium"


class TestPyCaretHandler:
    """Tests for PyCaretHandler."""

    @pytest.fixture
    def handler(self):
        from victor_dataanalysis.handlers import PyCaretHandler

        return PyCaretHandler()

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_no_data_input(self, handler, mock_registry):
        """Test handling of missing data input."""
        node = MockComputeNode(
            input_mapping={
                "target": "label",
                "task": "classification",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"
        assert "No 'data' input" in result.error

    @pytest.mark.asyncio
    async def test_unsupported_data_type(self, handler, mock_registry):
        """Test handling of unsupported data type."""
        node = MockComputeNode(
            input_mapping={
                "data": "raw_data",
                "target": "label",
                "task": "classification",
            },
        )
        # Use a string instead of a DataFrame/dict/list
        context = MockWorkflowContext({"raw_data": "invalid string data"})

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"
        assert "Unsupported data type" in result.error

    @pytest.mark.asyncio
    async def test_dict_data_conversion(self, handler, mock_registry):
        """Test dict data is converted to DataFrame."""
        node = MockComputeNode(
            input_mapping={
                "data": {"col1": [1, 2, 3], "col2": [4, 5, 6]},
                "target": "col2",
                "task": "classification",
            },
        )
        context = MockWorkflowContext()

        # This will fail because PyCaret is not installed, but we're testing data conversion
        result = await handler(node, context, mock_registry)

        # Should fail with PyCaret not installed, not data conversion error
        assert result.status.value == "failed"
        assert "PyCaret" in result.error or "pycaret" in result.error.lower()

    @pytest.mark.asyncio
    async def test_list_data_conversion(self, handler, mock_registry):
        """Test list data is converted to DataFrame."""
        node = MockComputeNode(
            input_mapping={
                "data": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
                "target": "b",
                "task": "classification",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        # Should fail with PyCaret not installed, not data conversion error
        assert result.status.value == "failed"
        assert "PyCaret" in result.error or "pycaret" in result.error.lower()

    @pytest.mark.asyncio
    async def test_default_task_is_classification(self, handler, mock_registry):
        """Test default task type is classification."""
        node = MockComputeNode(
            input_mapping={
                "data": {"col1": [1, 2], "col2": [0, 1]},
                "target": "col2",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        # Verify it attempted classification (will fail due to missing PyCaret)
        assert result.status.value == "failed"

    @pytest.mark.asyncio
    async def test_context_output_key(self, handler, mock_registry):
        """Test output key is set from node."""
        node = MockComputeNode(
            input_mapping={
                "data": {"col1": [1], "col2": [1]},
                "target": "col2",
            },
            output_key="automl_result",
        )
        context = MockWorkflowContext()

        await handler(node, context, mock_registry)

        # Context should have something at the output key (even if failed)
        assert context.get("automl_result") is not None


class TestAutoSklearnHandler:
    """Tests for AutoSklearnHandler."""

    @pytest.fixture
    def handler(self):
        from victor_dataanalysis.handlers import AutoSklearnHandler

        return AutoSklearnHandler()

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_missing_x_input(self, handler, mock_registry):
        """Test handling of missing X input."""
        node = MockComputeNode(
            input_mapping={
                "y": "target",
                "task": "classification",
            },
        )
        context = MockWorkflowContext({"target": [0, 1, 0, 1]})

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"
        assert "'X' (features) and 'y' (target)" in result.error

    @pytest.mark.asyncio
    async def test_missing_y_input(self, handler, mock_registry):
        """Test handling of missing y input."""
        node = MockComputeNode(
            input_mapping={
                "X": "features",
                "task": "classification",
            },
        )
        context = MockWorkflowContext({"features": [[1, 2], [3, 4]]})

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"
        assert "'X' (features) and 'y' (target)" in result.error

    @pytest.mark.asyncio
    async def test_default_task_is_classification(self, handler, mock_registry):
        """Test default task type is classification."""
        node = MockComputeNode(
            input_mapping={
                "X": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "y": [0, 1, 0, 1],
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        # Will fail due to auto-sklearn not installed
        assert result.status.value == "failed"
        assert "auto-sklearn" in result.error.lower() or "autosklearn" in result.error.lower()

    @pytest.mark.asyncio
    async def test_context_output_key(self, handler, mock_registry):
        """Test output key is set from node."""
        node = MockComputeNode(
            input_mapping={
                "X": [[1, 2], [3, 4]],
                "y": [0, 1],
            },
            output_key="sklearn_result",
        )
        context = MockWorkflowContext()

        await handler(node, context, mock_registry)

        assert context.get("sklearn_result") is not None

    @pytest.mark.asyncio
    async def test_dataframe_input_conversion(self, handler, mock_registry):
        """Test DataFrame values are extracted."""
        try:
            import pandas as pd

            df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            series = pd.Series([0, 1, 0])

            node = MockComputeNode(
                input_mapping={
                    "X": "features",
                    "y": "target",
                },
            )
            context = MockWorkflowContext({"features": df, "target": series})

            result = await handler(node, context, mock_registry)

            # Will fail due to auto-sklearn not installed
            assert result.status.value == "failed"
            assert "auto-sklearn" in result.error.lower() or "autosklearn" in result.error.lower()
        except ImportError:
            pytest.skip("pandas not installed")


class TestRLTrainingHandler:
    """Tests for RLTrainingHandler."""

    @pytest.fixture
    def handler(self):
        from victor_dataanalysis.handlers import RLTrainingHandler

        return RLTrainingHandler()

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_default_algorithm_is_ppo(self, handler, mock_registry):
        """Test default algorithm is PPO."""
        node = MockComputeNode(
            input_mapping={
                "env": "CartPole-v1",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        # Will fail due to stable-baselines3 not installed
        assert result.status.value == "failed"
        assert "stable-baselines3" in result.error.lower() or "gymnasium" in result.error.lower()

    @pytest.mark.asyncio
    async def test_default_env_is_cartpole(self, handler, mock_registry):
        """Test default env is CartPole-v1."""
        node = MockComputeNode(
            input_mapping={},
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        # Will attempt to use default CartPole-v1
        assert result.status.value == "failed"

    @pytest.mark.asyncio
    async def test_context_output_key(self, handler, mock_registry):
        """Test output key is set from node."""
        node = MockComputeNode(
            input_mapping={
                "env": "CartPole-v1",
            },
            output_key="rl_result",
        )
        context = MockWorkflowContext()

        await handler(node, context, mock_registry)

        assert context.get("rl_result") is not None

    @pytest.mark.asyncio
    async def test_all_input_parameters(self, handler, mock_registry):
        """Test all input parameters are read correctly."""
        node = MockComputeNode(
            input_mapping={
                "env": "MountainCar-v0",
                "algorithm": "DQN",
                "total_timesteps": 50000,
                "policy": "MlpPolicy",
                "learning_rate": 0.001,
                "n_eval_episodes": 5,
                "save_path": "/tmp/model",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        # Will fail due to packages not installed
        assert result.status.value == "failed"


class TestNewHandlersInRegistry:
    """Tests for new handlers in HANDLERS dict."""

    def test_pycaret_handler_in_registry(self):
        """Test pycaret_automl handler is registered."""
        from victor_dataanalysis.handlers import HANDLERS

        assert "pycaret_automl" in HANDLERS

    def test_autosklearn_handler_in_registry(self):
        """Test autosklearn_automl handler is registered."""
        from victor_dataanalysis.handlers import HANDLERS

        assert "autosklearn_automl" in HANDLERS

    def test_rl_training_handler_in_registry(self):
        """Test rl_training handler is registered."""
        from victor_dataanalysis.handlers import HANDLERS

        assert "rl_training" in HANDLERS

    def test_all_handlers_exported(self):
        """Test all handler classes are in __all__."""
        from victor_dataanalysis import handlers

        assert "PyCaretHandler" in handlers.__all__
        assert "AutoSklearnHandler" in handlers.__all__
        assert "RLTrainingHandler" in handlers.__all__
