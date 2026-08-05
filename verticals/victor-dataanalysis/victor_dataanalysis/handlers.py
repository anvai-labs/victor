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

"""DataAnalysis vertical compute handlers.

Domain-specific handlers for data analysis workflows:
- stats_compute: Statistical computations on datasets
- ml_training: Model training orchestration
- pycaret_automl: Automated ML using PyCaret (classification/regression)
- autosklearn_automl: Automated ML using Auto-sklearn

Usage:
    # Handlers are auto-registered when DataAnalysis vertical is loaded
    from victor.dataanalysis import handlers
    handlers.register_handlers()

    # Or in YAML workflow:
    - id: compute_stats
      type: compute
      handler: stats_compute
      inputs:
        data: $ctx.raw_data
        operations: [describe, correlation]
      output: statistics

    # AutoML with PyCaret:
    - id: automl_classify
      type: compute
      handler: pycaret_automl
      inputs:
        data: $ctx.df
        target: target_column
        task: classification
        top_n: 3
      output: automl_result

    # AutoML with Auto-sklearn:
    - id: automl_optimize
      type: compute
      handler: autosklearn_automl
      inputs:
        X: $ctx.features
        y: $ctx.target
        time_limit: 300
        task: classification
      output: autosklearn_result
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry
    from victor_contracts.workflow_runtime import ComputeNode
    from victor_contracts.workflow_runtime import NodeResult, WorkflowContext

logger = logging.getLogger(__name__)


@dataclass
class StatsComputeHandler:
    """Compute statistical measures on datasets.

    Runs statistical computations without LLM involvement.
    Supports common operations like mean, median, std, correlation.

    Example YAML:
        - id: compute_stats
          type: compute
          handler: stats_compute
          inputs:
            data: $ctx.raw_data
            operations: [describe, correlation, skewness]
          output: statistics
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor_contracts.workflow_runtime import NodeResult, ExecutorNodeStatus

        start_time = time.time()

        try:
            data = None
            operations = []

            for key, value in node.input_mapping.items():
                if key == "data":
                    data = context.get(value) if isinstance(value, str) else value
                elif key == "operations":
                    operations = value if isinstance(value, list) else [value]

            if data is None:
                return NodeResult(
                    node_id=node.id,
                    status=ExecutorNodeStatus.FAILED,
                    error="No 'data' input provided",
                    duration_seconds=time.time() - start_time,
                )

            results = {}
            for op in operations:
                results[op] = self._compute_stat(data, op)

            output_key = node.output_key or node.id
            context.set(output_key, results)

            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.COMPLETED,
                output=results,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    def _compute_stat(self, data: Any, operation: str) -> Any:
        """Compute a single statistic."""
        if isinstance(data, list) and data:
            numeric = [x for x in data if isinstance(x, (int, float))]
            if not numeric:
                return None

            if operation == "mean":
                return sum(numeric) / len(numeric)
            elif operation == "median":
                sorted_data = sorted(numeric)
                n = len(sorted_data)
                mid = n // 2
                return (sorted_data[mid] + sorted_data[~mid]) / 2
            elif operation == "min":
                return min(numeric)
            elif operation == "max":
                return max(numeric)
            elif operation == "sum":
                return sum(numeric)
            elif operation == "count":
                return len(numeric)
            elif operation == "std":
                mean = sum(numeric) / len(numeric)
                variance = sum((x - mean) ** 2 for x in numeric) / len(numeric)
                return variance**0.5
            elif operation == "describe":
                mean = sum(numeric) / len(numeric)
                return {
                    "count": len(numeric),
                    "mean": mean,
                    "min": min(numeric),
                    "max": max(numeric),
                    "sum": sum(numeric),
                }
        return None


@dataclass
class MLTrainingHandler:
    """Orchestrate ML model training.

    Manages training workflow including data split, training,
    and evaluation without LLM involvement.

    Example YAML:
        - id: train_model
          type: compute
          handler: ml_training
          inputs:
            features: $ctx.features
            target: $ctx.target
            model_type: random_forest
          output: trained_model
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor_contracts.workflow_runtime import NodeResult, ExecutorNodeStatus

        start_time = time.time()

        model_type = node.input_mapping.get("model_type", "linear")
        _features_key = node.input_mapping.get("features")  # noqa: F841
        _target_key = node.input_mapping.get("target")  # noqa: F841

        try:
            train_cmd = f"python -m victor.ml.train --model {model_type}"
            result = await tool_registry.execute("shell", command=train_cmd)

            output = {
                "model_type": model_type,
                "status": "trained" if result.success else "failed",
                "output": result.output,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=(
                    ExecutorNodeStatus.COMPLETED if result.success else ExecutorNodeStatus.FAILED
                ),
                output=output,
                duration_seconds=time.time() - start_time,
                tool_calls_used=1,
            )
        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


@dataclass
class PyCaretHandler:
    """Automated ML using PyCaret.

    Supports classification, regression, clustering, and anomaly detection.
    PyCaret provides a low-code interface to compare multiple models.

    Example YAML:
        - id: automl_classify
          type: compute
          handler: pycaret_automl
          inputs:
            data: $ctx.df
            target: target_column
            task: classification
            top_n: 3
            time_budget: 60
          output: automl_result

    Supported tasks:
        - classification: Binary/multiclass classification
        - regression: Continuous target prediction
        - clustering: Unsupervised clustering
        - anomaly: Anomaly/outlier detection

    Output includes:
        - best_model: The best performing model
        - leaderboard: Comparison of all models
        - metrics: Performance metrics for best model
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor_contracts.workflow_runtime import NodeResult, ExecutorNodeStatus

        start_time = time.time()

        # Extract inputs
        data_key = node.input_mapping.get("data")
        target = node.input_mapping.get("target")
        task = node.input_mapping.get("task", "classification")
        top_n = node.input_mapping.get("top_n", 3)
        time_budget = node.input_mapping.get("time_budget", 60)
        fold = node.input_mapping.get("fold", 5)
        sort_by = node.input_mapping.get("sort_by")  # Metric to sort by

        try:
            # Check if PyCaret is available
            try:
                import pandas as pd
            except ImportError:
                return NodeResult(
                    node_id=node.id,
                    status=ExecutorNodeStatus.FAILED,
                    error="pandas is required for PyCaret. Install with: pip install pandas",
                    duration_seconds=time.time() - start_time,
                )

            # Get data from context
            data = context.get(data_key) if isinstance(data_key, str) else data_key
            if data is None:
                return NodeResult(
                    node_id=node.id,
                    status=ExecutorNodeStatus.FAILED,
                    error="No 'data' input provided",
                    duration_seconds=time.time() - start_time,
                )

            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, dict):
                    data = pd.DataFrame(data)
                elif isinstance(data, list):
                    data = pd.DataFrame(data)
                else:
                    return NodeResult(
                        node_id=node.id,
                        status=ExecutorNodeStatus.FAILED,
                        error=f"Unsupported data type: {type(data).__name__}",
                        duration_seconds=time.time() - start_time,
                    )

            # Import appropriate PyCaret module based on task
            result = await self._run_pycaret(
                data=data,
                target=target,
                task=task,
                top_n=top_n,
                time_budget=time_budget,
                fold=fold,
                sort_by=sort_by,
            )

            output_key = node.output_key or node.id
            context.set(output_key, result)

            return NodeResult(
                node_id=node.id,
                status=(
                    ExecutorNodeStatus.COMPLETED
                    if result.get("success")
                    else ExecutorNodeStatus.FAILED
                ),
                output=result,
                duration_seconds=time.time() - start_time,
                error=result.get("error"),
            )

        except Exception as e:
            logger.exception(f"PyCaret AutoML failed: {e}")
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def _run_pycaret(
        self,
        data: Any,
        target: Optional[str],
        task: str,
        top_n: int,
        time_budget: int,
        fold: int,
        sort_by: Optional[str],
    ) -> Dict[str, Any]:
        """Run PyCaret AutoML pipeline."""
        import asyncio

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._run_pycaret_sync,
            data,
            target,
            task,
            top_n,
            time_budget,
            fold,
            sort_by,
        )

    def _run_pycaret_sync(
        self,
        data: Any,
        target: Optional[str],
        task: str,
        top_n: int,
        time_budget: int,
        fold: int,
        sort_by: Optional[str],
    ) -> Dict[str, Any]:
        """Synchronous PyCaret execution."""
        try:
            if task == "classification":
                from pycaret.classification import (
                    setup,
                    compare_models,
                    pull,
                    get_config,
                    finalize_model,
                )

                default_sort = "Accuracy"
            elif task == "regression":
                from pycaret.regression import (
                    setup,
                    compare_models,
                    pull,
                    get_config,
                    finalize_model,
                )

                default_sort = "R2"
            elif task == "clustering":
                from pycaret.clustering import (
                    setup,
                    create_model,
                    pull,
                    get_config,
                )

                # Clustering doesn't use compare_models the same way
                setup(data, session_id=42, verbose=False, html=False)
                model = create_model("kmeans")
                return {
                    "success": True,
                    "task": task,
                    "model_type": "kmeans",
                    "model": str(model),
                }
            elif task == "anomaly":
                from pycaret.anomaly import (
                    setup,
                    create_model,
                    pull,
                    get_config,
                )

                setup(data, session_id=42, verbose=False, html=False)
                model = create_model("iforest")
                return {
                    "success": True,
                    "task": task,
                    "model_type": "iforest",
                    "model": str(model),
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown task: {task}. Use: classification, regression, clustering, anomaly",
                }

            # Setup PyCaret environment
            setup(
                data,
                target=target,
                session_id=42,
                verbose=False,
                html=False,
                fold=fold,
            )

            # Compare models and get top N
            sort_metric = sort_by or default_sort
            best_models = compare_models(
                n_select=top_n,
                sort=sort_metric,
                budget_time=time_budget / 60,  # PyCaret uses minutes
            )

            # Get leaderboard
            leaderboard = pull()

            # Finalize best model
            if isinstance(best_models, list):
                best_model = best_models[0]
            else:
                best_model = best_models

            final_model = finalize_model(best_model)

            return {
                "success": True,
                "task": task,
                "best_model": str(type(final_model).__name__),
                "best_model_object": final_model,
                "leaderboard": (
                    leaderboard.to_dict() if hasattr(leaderboard, "to_dict") else str(leaderboard)
                ),
                "top_n_models": [
                    str(type(m).__name__)
                    for m in (best_models if isinstance(best_models, list) else [best_models])
                ],
                "sort_metric": sort_metric,
            }

        except ImportError as e:
            return {
                "success": False,
                "error": f"PyCaret not installed. Install with: pip install pycaret[full]. Error: {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"PyCaret execution failed: {e}",
            }


@dataclass
class AutoSklearnHandler:
    """Automated ML using Auto-sklearn.

    Auto-sklearn automates algorithm selection and hyperparameter tuning
    using Bayesian optimization. It's built on top of scikit-learn.

    Example YAML:
        - id: automl_optimize
          type: compute
          handler: autosklearn_automl
          inputs:
            X: $ctx.features
            y: $ctx.target
            time_limit: 300
            task: classification
            metric: accuracy
          output: autosklearn_result

    Supported tasks:
        - classification: Binary/multiclass classification
        - regression: Continuous target prediction

    Output includes:
        - best_model: The best performing model
        - cv_score: Cross-validation score
        - leaderboard: Model rankings
        - ensemble: Final ensemble (if applicable)
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor_contracts.workflow_runtime import NodeResult, ExecutorNodeStatus

        start_time = time.time()

        # Extract inputs
        X_key = node.input_mapping.get("X")
        y_key = node.input_mapping.get("y")
        task = node.input_mapping.get("task", "classification")
        time_limit = node.input_mapping.get("time_limit", 300)
        memory_limit = node.input_mapping.get("memory_limit", 3072)  # MB
        metric = node.input_mapping.get("metric")
        n_jobs = node.input_mapping.get("n_jobs", -1)
        ensemble_size = node.input_mapping.get("ensemble_size", 50)

        try:
            import numpy as np
        except ImportError:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error="numpy is required. Install with: pip install numpy",
                duration_seconds=time.time() - start_time,
            )

        try:
            # Get data from context
            X = context.get(X_key) if isinstance(X_key, str) else X_key
            y = context.get(y_key) if isinstance(y_key, str) else y_key

            if X is None or y is None:
                return NodeResult(
                    node_id=node.id,
                    status=ExecutorNodeStatus.FAILED,
                    error="Both 'X' (features) and 'y' (target) inputs are required",
                    duration_seconds=time.time() - start_time,
                )

            # Convert to numpy arrays if needed
            if hasattr(X, "values"):  # DataFrame
                X = X.values
            if hasattr(y, "values"):  # Series
                y = y.values
            X = np.asarray(X)
            y = np.asarray(y)

            # Run Auto-sklearn
            result = await self._run_autosklearn(
                X=X,
                y=y,
                task=task,
                time_limit=time_limit,
                memory_limit=memory_limit,
                metric=metric,
                n_jobs=n_jobs,
                ensemble_size=ensemble_size,
            )

            output_key = node.output_key or node.id
            context.set(output_key, result)

            return NodeResult(
                node_id=node.id,
                status=(
                    ExecutorNodeStatus.COMPLETED
                    if result.get("success")
                    else ExecutorNodeStatus.FAILED
                ),
                output=result,
                duration_seconds=time.time() - start_time,
                error=result.get("error"),
            )

        except Exception as e:
            logger.exception(f"Auto-sklearn AutoML failed: {e}")
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def _run_autosklearn(
        self,
        X: Any,
        y: Any,
        task: str,
        time_limit: int,
        memory_limit: int,
        metric: Optional[str],
        n_jobs: int,
        ensemble_size: int,
    ) -> Dict[str, Any]:
        """Run Auto-sklearn AutoML pipeline."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._run_autosklearn_sync,
            X,
            y,
            task,
            time_limit,
            memory_limit,
            metric,
            n_jobs,
            ensemble_size,
        )

    def _run_autosklearn_sync(
        self,
        X: Any,
        y: Any,
        task: str,
        time_limit: int,
        memory_limit: int,
        metric: Optional[str],
        n_jobs: int,
        ensemble_size: int,
    ) -> Dict[str, Any]:
        """Synchronous Auto-sklearn execution."""
        try:
            from sklearn.model_selection import train_test_split

            if task == "classification":
                from autosklearn.classification import AutoSklearnClassifier

                default_metric = "accuracy"
                automl = AutoSklearnClassifier(
                    time_left_for_this_task=time_limit,
                    per_run_time_limit=time_limit // 10,
                    memory_limit=memory_limit,
                    n_jobs=n_jobs,
                    ensemble_size=ensemble_size,
                    seed=42,
                )
            elif task == "regression":
                from autosklearn.regression import AutoSklearnRegressor

                default_metric = "r2"
                automl = AutoSklearnRegressor(
                    time_left_for_this_task=time_limit,
                    per_run_time_limit=time_limit // 10,
                    memory_limit=memory_limit,
                    n_jobs=n_jobs,
                    ensemble_size=ensemble_size,
                    seed=42,
                )
            else:
                return {
                    "success": False,
                    "error": f"Unknown task: {task}. Use: classification or regression",
                }

            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Fit Auto-sklearn
            automl.fit(X_train, y_train)

            # Get score
            score = automl.score(X_test, y_test)

            # Get leaderboard
            leaderboard = automl.leaderboard()

            # Get statistics
            stats = automl.sprint_statistics()

            return {
                "success": True,
                "task": task,
                "cv_score": float(score),
                "metric": metric or default_metric,
                "leaderboard": (
                    leaderboard.to_dict() if hasattr(leaderboard, "to_dict") else str(leaderboard)
                ),
                "statistics": stats,
                "n_models_evaluated": (
                    len(automl.cv_results_) if hasattr(automl, "cv_results_") else None
                ),
                "automl_object": automl,
            }

        except ImportError as e:
            return {
                "success": False,
                "error": f"Auto-sklearn not installed. Install with: pip install auto-sklearn. Note: Requires Linux. Error: {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Auto-sklearn execution failed: {e}",
            }


@dataclass
class RLTrainingHandler:
    """Reinforcement Learning training handler.

    Supports training RL agents using Stable-Baselines3 and Gymnasium.

    Example YAML:
        - id: train_rl_agent
          type: compute
          handler: rl_training
          inputs:
            env: CartPole-v1
            algorithm: PPO
            total_timesteps: 100000
            policy: MlpPolicy
          output: rl_result

    Supported algorithms (Stable-Baselines3):
        - PPO: Proximal Policy Optimization
        - A2C: Advantage Actor-Critic
        - DQN: Deep Q-Network (discrete actions only)
        - SAC: Soft Actor-Critic (continuous actions)
        - TD3: Twin Delayed DDPG (continuous actions)

    Output includes:
        - model: Trained model object
        - mean_reward: Mean episode reward
        - std_reward: Std deviation of rewards
        - training_time: Time taken to train
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor_contracts.workflow_runtime import NodeResult, ExecutorNodeStatus

        start_time = time.time()

        # Extract inputs
        env_id = node.input_mapping.get("env", "CartPole-v1")
        algorithm = node.input_mapping.get("algorithm", "PPO")
        total_timesteps = node.input_mapping.get("total_timesteps", 10000)
        policy = node.input_mapping.get("policy", "MlpPolicy")
        learning_rate = node.input_mapping.get("learning_rate", 3e-4)
        n_eval_episodes = node.input_mapping.get("n_eval_episodes", 10)
        save_path = node.input_mapping.get("save_path")

        try:
            result = await self._train_rl_agent(
                env_id=env_id,
                algorithm=algorithm,
                total_timesteps=total_timesteps,
                policy=policy,
                learning_rate=learning_rate,
                n_eval_episodes=n_eval_episodes,
                save_path=save_path,
            )

            output_key = node.output_key or node.id
            context.set(output_key, result)

            return NodeResult(
                node_id=node.id,
                status=(
                    ExecutorNodeStatus.COMPLETED
                    if result.get("success")
                    else ExecutorNodeStatus.FAILED
                ),
                output=result,
                duration_seconds=time.time() - start_time,
                error=result.get("error"),
            )

        except Exception as e:
            logger.exception(f"RL training failed: {e}")
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def _train_rl_agent(
        self,
        env_id: str,
        algorithm: str,
        total_timesteps: int,
        policy: str,
        learning_rate: float,
        n_eval_episodes: int,
        save_path: Optional[str],
    ) -> Dict[str, Any]:
        """Train RL agent asynchronously."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._train_rl_agent_sync,
            env_id,
            algorithm,
            total_timesteps,
            policy,
            learning_rate,
            n_eval_episodes,
            save_path,
        )

    def _train_rl_agent_sync(
        self,
        env_id: str,
        algorithm: str,
        total_timesteps: int,
        policy: str,
        learning_rate: float,
        n_eval_episodes: int,
        save_path: Optional[str],
    ) -> Dict[str, Any]:
        """Synchronous RL training."""
        try:
            import gymnasium as gym
            from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
            from stable_baselines3.common.evaluation import evaluate_policy

            # Algorithm mapping
            algo_map = {
                "PPO": PPO,
                "A2C": A2C,
                "DQN": DQN,
                "SAC": SAC,
                "TD3": TD3,
            }

            if algorithm not in algo_map:
                return {
                    "success": False,
                    "error": f"Unknown algorithm: {algorithm}. Supported: {list(algo_map.keys())}",
                }

            # Create environment
            env = gym.make(env_id)

            # Create model
            AlgoClass = algo_map[algorithm]
            model = AlgoClass(
                policy,
                env,
                learning_rate=learning_rate,
                verbose=0,
            )

            # Train
            train_start = time.time()
            model.learn(total_timesteps=total_timesteps)
            train_time = time.time() - train_start

            # Evaluate
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)

            # Save if path provided
            if save_path:
                model.save(save_path)

            env.close()

            return {
                "success": True,
                "algorithm": algorithm,
                "env": env_id,
                "total_timesteps": total_timesteps,
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "training_time_seconds": train_time,
                "model_saved": save_path if save_path else None,
                "model": model,
            }

        except ImportError as e:
            return {
                "success": False,
                "error": f"Required packages not installed. Install with: pip install stable-baselines3 gymnasium. Error: {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"RL training failed: {e}",
            }


# Handler instances
HANDLERS = {
    "stats_compute": StatsComputeHandler(),
    "ml_training": MLTrainingHandler(),
    "pycaret_automl": PyCaretHandler(),
    "autosklearn_automl": AutoSklearnHandler(),
    "rl_training": RLTrainingHandler(),
}


def register_handlers() -> None:
    """Register DataAnalysis handlers with the workflow executor."""
    from victor_contracts.workflow_runtime import register_compute_handler

    for name, handler in HANDLERS.items():
        register_compute_handler(name, handler)
        logger.debug(f"Registered DataAnalysis handler: {name}")


__all__ = [
    "StatsComputeHandler",
    "MLTrainingHandler",
    "PyCaretHandler",
    "AutoSklearnHandler",
    "RLTrainingHandler",
    "HANDLERS",
    "register_handlers",
]
