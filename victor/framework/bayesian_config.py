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

"""Bayesian orchestration configuration."""

from dataclasses import dataclass, field


@dataclass
class BayesianConfig:
    """Configuration for Bayesian orchestration.

    Attributes:
        enabled: Whether Bayesian orchestration is enabled
        force_all: Force all queries through Bayesian (disable complexity routing)
        simple_threshold: Score below which queries use simple path (0.0-1.0)
        complex_threshold: Score above which queries use Bayesian path (0.0-1.0)
        enable_voi: Enable Value of Information-based agent selection
        enable_correlation: Enable correlation-aware consensus
        track_performance: Track performance metrics for simple vs Bayesian
        min_agents_for_bayesian: Minimum number of agents to trigger Bayesian
    """

    enabled: bool = True
    force_all: bool = False
    simple_threshold: float = 0.3
    complex_threshold: float = 0.7
    enable_voi: bool = True
    enable_correlation: bool = True
    track_performance: bool = True
    min_agents_for_bayesian: int = 2

    @classmethod
    def from_cli_flags(
        cls,
        *,
        enable_bayesian: bool = True,
        force_bayesian: bool = False,
        simple_threshold: float = 0.3,
        complex_threshold: float = 0.7,
        enable_voi: bool = True,
        enable_correlation: bool = True,
        min_agents_for_bayesian: int = 2,
    ) -> "BayesianConfig":
        """Create BayesianConfig from CLI flags.

        Args:
            enable_bayesian: Enable Bayesian orchestration
            force_bayesian: Force all queries through Bayesian
            simple_threshold: Complexity threshold for simple path
            complex_threshold: Complexity threshold for Bayesian path
            enable_voi: Enable Value of Information
            enable_correlation: Enable correlation tracking
            min_agents_for_bayesian: Minimum agents for Bayesian

        Returns:
            BayesianConfig instance
        """
        return cls(
            enabled=enable_bayesian,
            force_all=force_bayesian,
            simple_threshold=simple_threshold,
            complex_threshold=complex_threshold,
            enable_voi=enable_voi,
            enable_correlation=enable_correlation,
            min_agents_for_bayesian=min_agents_for_bayesian,
        )
