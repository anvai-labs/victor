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

"""Statistical analysis for A/B testing.

This module provides statistical tests and analysis methods for comparing
experiment variants.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

# scipy is heavy (~180ms to import) and ships in the optional [ml] extra.
# Probe availability without importing (find_spec) so it stays out of the CLI
# cold-start path; the real import is deferred to first use via the ``stats``
# proxy below. numpy is comparatively cheap and used pervasively, so it stays
# eager.
import importlib.util

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


def _require_scipy() -> None:
    """Fail with an actionable message when statistical tests need scipy."""
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for A/B statistical analysis. "
            'Install it with: pip install "victor-ai[ml]"'
        )


class _LazyStats:
    """Lazy proxy for ``scipy.stats``; imports scipy on first attribute access.

    Each ``stats.<name>`` access first runs ``_require_scipy()`` (actionable
    error when scipy is absent) then resolves the attribute from the real
    module. This also guards methods that historically used ``stats`` without a
    preceding ``_require_scipy()`` call.
    """

    def __getattr__(self, name: str) -> Any:
        _require_scipy()
        from scipy import stats

        return getattr(stats, name)


stats: Any = _LazyStats()


# =============================================================================
# Enums for A/B Testing Optimization
# =============================================================================


class OptimizationGoal(str, Enum):
    """Optimization goal for A/B testing metrics.

    Determines whether higher or lower values are better for the metric
    being optimized.
    """

    MINIMIZE = "minimize"  # Lower values are better (e.g., error rate, latency)
    MAXIMIZE = "maximize"  # Higher values are better (e.g., conversion, revenue)


# Backward compatibility
_OptimizationGoalLiteral = Literal["minimize", "maximize"]


class StatisticalAnalyzer:
    """Statistical analysis for A/B experiments.

    This class provides methods for comparing variants using various
    statistical tests.

    Usage:
        analyzer = StatisticalAnalyzer()

        # Compare means (t-test)
        result = analyzer.compare_means(control_data, treatment_data, alpha=0.05)

        # Compare proportions (chi-square)
        result = analyzer.compare_proportions(
            control_successes=80,
            control_total=100,
            treatment_successes=90,
            treatment_total=100,
        )
    """

    def __init__(self) -> None:
        _require_scipy()

    def compare_means(
        self,
        control: List[float],
        treatment: List[float],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Perform two-sample t-test to compare means.

        Tests whether the mean of treatment is different from control.

        H0: control_mean == treatment_mean
        H1: control_mean != treatment_mean

        Args:
            control: Control group data
            treatment: Treatment group data
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with test results including:
            - test: Test name ("t-test")
            - t_statistic: T-statistic
            - p_value: P-value
            - significant: Whether result is significant
            - effect_size: Cohen's d
            - mean_difference: Difference in means
            - confidence_interval: 95% CI for difference
            - control_mean: Control group mean
            - treatment_mean: Treatment group mean

        Raises:
            ValueError: If data is insufficient
        """
        if len(control) < 2 or len(treatment) < 2:
            raise ValueError("Both groups must have at least 2 samples")

        # Convert to arrays
        control_arr = np.array(control)
        treatment_arr = np.array(treatment)

        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(treatment_arr, control_arr)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            (
                (len(control_arr) - 1) * np.var(control_arr, ddof=1)
                + (len(treatment_arr) - 1) * np.var(treatment_arr, ddof=1)
            )
            / (len(control_arr) + len(treatment_arr) - 2)
        )
        cohens_d = (np.mean(treatment_arr) - np.mean(control_arr)) / pooled_std

        # Confidence interval for difference
        diff = np.mean(treatment_arr) - np.mean(control_arr)
        se_diff = np.sqrt(
            np.var(control_arr, ddof=1) / len(control_arr)
            + np.var(treatment_arr, ddof=1) / len(treatment_arr)
        )
        ci = (
            float(
                diff
                - stats.t.ppf(1 - alpha / 2, len(control_arr) + len(treatment_arr) - 2) * se_diff
            ),
            float(
                diff
                + stats.t.ppf(1 - alpha / 2, len(control_arr) + len(treatment_arr) - 2) * se_diff
            ),
        )

        # Determine significance
        significant = bool(p_value < alpha)

        return {
            "test": "t-test",
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
            "significant": significant,
            "effect_size": float(cohens_d),
            "mean_difference": float(diff),
            "confidence_interval": ci,
            "control_mean": float(np.mean(control_arr)),
            "treatment_mean": float(np.mean(treatment_arr)),
        }

    def compare_proportions(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Perform chi-square test to compare proportions.

        Tests whether the proportion of successes is different between groups.

        H0: p_control == p_treatment
        H1: p_control != p_treatment

        Args:
            control_successes: Number of successes in control
            control_total: Total samples in control
            treatment_successes: Number of successes in treatment
            treatment_total: Total samples in treatment
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with test results including:
            - test: Test name ("chi-square")
            - chi2_statistic: Chi-square statistic
            - p_value: P-value
            - significant: Whether result is significant
            - control_rate: Control group success rate
            - treatment_rate: Treatment group success rate
            - rate_difference: Difference in rates
            - confidence_interval: 95% CI for rate difference

        Raises:
            ValueError: If data is insufficient
        """
        if control_total < 1 or treatment_total < 1:
            raise ValueError("Both groups must have at least 1 sample")

        # Create contingency table
        observed = [
            [treatment_successes, treatment_total - treatment_successes],
            [control_successes, control_total - control_successes],
        ]

        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)

        # Calculate proportions
        control_rate = control_successes / control_total
        treatment_rate = treatment_successes / treatment_total
        rate_diff = treatment_rate - control_rate

        # Confidence interval for rate difference
        se_rate = np.sqrt(
            (control_rate * (1 - control_rate) / control_total)
            + (treatment_rate * (1 - treatment_rate) / treatment_total)
        )
        z = stats.norm.ppf(1 - alpha / 2)
        ci = (float(rate_diff - z * se_rate), float(rate_diff + z * se_rate))

        # Determine significance
        significant = p_value < alpha

        return {
            "test": "chi-square",
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "significant": significant,
            "control_rate": float(control_rate),
            "treatment_rate": float(treatment_rate),
            "rate_difference": float(rate_diff),
            "confidence_interval": ci,
        }

    def mann_whitney(
        self,
        control: List[float],
        treatment: List[float],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Perform Mann-Whitney U test (non-parametric).

        Tests whether distributions are different. Use when normality
        assumption fails.

        H0: Distributions are equal
        H1: Distributions are different

        Args:
            control: Control group data
            treatment: Treatment group data
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with test results including:
            - test: Test name ("mann-whitney-u")
            - u_statistic: U statistic
            - p_value: P-value
            - significant: Whether result is significant
            - effect_size: Rank-biserial correlation
            - control_median: Control group median
            - treatment_median: Treatment group median

        Raises:
            ValueError: If data is insufficient
        """
        if len(control) < 1 or len(treatment) < 1:
            raise ValueError("Both groups must have at least 1 sample")

        # Perform Mann-Whitney U test
        u_statistic, p_value = stats.mannwhitneyu(
            treatment,
            control,
            alternative="two-sided",
        )

        # Calculate rank-biserial correlation (effect size)
        n1 = len(treatment)
        n2 = len(control)
        rank_biserial = 1 - (2 * u_statistic) / (n1 * n2)

        # Determine significance
        significant = p_value < alpha

        return {
            "test": "mann-whitney-u",
            "u_statistic": float(u_statistic),
            "p_value": float(p_value),
            "significant": significant,
            "effect_size": float(rank_biserial),
            "control_median": float(np.median(control)),
            "treatment_median": float(np.median(treatment)),
        }

    def calculate_confidence_interval(
        self,
        data: List[float],
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Calculate confidence interval for mean.

        Args:
            data: Sample data
            confidence: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)

        Raises:
            ValueError: If data is insufficient
        """
        if len(data) < 2:
            raise ValueError("Must have at least 2 samples")

        arr = np.array(data)
        n = len(arr)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        se = std / np.sqrt(n)

        # Calculate t-value for confidence level
        alpha = 1 - confidence
        t_value = stats.t.ppf(1 - alpha / 2, n - 1)

        # Calculate margin of error
        margin = t_value * se

        return (float(mean - margin), float(mean + margin))

    def calculate_proportion_ci(
        self,
        successes: int,
        total: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Calculate confidence interval for proportion (Wilson score).

        Args:
            successes: Number of successes
            total: Total sample size
            confidence: Confidence level

        Returns:
            Tuple of (lower_bound, upper_bound)

        Raises:
            ValueError: If total is 0
        """
        if total == 0:
            raise ValueError("Total sample size must be > 0")

        p = successes / total
        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator

        return (float(center - margin), float(center + margin))

    def is_significant(
        self,
        p_value: float,
        alpha: float = 0.05,
    ) -> bool:
        """Check if p-value indicates statistical significance.

        Args:
            p_value: P-value from statistical test
            alpha: Significance level (default: 0.05)

        Returns:
            True if significant, False otherwise
        """
        return p_value < alpha

    def calculate_sample_size(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.8,
        ratio: float = 1.0,
    ) -> int:
        """Calculate required sample size per variant.

        Args:
            effect_size: Expected effect size (Cohen's d)
                - Small: 0.2
                - Medium: 0.5
                - Large: 0.8
            alpha: Significance level (Type I error rate)
            power: Statistical power (1 - Type II error rate)
            ratio: Ratio of treatment to control sample size

        Returns:
            Required sample size per variant

        Raises:
            ImportError: If statsmodels is not available
        """
        try:
            from statsmodels.stats.power import tt_ind_solve_power
        except ImportError:
            raise ImportError(
                "statsmodels is required for sample size calculation. "
                "Install it with: pip install statsmodels"
            )

        # Calculate sample size
        n = tt_ind_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=ratio,
            alternative="two-sided",
        )

        return int(np.ceil(n))

    def minimum_detectable_effect(
        self,
        sample_size: int,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> float:
        """Calculate minimum detectable effect size.

        Args:
            sample_size: Sample size per variant
            alpha: Significance level
            power: Statistical power

        Returns:
            Minimum detectable effect size (Cohen's d)

        Raises:
            ImportError: If statsmodels is not available
        """
        try:
            from statsmodels.stats.power import tt_ind_solve_power
        except ImportError:
            raise ImportError(
                "statsmodels is required for MDE calculation. "
                "Install it with: pip install statsmodels"
            )

        effect_size = tt_ind_solve_power(
            nobs1=sample_size,
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative="two-sided",
        )

        return float(effect_size)

    def determine_winner(
        self,
        variant_metrics: Dict[str, List[float]],
        optimization_goal: OptimizationGoal = OptimizationGoal.MAXIMIZE,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Determine winning variant using statistical tests.

        Args:
            variant_metrics: Dictionary mapping variant_id to metric values
            optimization_goal: Whether to minimize or maximize the metric
            alpha: Significance level

        Returns:
            Dictionary with:
            - winner: Winning variant_id
            - confidence: Confidence in winner
            - significant: Whether result is statistically significant
            - p_values: Pairwise p-values
            - recommendation: "deploy_winner", "continue", or "inconclusive"

        Raises:
            ValueError: If insufficient data
        """
        if len(variant_metrics) < 2:
            raise ValueError("Need at least 2 variants to determine winner")

        variant_ids = list(variant_metrics.keys())

        # Find control variant (first variant)
        control_id = variant_ids[0]
        control_data = variant_metrics[control_id]

        # Compare each treatment against control
        p_values = {}
        improvements = {}

        for treatment_id in variant_ids[1:]:
            treatment_data = variant_metrics[treatment_id]

            # Perform t-test
            result = self.compare_means(control_data, treatment_data, alpha=alpha)
            p_values[treatment_id] = result["p_value"]
            improvements[treatment_id] = result["mean_difference"]

        # Find best variant
        if optimization_goal == OptimizationGoal.MAXIMIZE:
            # Higher is better
            best_id = max(variant_ids, key=lambda k: np.mean(variant_metrics[k]))
        else:
            # Lower is better
            best_id = min(variant_ids, key=lambda k: np.mean(variant_metrics[k]))

        # Check if best is significantly better than control
        if best_id == control_id:
            # Control is best - check if significantly better than all treatments
            all_not_significant = all(not self.is_significant(p, alpha) for p in p_values.values())
            significant = all_not_significant
        else:
            # Treatment is best - check if significantly better than control
            significant = self.is_significant(p_values[best_id], alpha)

        # Calculate confidence
        if significant:
            confidence = 1 - alpha
        else:
            confidence = None

        # Make recommendation
        if significant:
            recommendation = "deploy_winner"
        elif len(variant_metrics) < 3:
            # Only 2 variants and not significant
            recommendation = "continue"
        else:
            # Multiple variants - might need more testing
            recommendation = "inconclusive"

        return {
            "winner": best_id,
            "confidence": confidence,
            "significant": significant,
            "p_values": p_values,
            "improvements": improvements,
            "recommendation": recommendation,
        }
