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

"""Unit tests for statistical_tests module.

Tests the A/B experiment statistical testing utilities.
"""

import pytest
import math

from victor.framework.rl.statistical_tests import (
    SignificanceLevel,
    StatisticalResult,
    welch_t_test,
    two_proportion_z_test,
    chi_squared_test,
    mann_whitney_u_test,
    compute_sample_size,
    compute_power,
    _normal_cdf,
    _normal_ppf,
    _t_cdf,
)


class TestStatisticalResult:
    """Tests for StatisticalResult dataclass."""

    def test_creation(self) -> None:
        """Test creating result."""
        result = StatisticalResult(
            test_name="t-test",
            statistic=2.5,
            p_value=0.01,
            significant=True,
            effect_size=0.5,
        )

        assert result.test_name == "t-test"
        assert result.statistic == 2.5
        assert result.p_value == 0.01
        assert result.significant is True

    def test_defaults(self) -> None:
        """Test default values."""
        result = StatisticalResult(
            test_name="test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
        )

        assert result.effect_size == 0.0
        assert result.confidence_interval == (0.0, 0.0)
        assert result.power is None
        assert result.recommendation == ""


class TestSignificanceLevel:
    """Tests for SignificanceLevel enum."""

    def test_values(self) -> None:
        """Test significance level values."""
        assert SignificanceLevel.HIGH == 0.01
        assert SignificanceLevel.STANDARD == 0.05
        assert SignificanceLevel.LOW == 0.10


class TestNormalCDF:
    """Tests for normal CDF approximation."""

    def test_standard_values(self) -> None:
        """Test known standard normal CDF values."""
        # P(Z <= 0) = 0.5
        assert abs(_normal_cdf(0.0) - 0.5) < 0.001

        # P(Z <= 1.96) ≈ 0.975
        assert abs(_normal_cdf(1.96) - 0.975) < 0.01

        # P(Z <= -1.96) ≈ 0.025
        assert abs(_normal_cdf(-1.96) - 0.025) < 0.01

    def test_extreme_values(self) -> None:
        """Test extreme values."""
        # Very large positive
        assert _normal_cdf(5.0) > 0.999

        # Very large negative
        assert _normal_cdf(-5.0) < 0.001


class TestNormalPPF:
    """Tests for normal inverse CDF."""

    def test_standard_values(self) -> None:
        """Test known inverse values."""
        # 50th percentile = 0
        assert abs(_normal_ppf(0.5) - 0.0) < 0.001

        # 97.5th percentile ≈ 1.96
        assert abs(_normal_ppf(0.975) - 1.96) < 0.05

        # 2.5th percentile ≈ -1.96
        assert abs(_normal_ppf(0.025) - (-1.96)) < 0.05

    def test_invalid_values(self) -> None:
        """Test invalid probability values."""
        with pytest.raises(ValueError):
            _normal_ppf(0.0)

        with pytest.raises(ValueError):
            _normal_ppf(1.0)

        with pytest.raises(ValueError):
            _normal_ppf(-0.1)


class TestTCDF:
    """Tests for t-distribution CDF."""

    def test_large_df(self) -> None:
        """Test t-CDF with large df approaches normal."""
        # With large df, t-distribution approaches normal
        t_value = 1.96
        df = 100

        t_prob = _t_cdf(t_value, df)
        normal_prob = _normal_cdf(t_value)

        assert abs(t_prob - normal_prob) < 0.01

    def test_symmetry(self) -> None:
        """Test t-CDF symmetry."""
        df = 10
        t_value = 2.0

        # P(T <= -t) = 1 - P(T <= t) for symmetric distribution
        left = _t_cdf(-t_value, df)
        right = _t_cdf(t_value, df)

        assert abs(left + right - 1.0) < 0.05


class TestWelchTTest:
    """Tests for Welch's t-test."""

    def test_identical_samples(self) -> None:
        """Test t-test with identical samples."""
        result = welch_t_test(
            mean1=10.0,
            std1=2.0,
            n1=30,
            mean2=10.0,
            std2=2.0,
            n2=30,
        )

        assert result.test_name == "Welch's t-test"
        assert result.statistic == 0.0
        assert result.significant is False
        assert result.effect_size == 0.0

    def test_significant_difference(self) -> None:
        """Test t-test with significant difference."""
        result = welch_t_test(
            mean1=10.0,
            std1=2.0,
            n1=100,
            mean2=12.0,
            std2=2.0,
            n2=100,
        )

        assert result.significant is True
        assert result.p_value < 0.05
        assert result.effect_size > 0  # Positive effect

    def test_insufficient_samples(self) -> None:
        """Test t-test with insufficient samples."""
        result = welch_t_test(
            mean1=10.0,
            std1=2.0,
            n1=1,
            mean2=12.0,
            std2=2.0,
            n2=30,
        )

        assert result.significant is False
        assert "Insufficient" in result.recommendation

    def test_zero_variance(self) -> None:
        """Test t-test with zero variance."""
        result = welch_t_test(
            mean1=10.0,
            std1=0.0,
            n1=30,
            mean2=10.0,
            std2=0.0,
            n2=30,
        )

        assert result.significant is False

    def test_effect_size_interpretation(self) -> None:
        """Test effect size recommendations."""
        # Large effect
        result = welch_t_test(
            mean1=10.0,
            std1=1.0,
            n1=50,
            mean2=11.5,
            std2=1.0,
            n2=50,
        )
        assert "Large" in result.recommendation or "Strong" in result.recommendation

    def test_confidence_interval(self) -> None:
        """Test confidence interval calculation."""
        result = welch_t_test(
            mean1=10.0,
            std1=2.0,
            n1=50,
            mean2=12.0,
            std2=2.0,
            n2=50,
        )

        ci_low, ci_high = result.confidence_interval
        actual_diff = 12.0 - 10.0

        # Actual difference should be within CI
        assert ci_low <= actual_diff <= ci_high


class TestTwoProportionZTest:
    """Tests for two-proportion z-test."""

    def test_identical_proportions(self) -> None:
        """Test z-test with identical proportions."""
        result = two_proportion_z_test(
            successes1=50,
            n1=100,
            successes2=50,
            n2=100,
        )

        assert result.test_name == "Two-proportion z-test"
        assert result.significant is False
        assert result.effect_size == 0.0

    def test_significant_difference(self) -> None:
        """Test z-test with significant difference."""
        result = two_proportion_z_test(
            successes1=50,
            n1=200,
            successes2=70,
            n2=200,
        )

        assert result.significant is True
        assert result.effect_size > 0  # Treatment is better

    def test_insufficient_samples(self) -> None:
        """Test z-test with no samples."""
        result = two_proportion_z_test(
            successes1=0,
            n1=0,
            successes2=5,
            n2=10,
        )

        assert result.significant is False
        assert "Insufficient" in result.recommendation

    def test_all_successes_or_failures(self) -> None:
        """Test z-test with all successes or failures."""
        result = two_proportion_z_test(
            successes1=100,
            n1=100,
            successes2=100,
            n2=100,
        )

        assert "Zero variance" in result.recommendation or result.significant is False

    def test_relative_lift(self) -> None:
        """Test relative lift calculation."""
        result = two_proportion_z_test(
            successes1=50,
            n1=100,  # 50%
            successes2=60,
            n2=100,  # 60%
        )

        # 20% relative lift expected (60-50)/50 = 0.2
        expected_lift = 0.2
        assert abs(result.effect_size - expected_lift) < 0.01


class TestChiSquaredTest:
    """Tests for chi-squared test."""

    def test_independent_variables(self) -> None:
        """Test with independent variables."""
        # Proportional distribution
        observed = [
            [50, 50],
            [50, 50],
        ]

        result = chi_squared_test(observed)

        assert result.test_name == "Chi-squared test"
        assert result.significant is False

    def test_dependent_variables(self) -> None:
        """Test with dependent variables."""
        # Clear association
        observed = [
            [90, 10],
            [10, 90],
        ]

        result = chi_squared_test(observed)

        assert result.significant is True
        assert result.effect_size > 0.3  # Moderate+ association

    def test_empty_table(self) -> None:
        """Test with empty table."""
        result = chi_squared_test([])
        assert result.significant is False
        assert "Empty" in result.recommendation

    def test_no_observations(self) -> None:
        """Test with all zeros."""
        observed = [
            [0, 0],
            [0, 0],
        ]

        result = chi_squared_test(observed)
        assert result.significant is False

    def test_cramers_v(self) -> None:
        """Test Cramer's V effect size."""
        # Strong association
        observed = [
            [95, 5],
            [5, 95],
        ]

        result = chi_squared_test(observed)
        assert result.effect_size > 0.5  # Strong association


class TestMannWhitneyU:
    """Tests for Mann-Whitney U test."""

    def test_identical_samples(self) -> None:
        """Test with identical samples."""
        sample = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = mann_whitney_u_test(sample, sample.copy())

        assert result.test_name == "Mann-Whitney U test"
        assert result.significant is False

    def test_different_distributions(self) -> None:
        """Test with clearly different distributions."""
        sample1 = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
        sample2 = [6.0, 7.0, 8.0, 9.0, 10.0] * 10

        result = mann_whitney_u_test(sample1, sample2)

        assert result.significant is True

    def test_insufficient_samples(self) -> None:
        """Test with no samples."""
        result = mann_whitney_u_test([], [1.0, 2.0])
        assert result.significant is False
        assert "Insufficient" in result.recommendation

    def test_effect_size_interpretation(self) -> None:
        """Test effect size bounds."""
        sample1 = [1.0, 2.0, 3.0]
        sample2 = [4.0, 5.0, 6.0]

        result = mann_whitney_u_test(sample1, sample2)

        # Effect size should be between 0 and 1
        assert 0.0 <= result.effect_size <= 1.0


class TestComputeSampleSize:
    """Tests for sample size calculation."""

    def test_standard_calculation(self) -> None:
        """Test standard sample size calculation."""
        n = compute_sample_size(
            baseline_rate=0.5,
            min_detectable_effect=0.1,
            power=0.8,
            significance_level=0.05,
        )

        assert n > 0
        assert isinstance(n, int)

    def test_smaller_effect_needs_more_samples(self) -> None:
        """Test that smaller effects need more samples."""
        n_small = compute_sample_size(
            baseline_rate=0.5,
            min_detectable_effect=0.05,
        )
        n_large = compute_sample_size(
            baseline_rate=0.5,
            min_detectable_effect=0.2,
        )

        assert n_small > n_large

    def test_higher_power_needs_more_samples(self) -> None:
        """Test that higher power needs more samples."""
        n_80 = compute_sample_size(
            baseline_rate=0.5,
            min_detectable_effect=0.1,
            power=0.8,
        )
        n_95 = compute_sample_size(
            baseline_rate=0.5,
            min_detectable_effect=0.1,
            power=0.95,
        )

        assert n_95 > n_80

    def test_zero_effect(self) -> None:
        """Test with zero effect size."""
        n = compute_sample_size(
            baseline_rate=0.5,
            min_detectable_effect=0.0,
        )

        # Should return fallback
        assert n >= 10


class TestComputePower:
    """Tests for power calculation."""

    def test_standard_calculation(self) -> None:
        """Test standard power calculation."""
        power = compute_power(
            n=1000,
            baseline_rate=0.5,
            effect_size=0.1,
        )

        assert 0.0 <= power <= 1.0

    def test_more_samples_more_power(self) -> None:
        """Test that more samples increase power."""
        power_small = compute_power(
            n=100,
            baseline_rate=0.5,
            effect_size=0.1,
        )
        power_large = compute_power(
            n=1000,
            baseline_rate=0.5,
            effect_size=0.1,
        )

        assert power_large > power_small

    def test_larger_effect_more_power(self) -> None:
        """Test that larger effects increase power."""
        power_small = compute_power(
            n=500,
            baseline_rate=0.5,
            effect_size=0.05,
        )
        power_large = compute_power(
            n=500,
            baseline_rate=0.5,
            effect_size=0.2,
        )

        assert power_large > power_small

    def test_zero_variance(self) -> None:
        """Test with zero variance case."""
        power = compute_power(
            n=100,
            baseline_rate=1.0,  # All successes
            effect_size=0.1,
        )

        # Should handle gracefully
        assert 0.0 <= power <= 1.0


class TestIntegration:
    """Integration tests for statistical testing workflow."""

    def test_ab_test_workflow(self) -> None:
        """Test complete A/B test workflow."""
        # 1. Calculate required sample size
        n = compute_sample_size(
            baseline_rate=0.2,
            min_detectable_effect=0.25,  # 25% relative improvement
            power=0.8,
        )

        assert n > 0

        # 2. Check power at calculated sample size
        power = compute_power(
            n=n,
            baseline_rate=0.2,
            effect_size=0.25,
        )

        # Should be close to 0.8
        assert power >= 0.7

        # 3. Run test with simulated data
        result = two_proportion_z_test(
            successes1=int(0.2 * n),  # 20% baseline
            n1=n,
            successes2=int(0.25 * n),  # 25% treatment
            n2=n,
        )

        # Result should include recommendation
        assert len(result.recommendation) > 0

    def test_multiple_test_comparison(self) -> None:
        """Test comparing parametric and non-parametric tests."""
        # Generate samples
        control = [10.0 + i * 0.1 for i in range(50)]
        treatment = [12.0 + i * 0.1 for i in range(50)]

        # Parametric test
        t_result = welch_t_test(
            mean1=sum(control) / len(control),
            std1=2.0,
            n1=len(control),
            mean2=sum(treatment) / len(treatment),
            std2=2.0,
            n2=len(treatment),
        )

        # Non-parametric test
        mw_result = mann_whitney_u_test(control, treatment)

        # Both should detect the difference
        assert t_result.significant is True
        assert mw_result.significant is True
