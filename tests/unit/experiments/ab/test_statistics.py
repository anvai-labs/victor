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

"""Unit tests for A/B testing statistical analysis."""

import pytest

from victor.experiments.ab_testing.statistics import StatisticalAnalyzer


class TestCompareMeans:
    """Tests for compare_means (t-test)."""

    def test_compare_means_significant_difference(self):
        """Test t-test with significant difference."""
        analyzer = StatisticalAnalyzer()

        control = [10.0, 12.0, 11.0, 13.0, 12.0]
        treatment = [8.0, 9.0, 8.5, 9.5, 9.0]

        result = analyzer.compare_means(control, treatment, alpha=0.05)

        # Check result structure
        assert result["test"] == "t-test"
        assert "t_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "effect_size" in result
        assert "mean_difference" in result
        assert "confidence_interval" in result
        assert len(result["confidence_interval"]) == 2

        # Treatment has lower mean, should be significant
        assert result["treatment_mean"] < result["control_mean"]
        assert result["mean_difference"] < 0
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_compare_means_no_significant_difference(self):
        """Test t-test with no significant difference."""
        analyzer = StatisticalAnalyzer()

        control = [10.0, 11.0, 10.5, 10.8, 10.2]
        treatment = [10.1, 10.9, 10.6, 10.7, 10.3]

        result = analyzer.compare_means(control, treatment, alpha=0.05)

        # Should not be significant
        assert not result["significant"]  # numpy bool, use truthiness not identity
        assert result["p_value"] > 0.05

    def test_compare_means_insufficient_data_raises(self):
        """Test that insufficient data raises ValueError."""
        analyzer = StatisticalAnalyzer()

        with pytest.raises(ValueError, match="at least 2 samples"):
            analyzer.compare_means([10.0], [11.0])

    def test_compare_means_effect_size(self):
        """Test Cohen's d effect size calculation."""
        analyzer = StatisticalAnalyzer()

        control = [10.0] * 10
        treatment = [12.0] * 10

        result = analyzer.compare_means(control, treatment)

        # Effect size should be large (> 0.8)
        assert result["effect_size"] > 0.8


class TestCompareProportions:
    """Tests for compare_proportions (chi-square test)."""

    def test_compare_proportions_significant_difference(self):
        """Test chi-square test with significant difference."""
        analyzer = StatisticalAnalyzer()

        result = analyzer.compare_proportions(
            control_successes=80,
            control_total=100,
            treatment_successes=95,
            treatment_total=100,
        )

        # Check result structure
        assert result["test"] == "chi-square"
        assert "chi2_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "control_rate" in result
        assert "treatment_rate" in result
        assert "rate_difference" in result
        assert len(result["confidence_interval"]) == 2

        # Treatment has higher success rate, should be significant
        assert result["treatment_rate"] > result["control_rate"]
        assert result["rate_difference"] > 0
        assert result["significant"]  # numpy bool, use truthiness not identity
        assert result["p_value"] < 0.05

    def test_compare_proportions_no_significant_difference(self):
        """Test chi-square test with no significant difference."""
        analyzer = StatisticalAnalyzer()

        result = analyzer.compare_proportions(
            control_successes=80,
            control_total=100,
            treatment_successes=82,
            treatment_total=100,
        )

        # Should not be significant
        assert not result["significant"]  # numpy bool, use truthiness not identity
        assert result["p_value"] > 0.05

    def test_compare_proportions_insufficient_data_raises(self):
        """Test that insufficient data raises ValueError."""
        analyzer = StatisticalAnalyzer()

        with pytest.raises(ValueError, match="at least 1 sample"):
            analyzer.compare_proportions(
                control_successes=0,
                control_total=0,
                treatment_successes=1,
                treatment_total=1,
            )


class TestMannWhitney:
    """Tests for mann_whitney (Mann-Whitney U test)."""

    def test_mann_whitney_significant_difference(self):
        """Test Mann-Whitney U test with significant difference."""
        analyzer = StatisticalAnalyzer()

        control = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        treatment = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

        result = analyzer.mann_whitney(control, treatment, alpha=0.05)

        # Check result structure
        assert result["test"] == "mann-whitney-u"
        assert "u_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "effect_size" in result
        assert "control_median" in result
        assert "treatment_median" in result

        # Treatment has higher values, should be significant
        assert result["treatment_median"] > result["control_median"]
        assert result["significant"]  # numpy bool, use truthiness not identity
        assert result["p_value"] < 0.05

    def test_mann_whitney_insufficient_data_raises(self):
        """Test that insufficient data raises ValueError."""
        analyzer = StatisticalAnalyzer()

        with pytest.raises(ValueError, match="at least 1 sample"):
            analyzer.mann_whitney([], [1])


class TestCalculateConfidenceInterval:
    """Tests for calculate_confidence_interval."""

    def test_calculate_confidence_interval_mean(self):
        """Test confidence interval calculation for mean."""
        analyzer = StatisticalAnalyzer()

        data = [10.0, 12.0, 11.0, 13.0, 12.0]
        ci = analyzer.calculate_confidence_interval(data, confidence=0.95)

        # Should return tuple
        assert isinstance(ci, tuple)
        assert len(ci) == 2

        # Lower bound should be less than upper bound
        assert ci[0] < ci[1]

        # Mean should be within CI
        import statistics

        mean = statistics.mean(data)
        assert ci[0] <= mean <= ci[1]

    def test_calculate_confidence_interval_different_levels(self):
        """Test confidence intervals at different levels."""
        analyzer = StatisticalAnalyzer()

        data = [10.0, 12.0, 11.0, 13.0, 12.0, 11.5, 12.5]

        ci_95 = analyzer.calculate_confidence_interval(data, confidence=0.95)
        ci_99 = analyzer.calculate_confidence_interval(data, confidence=0.99)

        # 99% CI should be wider than 95% CI
        assert (ci_99[1] - ci_99[0]) > (ci_95[1] - ci_95[0])

    def test_calculate_confidence_interval_insufficient_data_raises(self):
        """Test that insufficient data raises ValueError."""
        analyzer = StatisticalAnalyzer()

        with pytest.raises(ValueError, match="at least 2 samples"):
            analyzer.calculate_confidence_interval([10.0])


class TestCalculateProportionCI:
    """Tests for calculate_proportion_ci."""

    def test_calculate_proportion_ci(self):
        """Test confidence interval for proportion."""
        analyzer = StatisticalAnalyzer()

        ci = analyzer.calculate_proportion_ci(successes=80, total=100, confidence=0.95)

        # Should return tuple
        assert isinstance(ci, tuple)
        assert len(ci) == 2

        # Lower bound should be less than upper bound
        assert ci[0] < ci[1]

        # Proportion should be within CI
        proportion = 80 / 100
        assert ci[0] <= proportion <= ci[1]

    def test_calculate_proportion_ci_edge_cases(self):
        """Test proportion CI with edge cases."""
        analyzer = StatisticalAnalyzer()

        # All successes
        ci = analyzer.calculate_proportion_ci(successes=100, total=100)
        assert ci[0] > 0.95  # Should be very high

        # No successes
        ci = analyzer.calculate_proportion_ci(successes=0, total=100)
        assert ci[1] < 0.05  # Should be very low

    def test_calculate_proportion_ci_zero_total_raises(self):
        """Test that zero total raises ValueError."""
        analyzer = StatisticalAnalyzer()

        with pytest.raises(ValueError, match="must be > 0"):
            analyzer.calculate_proportion_ci(successes=0, total=0)


class TestIsSignificant:
    """Tests for is_significant."""

    def test_is_significant_true(self):
        """Test is_significant returns True for small p-value."""
        analyzer = StatisticalAnalyzer()

        assert analyzer.is_significant(0.01, alpha=0.05) is True
        assert analyzer.is_significant(0.001, alpha=0.05) is True

    def test_is_significant_false(self):
        """Test is_significant returns False for large p-value."""
        analyzer = StatisticalAnalyzer()

        assert analyzer.is_significant(0.10, alpha=0.05) is False
        assert analyzer.is_significant(0.50, alpha=0.05) is False

    def test_is_significant_boundary(self):
        """Test is_significant at boundary."""
        analyzer = StatisticalAnalyzer()

        # Exactly at alpha should be False (not < alpha)
        assert analyzer.is_significant(0.05, alpha=0.05) is False


class TestDetermineWinner:
    """Tests for determine_winner."""

    def test_determine_winner_with_significant_result(self):
        """Test winner determination with significant result."""
        analyzer = StatisticalAnalyzer()

        variant_metrics = {
            "control": [10.0, 11.0, 12.0, 10.5, 11.5],
            "treatment": [8.0, 9.0, 8.5, 9.5, 9.0],
        }

        result = analyzer.determine_winner(
            variant_metrics, optimization_goal="minimize", alpha=0.05
        )

        # Check result structure
        assert "winner" in result
        assert "confidence" in result
        assert "significant" in result
        assert "p_values" in result
        assert "recommendation" in result

        # Treatment should win (lower is better)
        assert result["winner"] == "treatment"
        assert result["significant"] is True
        assert result["recommendation"] == "deploy_winner"

    def test_determine_winner_maximize(self):
        """Test winner determination with maximize goal."""
        analyzer = StatisticalAnalyzer()

        variant_metrics = {
            "control": [10.0, 11.0, 12.0, 10.5, 11.5],
            "treatment": [15.0, 16.0, 17.0, 15.5, 16.5],
        }

        result = analyzer.determine_winner(
            variant_metrics, optimization_goal="maximize", alpha=0.05
        )

        # Treatment should win (higher is better)
        assert result["winner"] == "treatment"

    def test_determine_winner_insufficient_variants_raises(self):
        """Test that insufficient variants raises ValueError."""
        analyzer = StatisticalAnalyzer()

        with pytest.raises(ValueError, match="at least 2 variants"):
            analyzer.determine_winner({"control": [1, 2, 3]})


class TestSampleSizeCalculations:
    """Tests for sample size calculations."""

    def test_calculate_sample_size(self):
        """Test sample size calculation."""
        analyzer = StatisticalAnalyzer()

        try:
            from statsmodels.stats.power import tt_ind_solve_power

            # Medium effect size
            n = analyzer.calculate_sample_size(effect_size=0.5, alpha=0.05, power=0.8)

            # Should be positive
            assert n > 0

            # Should be reasonable (around 100-200 for medium effect)
            assert 50 < n < 300

        except ImportError:
            pytest.skip("statsmodels not available")

    def test_minimum_detectable_effect(self):
        """Test minimum detectable effect calculation."""
        analyzer = StatisticalAnalyzer()

        try:
            from statsmodels.stats.power import tt_ind_solve_power

            # With 100 samples per variant
            mde = analyzer.minimum_detectable_effect(sample_size=100, alpha=0.05, power=0.8)

            # Should be positive
            assert mde > 0

            # Should be reasonable (medium-large effect)
            assert 0.2 < mde < 1.0

        except ImportError:
            pytest.skip("statsmodels not available")
