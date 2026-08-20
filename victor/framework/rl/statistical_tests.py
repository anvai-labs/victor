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

"""Statistical tests for A/B experiment analysis.

This module provides statistical testing utilities for evaluating
A/B experiments in the RL framework without external dependencies.

Supported Tests:
- Two-sample t-test (Welch's)
- Two-proportion z-test
- Chi-squared test for independence
- Mann-Whitney U test (non-parametric)
- Effect size calculations (Cohen's d, relative lift)

All implementations use only Python standard library (math module)
to avoid adding dependencies.

Sprint 5: Advanced RL Patterns
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class SignificanceLevel(float, Enum):
    """Common significance levels."""

    HIGH = 0.01  # 99% confidence
    STANDARD = 0.05  # 95% confidence
    LOW = 0.10  # 90% confidence


@dataclass
class StatisticalResult:
    """Result of a statistical test.

    Attributes:
        test_name: Name of the test performed
        statistic: Test statistic value
        p_value: Probability of observing result under null hypothesis
        significant: Whether result is statistically significant
        effect_size: Standardized effect size (e.g., Cohen's d)
        confidence_interval: 95% confidence interval for effect
        power: Statistical power (if calculable)
        recommendation: Human-readable recommendation
    """

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    power: Optional[float] = None
    recommendation: str = ""


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using error function approximation.

    Uses Abramowitz and Stegun approximation (maximum error 1.5e-7).

    Args:
        x: Value to compute CDF for

    Returns:
        P(Z <= x) for standard normal Z
    """
    # Constants for approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # Save the sign
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2)

    # Approximation of erf
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return 0.5 * (1.0 + sign * y)


def _normal_ppf(p: float) -> float:
    """Approximate inverse standard normal CDF (percent point function).

    Uses Rational approximation from Abramowitz and Stegun.

    Args:
        p: Probability value (0 < p < 1)

    Returns:
        z such that P(Z <= z) = p
    """
    if p <= 0 or p >= 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")

    # Coefficients
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )


def _t_cdf(t: float, df: float) -> float:
    """Approximate Student's t CDF using normal approximation for large df.

    For df > 30, uses normal approximation. For smaller df, uses
    a series approximation.

    Args:
        t: t-statistic value
        df: Degrees of freedom

    Returns:
        P(T <= t) for Student's t distribution
    """
    if df > 30:
        # Normal approximation for large df
        return _normal_cdf(t)

    # For smaller df, use Beta function approximation
    x = df / (df + t * t)
    if t < 0:
        return 0.5 * _incomplete_beta(df / 2, 0.5, x)
    else:
        return 1 - 0.5 * _incomplete_beta(df / 2, 0.5, x)


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Approximate regularized incomplete beta function.

    Uses continued fraction expansion.

    Args:
        a: First shape parameter
        b: Second shape parameter
        x: Value (0 <= x <= 1)

    Returns:
        I_x(a, b) regularized incomplete beta function
    """
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0

    # Use continued fraction
    max_iter = 100
    eps = 1e-10

    # Compute using Lentz's algorithm
    qab = a + b
    qap = a + 1
    qam = a - 1

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < eps:
            break

    # Compute the prefactor
    try:
        log_prefactor = (
            a * math.log(x)
            + b * math.log(1 - x)
            + math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
        )
        prefactor = math.exp(log_prefactor)
    except (ValueError, OverflowError):
        prefactor = 0.0

    return prefactor * h / a


def welch_t_test(
    mean1: float,
    std1: float,
    n1: int,
    mean2: float,
    std2: float,
    n2: int,
    significance_level: float = 0.05,
) -> StatisticalResult:
    """Perform Welch's two-sample t-test.

    Tests whether two samples have different means, not assuming
    equal variances (more robust than Student's t-test).

    Args:
        mean1: Mean of first sample (control)
        std1: Standard deviation of first sample
        n1: Size of first sample
        mean2: Mean of second sample (treatment)
        std2: Standard deviation of second sample
        n2: Size of second sample
        significance_level: Alpha level for significance

    Returns:
        StatisticalResult with test outcome
    """
    if n1 < 2 or n2 < 2:
        return StatisticalResult(
            test_name="Welch's t-test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            recommendation="Insufficient samples (need n >= 2 for each group)",
        )

    # Compute standard error
    se1 = (std1**2) / n1
    se2 = (std2**2) / n2
    se_diff = math.sqrt(se1 + se2)

    if se_diff == 0:
        return StatisticalResult(
            test_name="Welch's t-test",
            statistic=0.0,
            p_value=1.0 if mean1 == mean2 else 0.0,
            significant=mean1 != mean2,
            recommendation="Zero variance in both groups",
        )

    # Compute t-statistic
    t_stat = (mean2 - mean1) / se_diff

    # Welch-Satterthwaite degrees of freedom
    df_num = (se1 + se2) ** 2
    df_denom = (se1**2) / (n1 - 1) + (se2**2) / (n2 - 1)
    df = df_num / df_denom if df_denom > 0 else n1 + n2 - 2

    # Compute two-tailed p-value
    p_value = 2 * (1 - _t_cdf(abs(t_stat), df))

    # Compute Cohen's d effect size
    pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0.0

    # Compute confidence interval for difference
    t_crit = _normal_ppf(1 - significance_level / 2)  # Approximate
    margin = t_crit * se_diff
    ci = (mean2 - mean1 - margin, mean2 - mean1 + margin)

    significant = p_value < significance_level

    # Generate recommendation
    if not significant:
        rec = "No significant difference detected. Continue experiment or accept null hypothesis."
    elif cohens_d > 0.8:
        rec = f"Large positive effect (d={cohens_d:.2f}). Strong evidence to adopt treatment."
    elif cohens_d > 0.5:
        rec = f"Medium positive effect (d={cohens_d:.2f}). Consider adopting treatment."
    elif cohens_d > 0.2:
        rec = f"Small positive effect (d={cohens_d:.2f}). Marginal improvement, evaluate cost-benefit."
    elif cohens_d < -0.2:
        rec = f"Negative effect (d={cohens_d:.2f}). Treatment performs worse than control."
    else:
        rec = "Effect too small to be practically significant despite statistical significance."

    return StatisticalResult(
        test_name="Welch's t-test",
        statistic=t_stat,
        p_value=p_value,
        significant=significant,
        effect_size=cohens_d,
        confidence_interval=ci,
        recommendation=rec,
    )


def two_proportion_z_test(
    successes1: int,
    n1: int,
    successes2: int,
    n2: int,
    significance_level: float = 0.05,
) -> StatisticalResult:
    """Perform two-proportion z-test.

    Tests whether two proportions (success rates) are different.
    Useful for comparing conversion rates, success rates, etc.

    Args:
        successes1: Number of successes in control
        n1: Total trials in control
        successes2: Number of successes in treatment
        n2: Total trials in treatment
        significance_level: Alpha level for significance

    Returns:
        StatisticalResult with test outcome
    """
    if n1 < 1 or n2 < 1:
        return StatisticalResult(
            test_name="Two-proportion z-test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            recommendation="Insufficient samples",
        )

    p1 = successes1 / n1
    p2 = successes2 / n2

    # Pooled proportion under null hypothesis
    p_pooled = (successes1 + successes2) / (n1 + n2)

    # Standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

    if se == 0:
        return StatisticalResult(
            test_name="Two-proportion z-test",
            statistic=0.0,
            p_value=1.0 if p1 == p2 else 0.0,
            significant=p1 != p2,
            recommendation="Zero variance (all successes or all failures)",
        )

    # Z-statistic
    z_stat = (p2 - p1) / se

    # Two-tailed p-value
    p_value = 2 * (1 - _normal_cdf(abs(z_stat)))

    # Relative lift as effect size
    relative_lift = (p2 - p1) / p1 if p1 > 0 else float("inf") if p2 > p1 else 0.0

    # Confidence interval for difference
    se_diff = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_crit = _normal_ppf(1 - significance_level / 2)
    margin = z_crit * se_diff
    ci = (p2 - p1 - margin, p2 - p1 + margin)

    significant = p_value < significance_level

    # Generate recommendation
    if not significant:
        rec = f"No significant difference in proportions ({p1:.1%} vs {p2:.1%})."
    elif relative_lift > 0.1:
        rec = f"Significant improvement: {relative_lift:+.1%} lift ({p1:.1%} → {p2:.1%}). Recommend rollout."
    elif relative_lift > 0:
        rec = f"Small improvement: {relative_lift:+.1%} lift. Evaluate if worth complexity."
    else:
        rec = f"Treatment worse: {relative_lift:+.1%} ({p1:.1%} → {p2:.1%}). Keep control."

    return StatisticalResult(
        test_name="Two-proportion z-test",
        statistic=z_stat,
        p_value=p_value,
        significant=significant,
        effect_size=relative_lift,
        confidence_interval=ci,
        recommendation=rec,
    )


def chi_squared_test(
    observed: List[List[int]],
    significance_level: float = 0.05,
) -> StatisticalResult:
    """Perform chi-squared test for independence.

    Tests whether two categorical variables are independent.
    Useful for testing if variant assignment affects categorical outcomes.

    Args:
        observed: 2D contingency table (rows=variants, cols=outcomes)
        significance_level: Alpha level for significance

    Returns:
        StatisticalResult with test outcome
    """
    if not observed or not observed[0]:
        return StatisticalResult(
            test_name="Chi-squared test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            recommendation="Empty contingency table",
        )

    rows = len(observed)
    cols = len(observed[0])

    # Compute totals
    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[i][j] for i in range(rows)) for j in range(cols)]
    total = sum(row_totals)

    if total == 0:
        return StatisticalResult(
            test_name="Chi-squared test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            recommendation="No observations",
        )

    # Compute expected values and chi-squared statistic
    chi_sq = 0.0
    for i in range(rows):
        for j in range(cols):
            expected = row_totals[i] * col_totals[j] / total
            if expected > 0:
                chi_sq += (observed[i][j] - expected) ** 2 / expected

    # Degrees of freedom
    df = (rows - 1) * (cols - 1)

    # Approximate p-value using chi-squared distribution
    # Using Wilson-Hilferty transformation
    if df > 0:
        z = (chi_sq / df) ** (1 / 3) - (1 - 2 / (9 * df))
        z /= math.sqrt(2 / (9 * df))
        p_value = 1 - _normal_cdf(z)
    else:
        p_value = 1.0

    # Cramér's V as effect size
    min_dim = min(rows - 1, cols - 1)
    cramers_v = math.sqrt(chi_sq / (total * min_dim)) if total > 0 and min_dim > 0 else 0.0

    significant = p_value < significance_level

    # Generate recommendation
    if not significant:
        rec = (
            "Variables appear independent. Variant assignment doesn't affect outcome distribution."
        )
    elif cramers_v > 0.5:
        rec = f"Strong association (V={cramers_v:.2f}). Variants have very different outcome distributions."
    elif cramers_v > 0.3:
        rec = f"Moderate association (V={cramers_v:.2f}). Notable difference between variants."
    else:
        rec = f"Weak association (V={cramers_v:.2f}). Small but detectable difference."

    return StatisticalResult(
        test_name="Chi-squared test",
        statistic=chi_sq,
        p_value=p_value,
        significant=significant,
        effect_size=cramers_v,
        recommendation=rec,
    )


def mann_whitney_u_test(
    sample1: List[float],
    sample2: List[float],
    significance_level: float = 0.05,
) -> StatisticalResult:
    """Perform Mann-Whitney U test (non-parametric).

    Tests whether two independent samples come from the same distribution.
    More robust than t-test for non-normal distributions.

    Args:
        sample1: First sample (control)
        sample2: Second sample (treatment)
        significance_level: Alpha level for significance

    Returns:
        StatisticalResult with test outcome
    """
    n1 = len(sample1)
    n2 = len(sample2)

    if n1 < 1 or n2 < 1:
        return StatisticalResult(
            test_name="Mann-Whitney U test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            recommendation="Insufficient samples",
        )

    # Combine and rank
    combined = [(x, 1) for x in sample1] + [(x, 2) for x in sample2]
    combined.sort(key=lambda t: t[0])

    # Assign ranks (handle ties with average rank)
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2  # Average rank for ties
        for k in range(i, j):
            if combined[k][1] == 1:
                ranks[("1", k)] = avg_rank
            else:
                ranks[("2", k)] = avg_rank
        i = j

    # Sum of ranks for sample 1
    r1 = sum(v for k, v in ranks.items() if k[0] == "1")

    # U statistic
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    # Normal approximation for large samples
    mean_u = n1 * n2 / 2
    std_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    if std_u == 0:
        z = 0.0
    else:
        z = (u - mean_u) / std_u

    # Two-tailed p-value
    p_value = 2 * _normal_cdf(-abs(z))

    # Common language effect size (probability that random sample2 > sample1)
    effect_size = u1 / (n1 * n2) if n1 * n2 > 0 else 0.5

    significant = p_value < significance_level

    # Generate recommendation
    if not significant:
        rec = "No significant difference in distributions."
    elif effect_size > 0.7:
        rec = f"Strong effect: treatment tends to produce higher values (P={effect_size:.2f})."
    elif effect_size > 0.6:
        rec = f"Medium effect: treatment slightly favored (P={effect_size:.2f})."
    elif effect_size < 0.3:
        rec = f"Control favored: treatment produces lower values (P={effect_size:.2f})."
    else:
        rec = f"Weak effect: distributions slightly different (P={effect_size:.2f})."

    return StatisticalResult(
        test_name="Mann-Whitney U test",
        statistic=u,
        p_value=p_value,
        significant=significant,
        effect_size=effect_size,
        recommendation=rec,
    )


def compute_sample_size(
    baseline_rate: float,
    min_detectable_effect: float,
    power: float = 0.8,
    significance_level: float = 0.05,
) -> int:
    """Compute required sample size for detecting an effect.

    Uses standard formula for two-proportion test.

    Args:
        baseline_rate: Expected baseline conversion/success rate
        min_detectable_effect: Minimum relative effect to detect (e.g., 0.1 for 10%)
        power: Desired statistical power (default 0.8)
        significance_level: Alpha level (default 0.05)

    Returns:
        Required sample size per variant
    """
    p1 = baseline_rate
    p2 = p1 * (1 + min_detectable_effect)

    # Pooled proportion
    p_avg = (p1 + p2) / 2

    # Z-scores
    z_alpha = _normal_ppf(1 - significance_level / 2)
    z_beta = _normal_ppf(power)

    # Sample size formula
    numerator = (
        z_alpha * math.sqrt(2 * p_avg * (1 - p_avg))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p2 - p1) ** 2

    if denominator == 0:
        return 10000  # Fallback

    n = numerator / denominator
    return max(10, int(math.ceil(n)))


def compute_power(
    n: int,
    baseline_rate: float,
    effect_size: float,
    significance_level: float = 0.05,
) -> float:
    """Compute statistical power for given sample size and effect.

    Args:
        n: Sample size per variant
        baseline_rate: Baseline conversion/success rate
        effect_size: Relative effect size to detect
        significance_level: Alpha level

    Returns:
        Statistical power (probability of detecting effect if it exists)
    """
    p1 = baseline_rate
    p2 = p1 * (1 + effect_size)

    # Clamp proportions to valid range
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))

    # Handle edge cases where variance is zero or negative
    var1 = p1 * (1 - p1)
    var2 = p2 * (1 - p2)

    if var1 < 0 or var2 < 0:
        # Invalid proportions
        return 1.0 if p1 != p2 else 0.0

    # Standard error under alternative hypothesis
    se_squared = var1 / n + var2 / n

    if se_squared <= 0:
        return 1.0 if p1 != p2 else 0.0

    se = math.sqrt(se_squared)

    # Critical value
    z_alpha = _normal_ppf(1 - significance_level / 2)

    # Power = P(reject H0 | H1 true)
    z_power = (abs(p2 - p1) - z_alpha * se) / se
    power = _normal_cdf(z_power)

    return min(1.0, max(0.0, power))
