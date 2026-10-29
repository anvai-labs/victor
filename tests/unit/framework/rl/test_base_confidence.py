# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""The shared learner confidence policy (one definition, seven former copies)."""

import math

import pytest

from victor.framework.rl.base import sample_confidence, saturating_confidence


def _old_saturating(n, scale):
    return min(0.95, 0.5 + 0.45 * (1 - math.exp(-n / scale)))


def _old_warmup(n, min_samples, scale):
    if n < min_samples:
        return 0.3 + 0.2 * (n / min_samples), True
    return _old_saturating(n, scale), False


@pytest.mark.parametrize("scale", [20, 30, 50])  # the scales the learners use
def test_saturating_matches_the_former_inline_formula(scale):
    for n in range(0, 500):
        assert saturating_confidence(n, scale) == _old_saturating(n, scale)


@pytest.mark.parametrize("min_samples, scale", [(10, 20), (5, 30), (10, 50), (3, 20)])
def test_sample_confidence_matches_the_former_inline_branches(min_samples, scale):
    for n in range(0, 300):
        assert sample_confidence(n, min_samples, scale) == _old_warmup(n, min_samples, scale)


def test_policy_shape():
    assert sample_confidence(0, 10, 20) == (0.3, True)
    conf, baseline = sample_confidence(10, 10, 20)
    assert baseline is False and 0.5 < conf < 0.95
    assert saturating_confidence(10**6, 20) == 0.95
    assert saturating_confidence(0, 20) == 0.5
