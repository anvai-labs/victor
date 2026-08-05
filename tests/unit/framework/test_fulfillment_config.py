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

"""Tests for FulfillmentConfig.__post_init__ validators — Wave H."""

import pytest


class TestFulfillmentConfigValidation:
    """FulfillmentConfig: threshold ranges and per-strategy weight group sums."""

    def test_default_config_passes_validation(self):
        from victor.framework.fulfillment import FulfillmentConfig

        cfg = FulfillmentConfig()
        assert cfg is not None

    def test_code_gen_weights_sum_to_one_by_default(self):
        from victor.framework.fulfillment import FulfillmentConfig

        cfg = FulfillmentConfig()
        total = (
            cfg.file_exists_weight
            + cfg.syntax_valid_weight
            + cfg.non_empty_weight
            + cfg.pattern_weight
        )
        assert abs(total - 1.0) <= 0.01

    def test_test_weights_sum_to_one_by_default(self):
        from victor.framework.fulfillment import FulfillmentConfig

        cfg = FulfillmentConfig()
        total = cfg.test_pass_weight + cfg.test_error_weight + cfg.test_files_weight
        assert abs(total - 1.0) <= 0.01

    def test_code_gen_weights_not_summing_raises(self):
        from victor.framework.fulfillment import FulfillmentConfig

        with pytest.raises(ValueError, match="[Cc]ode"):
            FulfillmentConfig(
                file_exists_weight=0.5,
                syntax_valid_weight=0.5,
                non_empty_weight=0.5,
                pattern_weight=0.5,
            )

    def test_test_weights_not_summing_raises(self):
        from victor.framework.fulfillment import FulfillmentConfig

        with pytest.raises(ValueError, match="[Tt]est"):
            FulfillmentConfig(
                test_pass_weight=0.9,
                test_error_weight=0.9,
                test_files_weight=0.9,
            )

    def test_fulfilled_threshold_out_of_range_raises(self):
        from victor.framework.fulfillment import FulfillmentConfig

        with pytest.raises(ValueError, match="fulfilled_threshold"):
            FulfillmentConfig(fulfilled_threshold=1.5)

    def test_partial_threshold_out_of_range_raises(self):
        from victor.framework.fulfillment import FulfillmentConfig

        with pytest.raises(ValueError, match="partial_threshold"):
            FulfillmentConfig(partial_threshold=-0.1)

    def test_near_one_within_tolerance_accepted_for_code_gen(self):
        from victor.framework.fulfillment import FulfillmentConfig

        # Sum = 1.009 — within ±0.01 tolerance
        cfg = FulfillmentConfig(
            file_exists_weight=0.302,
            syntax_valid_weight=0.300,
            non_empty_weight=0.200,
            pattern_weight=0.207,
        )
        assert cfg is not None

    def test_near_one_within_tolerance_accepted_for_test(self):
        from victor.framework.fulfillment import FulfillmentConfig

        # Sum = 1.009
        cfg = FulfillmentConfig(
            test_pass_weight=0.503,
            test_error_weight=0.300,
            test_files_weight=0.206,
        )
        assert cfg is not None

    def test_default_config_module_constant_still_valid(self):
        """DEFAULT_CONFIG at module level must remain valid after __post_init__ is added."""
        from victor.framework.fulfillment import DEFAULT_CONFIG

        assert DEFAULT_CONFIG is not None
        assert DEFAULT_CONFIG.fulfilled_threshold == 0.8
