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

"""Tests for BaseRLConfig (Phase 3.1: RL Config Centralization)."""

import pytest

from victor.framework.rl import LearnerType
from victor.framework.rl.config import (
    DEFAULT_ACTIVE_LEARNERS,
    DEFAULT_PATIENCE_MAP,
    BaseRLConfig,
)


class TestDefaultConstants:
    """Test shared default constants."""

    def test_default_active_learners_has_expected_types(self):
        """DEFAULT_ACTIVE_LEARNERS should include core learner types."""
        assert LearnerType.TOOL_SELECTOR in DEFAULT_ACTIVE_LEARNERS
        assert LearnerType.CONTINUATION_PATIENCE in DEFAULT_ACTIVE_LEARNERS
        assert LearnerType.GROUNDING_THRESHOLD in DEFAULT_ACTIVE_LEARNERS

    def test_default_patience_map_has_expected_providers(self):
        """DEFAULT_PATIENCE_MAP should include common providers."""
        assert "anthropic" in DEFAULT_PATIENCE_MAP
        assert "openai" in DEFAULT_PATIENCE_MAP
        assert "google" in DEFAULT_PATIENCE_MAP
        assert "ollama" in DEFAULT_PATIENCE_MAP

    def test_default_patience_values_are_reasonable(self):
        """Patience values should be in reasonable range (2-10)."""
        for provider, patience in DEFAULT_PATIENCE_MAP.items():
            assert 2 <= patience <= 10, f"{provider}: patience={patience} out of range"


class TestBaseRLConfigDefaults:
    """Test BaseRLConfig initialization with defaults."""

    @pytest.fixture
    def config(self):
        """Create a BaseRLConfig instance."""
        return BaseRLConfig()

    def test_active_learners_inherits_defaults(self, config):
        """active_learners should inherit DEFAULT_ACTIVE_LEARNERS."""
        assert config.active_learners == DEFAULT_ACTIVE_LEARNERS

    def test_default_patience_inherits_defaults(self, config):
        """default_patience should inherit DEFAULT_PATIENCE_MAP."""
        assert config.default_patience == DEFAULT_PATIENCE_MAP

    def test_task_type_mappings_starts_empty(self, config):
        """task_type_mappings should be empty dict by default."""
        assert config.task_type_mappings == {}

    def test_quality_thresholds_starts_empty(self, config):
        """quality_thresholds should be empty dict by default."""
        assert config.quality_thresholds == {}

    def test_exploration_bonus_has_default(self, config):
        """exploration_bonus should have default value."""
        assert hasattr(config, "exploration_bonus")
        assert config.exploration_bonus == 0.15


class TestBaseRLConfigMethods:
    """Test BaseRLConfig methods."""

    @pytest.fixture
    def config(self):
        """Create a BaseRLConfig with some test data."""
        cfg = BaseRLConfig()
        cfg.task_type_mappings = {
            "debug": ["read", "grep", "shell"],
            "test": ["shell", "read"],
        }
        cfg.quality_thresholds = {
            "debug": 0.85,
            "test": 0.90,
        }
        return cfg

    def test_get_tools_for_task_existing(self, config):
        """get_tools_for_task should return tools for existing task."""
        tools = config.get_tools_for_task("debug")
        assert tools == ["read", "grep", "shell"]

    def test_get_tools_for_task_missing(self, config):
        """get_tools_for_task should return empty list for missing task."""
        tools = config.get_tools_for_task("nonexistent")
        assert tools == []

    def test_get_tools_for_task_case_insensitive(self, config):
        """get_tools_for_task should be case-insensitive."""
        tools = config.get_tools_for_task("DEBUG")
        assert tools == ["read", "grep", "shell"]

    def test_get_quality_threshold_existing(self, config):
        """get_quality_threshold should return threshold for existing task."""
        threshold = config.get_quality_threshold("debug")
        assert threshold == 0.85

    def test_get_quality_threshold_missing(self, config):
        """get_quality_threshold should return 0.80 for missing task."""
        threshold = config.get_quality_threshold("nonexistent")
        assert threshold == 0.80

    def test_get_patience_known_provider(self, config):
        """get_patience should return patience for known provider."""
        patience = config.get_patience("anthropic")
        assert patience == DEFAULT_PATIENCE_MAP["anthropic"]

    def test_get_patience_unknown_provider(self, config):
        """get_patience should return 4 for unknown provider."""
        patience = config.get_patience("unknown_provider")
        assert patience == 4

    def test_get_patience_case_insensitive(self, config):
        """get_patience should be case-insensitive."""
        patience = config.get_patience("ANTHROPIC")
        assert patience == DEFAULT_PATIENCE_MAP["anthropic"]

    def test_is_learner_active_true(self, config):
        """is_learner_active should return True for active learner."""
        assert config.is_learner_active(LearnerType.TOOL_SELECTOR)

    def test_is_learner_active_false(self, config):
        """is_learner_active should return False for inactive learner."""
        assert not config.is_learner_active(LearnerType.CACHE_EVICTION)

    def test_get_rl_config_returns_dict(self, config):
        """get_rl_config should return dictionary with all config."""
        rl_config = config.get_rl_config()
        assert isinstance(rl_config, dict)
        assert "active_learners" in rl_config
        assert "task_type_mappings" in rl_config
        assert "quality_thresholds" in rl_config
        assert "default_patience" in rl_config

    def test_get_rl_config_learner_values(self, config):
        """get_rl_config should convert learners to values."""
        rl_config = config.get_rl_config()
        assert all(isinstance(learner, str) for learner in rl_config["active_learners"])

    def test_repr(self, config):
        """__repr__ should include learner and task type counts."""
        repr_str = repr(config)
        assert "BaseRLConfig" in repr_str
        assert "learners=3" in repr_str
        assert "task_types=2" in repr_str


class TestBaseRLConfigImmutability:
    """Test that BaseRLConfig returns copies, not references."""

    @pytest.fixture
    def config(self):
        """Create a BaseRLConfig with test data."""
        cfg = BaseRLConfig()
        cfg.task_type_mappings = {"test": ["read"]}
        cfg.quality_thresholds = {"test": 0.85}
        cfg.default_patience = {"anthropic": 3}
        return cfg

    def test_task_type_mappings_returns_copy(self, config):
        """task_type_mappings should be copiable."""
        original = config.task_type_mappings.copy()
        config.task_type_mappings["new"] = ["write"]
        assert "new" in config.task_type_mappings
        # The original dict should not be affected
        assert "new" not in original

    def test_quality_thresholds_returns_copy(self, config):
        """quality_thresholds should be copiable."""
        original = config.quality_thresholds.copy()
        config.quality_thresholds["new"] = 0.90
        assert "new" in config.quality_thresholds
        assert "new" not in original

    def test_default_patience_returns_copy(self, config):
        """default_patience should be copiable."""
        original = config.default_patience.copy()
        config.default_patience["new"] = 5
        assert "new" in config.default_patience
        assert "new" not in original


class TestBaseRLConfigVerticalInheritance:
    """Test that verticals can properly inherit from BaseRLConfig."""

    def test_research_config_inherits_base(self):
        """ResearchRLConfig should inherit from BaseRLConfig."""
        try:
            from victor_research.rl import ResearchRLConfig
        except ImportError:
            pytest.skip("victor-research package not installed")

        config = ResearchRLConfig()
        assert isinstance(config, BaseRLConfig)
        assert config.get_patience("anthropic") in (3, 4, 5)

    def test_dataanalysis_config_inherits_base(self):
        """DataAnalysisRLConfig should inherit from BaseRLConfig."""
        try:
            from victor_dataanalysis.rl import DataAnalysisRLConfig
        except ImportError:
            pytest.skip("victor-dataanalysis package not installed")

        config = DataAnalysisRLConfig()
        assert isinstance(config, BaseRLConfig)

    def test_devops_config_inherits_base(self):
        """DevOpsRLConfig should inherit from BaseRLConfig."""
        try:
            from victor_devops.rl import DevOpsRLConfig
        except ImportError:
            pytest.skip("victor-devops package not installed")

        config = DevOpsRLConfig()
        assert isinstance(config, BaseRLConfig)

    def test_rag_config_inherits_base(self):
        """RAGRLConfig should inherit from BaseRLConfig."""
        try:
            from victor_rag.rl import RAGRLConfig
        except ImportError:
            pytest.skip("victor-rag package not installed")

        config = RAGRLConfig()
        assert isinstance(config, BaseRLConfig)

    def test_coding_config_inherits_base(self):
        """CodingRLConfig should inherit from BaseRLConfig."""
        try:
            from victor_coding.rl.config import CodingRLConfig
        except ImportError:
            pytest.skip("victor-coding package not installed")

        config = CodingRLConfig()
        assert isinstance(config, BaseRLConfig)
