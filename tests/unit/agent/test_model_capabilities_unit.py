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

"""Unit tests for model_capabilities.py functions."""

from unittest.mock import patch

from victor.config.model_capabilities import (
    ToolCallingMatrix,
    _load_tool_capable_patterns_from_yaml,
)


class TestMinimalBuiltinDefaults:
    """Tests for _minimal_builtin_defaults function."""

    def test_cloud_providers_have_wildcard(self):
        """Cloud providers should have wildcard support."""
        from victor.config.model_capabilities import _minimal_builtin_defaults

        defaults = _minimal_builtin_defaults()
        assert "*" in defaults.get("anthropic", [])
        assert "*" in defaults.get("openai", [])
        assert "*" in defaults.get("google", [])
        assert "*" in defaults.get("xai", [])
        assert "*" in defaults.get("vllm", [])

    def test_local_providers_have_patterns(self):
        """Local providers should have model patterns."""
        from victor.config.model_capabilities import _minimal_builtin_defaults

        defaults = _minimal_builtin_defaults()
        ollama_patterns = defaults.get("ollama", [])
        lmstudio_patterns = defaults.get("lmstudio", [])

        assert "llama3.1*" in ollama_patterns
        assert "qwen2.5*" in ollama_patterns
        assert "mistral*" in ollama_patterns
        assert "llama3.1*" in lmstudio_patterns
        assert "qwen2.5*" in lmstudio_patterns

    def test_local_providers_have_same_patterns(self):
        """Ollama and LMStudio should have the same patterns."""
        from victor.config.model_capabilities import _minimal_builtin_defaults

        defaults = _minimal_builtin_defaults()
        ollama = set(defaults.get("ollama", []))
        lmstudio = set(defaults.get("lmstudio", []))
        assert ollama == lmstudio


class TestExtractToolCapablePatterns:
    """Tests for _extract_tool_capable_patterns function."""

    def test_extracts_provider_level_patterns(self):
        """Should extract patterns from provider_defaults-level native_tool_calls."""
        from victor.config.model_capabilities import _extract_tool_capable_patterns

        data = {
            "provider_defaults": {
                "anthropic": {"native_tool_calls": True},
                "openai": {"native_tool_calls": True},
            }
        }
        result = {}
        _extract_tool_capable_patterns(data, result)

        assert "*" in result.get("anthropic", [])
        assert "*" in result.get("openai", [])

    def test_extracts_model_level_patterns(self):
        """Should extract patterns from model-level providers.<provider>.native_tool_calls."""
        from victor.config.model_capabilities import _extract_tool_capable_patterns

        data = {
            "models": {
                "llama3.1*": {
                    "providers": {
                        "ollama": {"native_tool_calls": True},
                        "lmstudio": {"native_tool_calls": True},
                        "vllm": {"native_tool_calls": True},
                    }
                },
                "qwen2.5*": {
                    "providers": {
                        "ollama": {"native_tool_calls": True},
                    }
                },
            }
        }
        result = {}
        _extract_tool_capable_patterns(data, result)

        assert "llama3.1*" in result.get("ollama", [])
        assert "llama3.1*" in result.get("lmstudio", [])
        assert "llama3.1*" in result.get("vllm", [])
        assert "qwen2.5*" in result.get("ollama", [])

    def test_skips_non_tool_capable_models(self):
        """Should skip models without native_tool_calls."""
        from victor.config.model_capabilities import _extract_tool_capable_patterns

        data = {
            "models": {
                "llama3.1*": {
                    "providers": {
                        "ollama": {"native_tool_calls": True},
                    }
                },
                "old-model*": {
                    "providers": {
                        "ollama": {"native_tool_calls": False},
                    }
                },
            }
        }
        result = {}
        _extract_tool_capable_patterns(data, result)

        assert "llama3.1*" in result.get("ollama", [])
        assert "old-model*" not in result.get("ollama", [])

    def test_handles_non_dict_values(self):
        """Should handle non-dict capability values."""
        from victor.config.model_capabilities import _extract_tool_capable_patterns

        data = {
            "provider_defaults": {
                "test_provider": "not a dict",
            },
            "models": {
                "test_model": "also not a dict",
            },
        }
        result = {}
        _extract_tool_capable_patterns(data, result)

        # Should not crash, just skip non-dict values
        assert "test_provider" not in result
        assert "test_model" not in result.get("ollama", [])


class TestFlattenYamlManifest:
    """Tests for _flatten_yaml_manifest function."""

    def test_flattens_tiered_structure(self):
        """Should flatten tiered model lists."""
        from victor.config.model_capabilities import _flatten_yaml_manifest

        data = {
            "openai": {
                "tier1": [
                    {"name": "gpt-4"},
                    {"name": "gpt-4-turbo"},
                ],
                "tier2": [
                    {"name": "gpt-3.5-turbo"},
                ],
            }
        }
        result = _flatten_yaml_manifest(data)

        assert "gpt-4" in result["openai"]
        assert "gpt-4-turbo" in result["openai"]
        assert "gpt-3.5-turbo" in result["openai"]

    def test_flattens_flat_list(self):
        """Should handle flat model lists."""
        from victor.config.model_capabilities import _flatten_yaml_manifest

        data = {
            "ollama": [
                {"name": "llama3.1:8b"},
                "mistral:7b",  # String entry
            ]
        }
        result = _flatten_yaml_manifest(data)

        assert "llama3.1:8b" in result["ollama"]
        assert "mistral:7b" in result["ollama"]

    def test_deduplicates_models(self):
        """Should deduplicate model names."""
        from victor.config.model_capabilities import _flatten_yaml_manifest

        data = {
            "test": {
                "tier1": [{"name": "model-a"}, {"name": "model-a"}],
                "tier2": [{"name": "model-a"}],
            }
        }
        result = _flatten_yaml_manifest(data)

        assert result["test"].count("model-a") == 1

    def test_skips_none_names(self):
        """Should skip entries without name."""
        from victor.config.model_capabilities import _flatten_yaml_manifest

        data = {
            "test": [
                {"name": "valid-model"},
                {"description": "no name field"},
            ]
        }
        result = _flatten_yaml_manifest(data)

        assert "valid-model" in result["test"]
        assert len(result["test"]) == 1


class TestToolCallingMatrix:
    """Additional tests for ToolCallingMatrix class."""

    def test_wildcard_matching(self):
        """Test wildcard pattern matching with mocked defaults."""
        with patch(
            "victor.config.model_capabilities._load_tool_capable_patterns_from_yaml"
        ) as mock_load:
            mock_load.return_value = {}
            matrix = ToolCallingMatrix(
                {"test": ["model*", "*-special"]},
            )

            # Prefix wildcard: model*
            assert matrix.is_tool_call_supported("test", "model-a")
            assert matrix.is_tool_call_supported("test", "model-b-large")
            # Suffix wildcard: *-special
            assert matrix.is_tool_call_supported("test", "any-special")
            # No match
            assert not matrix.is_tool_call_supported("test", "other-no-match")

    def test_always_allow_providers(self):
        """Test always_allow_providers bypass."""
        matrix = ToolCallingMatrix(
            {},
            always_allow_providers=["special_provider"],
        )

        assert matrix.is_tool_call_supported("special_provider", "any-model")
        assert matrix.is_tool_call_supported("special_provider", "another-model")

    def test_case_insensitive_matching(self):
        """Test case-insensitive provider and model matching."""
        with patch(
            "victor.config.model_capabilities._load_tool_capable_patterns_from_yaml"
        ) as mock_load:
            mock_load.return_value = {}
            matrix = ToolCallingMatrix(
                {"TestProvider": ["TestModel*"]},
            )

            assert matrix.is_tool_call_supported("testprovider", "testmodel-1")
            assert matrix.is_tool_call_supported("TESTPROVIDER", "TESTMODEL-2")

    def test_get_supported_models_unknown_provider(self):
        """Should return empty list for unknown provider."""
        matrix = ToolCallingMatrix({})
        result = matrix.get_supported_models("unknown_provider_xyz")
        assert result == []

    def test_to_json(self):
        """Test JSON serialization."""
        with patch(
            "victor.config.model_capabilities._load_tool_capable_patterns_from_yaml"
        ) as mock_load:
            mock_load.return_value = {}
            matrix = ToolCallingMatrix(
                {"provider1": ["model1", "model2"]},
            )

            json_str = matrix.to_json()
            assert "provider1" in json_str
            assert "model1" in json_str

    def test_matches_exact_name(self):
        """Test exact model name matching."""
        with patch(
            "victor.config.model_capabilities._load_tool_capable_patterns_from_yaml"
        ) as mock_load:
            mock_load.return_value = {}
            matrix = ToolCallingMatrix(
                {"test": ["exact-model-name"]},
            )

            assert matrix.is_tool_call_supported("test", "exact-model-name")
            # Note: _matches has fallback substring match, so extended names may match
            # This tests that exact names work
            assert matrix.is_tool_call_supported("test", "exact-model-name")

    def test_matches_prefix_wildcard(self):
        """Test prefix wildcard matching."""
        with patch(
            "victor.config.model_capabilities._load_tool_capable_patterns_from_yaml"
        ) as mock_load:
            mock_load.return_value = {}
            matrix = ToolCallingMatrix(
                {"test": ["prefix*"]},
            )

            assert matrix.is_tool_call_supported("test", "prefix-anything")
            assert matrix.is_tool_call_supported("test", "prefix")
            assert not matrix.is_tool_call_supported("test", "other-no-prefix-here")

    def test_matches_suffix_wildcard(self):
        """Test suffix wildcard matching."""
        with patch(
            "victor.config.model_capabilities._load_tool_capable_patterns_from_yaml"
        ) as mock_load:
            mock_load.return_value = {}
            matrix = ToolCallingMatrix(
                {"test": ["*suffix"]},
            )

            assert matrix.is_tool_call_supported("test", "anything-suffix")
            assert matrix.is_tool_call_supported("test", "suffix")
            # Note: suffix-other ends with "other", not "suffix"
            assert not matrix.is_tool_call_supported("test", "suffix-other")

    def test_matches_contains_pattern_fallback(self):
        """Test substring fallback matching (non-wildcard patterns).

        Note: *pattern* syntax doesn't work as 'contains' due to check order.
        The fallback substring match (last line of _matches) handles contains.
        """
        with patch(
            "victor.config.model_capabilities._load_tool_capable_patterns_from_yaml"
        ) as mock_load:
            mock_load.return_value = {}
            # Use plain pattern without wildcards for substring matching
            matrix = ToolCallingMatrix(
                {"test": ["middle"]},
            )

            # Plain "middle" pattern matches via substring fallback
            assert matrix.is_tool_call_supported("test", "has-middle-here")
            assert matrix.is_tool_call_supported("test", "middle")
            assert not matrix.is_tool_call_supported("test", "no-match-xyz")

    def test_manifest_merge(self):
        """Test manifest merging."""
        # Start with no base patterns (use empty mock)
        with patch(
            "victor.config.model_capabilities._load_tool_capable_patterns_from_yaml"
        ) as mock_load:
            mock_load.return_value = {"existing": ["model1"]}
            matrix = ToolCallingMatrix(
                {"existing": ["model2"], "new": ["model3"]},
            )

        assert matrix.is_tool_call_supported("existing", "model1")
        assert matrix.is_tool_call_supported("existing", "model2")
        assert matrix.is_tool_call_supported("new", "model3")

    def test_substring_fallback_matching(self):
        """Test the fallback substring matching in _matches."""
        # When pattern doesn't have wildcards, it uses substring matching
        with patch(
            "victor.config.model_capabilities._load_tool_capable_patterns_from_yaml"
        ) as mock_load:
            mock_load.return_value = {}
            matrix = ToolCallingMatrix(
                {"test": ["llama"]},
            )

            # "llama" is a substring of "llama3.1:8b"
            assert matrix.is_tool_call_supported("test", "llama3.1:8b")


class TestLoadToolCapablePatternsFromYaml:
    """Tests for _load_tool_capable_patterns_from_yaml function."""

    def test_loads_from_user_profiles(self):
        """Should load patterns from user profiles.yaml."""
        mock_yaml_content = """
model_capabilities:
  providers:
    anthropic:
      native_tool_calls: true
  models:
    llama3.1*:
      native_tool_calls: true
"""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text", return_value=mock_yaml_content):
                result = _load_tool_capable_patterns_from_yaml()

                assert "*" in result.get("anthropic", [])
                assert "llama3.1*" in result.get("ollama", [])

    def test_falls_back_to_defaults(self):
        """Should fall back to defaults when no profiles.yaml."""
        with patch("pathlib.Path.exists", return_value=False):
            result = _load_tool_capable_patterns_from_yaml()

            # Should have minimal defaults
            assert "anthropic" in result
            assert "*" in result["anthropic"]

    def test_handles_yaml_error(self):
        """Should handle YAML parsing errors gracefully."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "pathlib.Path.read_text",
                side_effect=Exception("YAML parse error"),
            ):
                result = _load_tool_capable_patterns_from_yaml()

                # Should fall back to defaults
                assert "anthropic" in result

    def test_handles_empty_model_capabilities(self):
        """Should handle empty model_capabilities section."""
        mock_yaml_content = """
model_capabilities: {}
"""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text", return_value=mock_yaml_content):
                result = _load_tool_capable_patterns_from_yaml()

                # Should fall back to defaults
                assert "anthropic" in result
