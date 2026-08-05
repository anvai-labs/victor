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

"""Tests for consolidated environment variable filtering."""

from victor.security.env_filtering import (
    SAFE_ENV_VARS,
    SENSITIVE_ENV_VARS,
    get_filtered_env,
    get_minimal_env,
)


class TestGetFilteredEnv:
    def test_strips_sensitive_vars(self):
        env = {"PATH": "/bin", "ANTHROPIC_API_KEY": "secret", "HOME": "/home"}
        filtered = get_filtered_env(env)
        assert "PATH" in filtered
        assert "HOME" in filtered
        assert "ANTHROPIC_API_KEY" not in filtered

    def test_preserves_non_sensitive(self):
        env = {"MY_CUSTOM_VAR": "value", "PATH": "/bin"}
        filtered = get_filtered_env(env)
        assert filtered["MY_CUSTOM_VAR"] == "value"
        assert filtered["PATH"] == "/bin"

    def test_extra_vars_override(self):
        env = {"PATH": "/bin"}
        filtered = get_filtered_env(env, extra_vars={"CUSTOM": "val"})
        assert filtered["CUSTOM"] == "val"
        assert filtered["PATH"] == "/bin"

    def test_extra_vars_override_existing(self):
        env = {"PATH": "/bin"}
        filtered = get_filtered_env(env, extra_vars={"PATH": "/usr/bin"})
        assert filtered["PATH"] == "/usr/bin"

    def test_skip_strip(self):
        env = {"ANTHROPIC_API_KEY": "keep"}
        filtered = get_filtered_env(env, strip_sensitive=False)
        assert "ANTHROPIC_API_KEY" in filtered
        assert filtered["ANTHROPIC_API_KEY"] == "keep"

    def test_defaults_to_os_environ(self):
        # When base_env is None, it defaults to os.environ
        filtered = get_filtered_env()
        # Should have at least PATH (present on most systems)
        assert isinstance(filtered, dict)

    def test_strips_all_sensitive_vars(self):
        # Build an env with all sensitive vars present
        env = dict.fromkeys(SENSITIVE_ENV_VARS, "secret")
        env["PATH"] = "/bin"
        filtered = get_filtered_env(env)
        for var in SENSITIVE_ENV_VARS:
            assert var not in filtered
        assert filtered["PATH"] == "/bin"

    def test_empty_env(self):
        filtered = get_filtered_env({})
        assert filtered == {}

    def test_empty_env_with_extras(self):
        filtered = get_filtered_env({}, extra_vars={"FOO": "bar"})
        assert filtered == {"FOO": "bar"}


class TestGetMinimalEnv:
    def test_only_contains_safe_vars(self):
        result = get_minimal_env()
        for key in result:
            assert key in SAFE_ENV_VARS

    def test_extra_vars_added(self):
        result = get_minimal_env(extra_vars={"CUSTOM": "val"})
        assert result["CUSTOM"] == "val"

    def test_no_sensitive_vars(self):
        result = get_minimal_env()
        for var in SENSITIVE_ENV_VARS:
            assert var not in result


class TestConstants:
    def test_sensitive_vars_comprehensive(self):
        assert "AWS_SECRET_ACCESS_KEY" in SENSITIVE_ENV_VARS
        assert "OPENAI_API_KEY" in SENSITIVE_ENV_VARS
        assert "DATABASE_URL" in SENSITIVE_ENV_VARS
        assert "GITHUB_TOKEN" in SENSITIVE_ENV_VARS
        assert "ANTHROPIC_API_KEY" in SENSITIVE_ENV_VARS
        assert "HF_TOKEN" in SENSITIVE_ENV_VARS

    def test_safe_vars_basic(self):
        assert "PATH" in SAFE_ENV_VARS
        assert "HOME" in SAFE_ENV_VARS
        assert "LANG" in SAFE_ENV_VARS
        assert "SHELL" in SAFE_ENV_VARS

    def test_constants_are_frozensets(self):
        assert isinstance(SENSITIVE_ENV_VARS, frozenset)
        assert isinstance(SAFE_ENV_VARS, frozenset)

    def test_no_overlap(self):
        overlap = SENSITIVE_ENV_VARS & SAFE_ENV_VARS
        assert len(overlap) == 0, f"Vars in both sets: {overlap}"
