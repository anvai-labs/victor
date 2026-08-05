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

"""Tests for log-content truncation (TD-20 residual: bound error-path logs)."""

from victor.core.utils.log_helpers import MAX_LOG_CHARS, truncate_for_log


def test_short_text_passes_through_unchanged():
    assert truncate_for_log("boom") == "boom"


def test_text_at_limit_is_not_truncated():
    text = "x" * MAX_LOG_CHARS
    assert truncate_for_log(text) == text


def test_oversized_text_is_capped_with_elision_marker():
    text = "x" * (MAX_LOG_CHARS + 250)
    out = truncate_for_log(text)

    assert out.startswith("x" * MAX_LOG_CHARS)
    assert out.endswith("… (+250 more chars)")
    # The whole record stays bounded regardless of input size.
    assert len(out) < MAX_LOG_CHARS + 40


def test_custom_limit_is_respected():
    assert truncate_for_log("abcdef", limit=3) == "abc… (+3 more chars)"


def test_non_positive_limit_disables_truncation():
    huge = "y" * 10_000
    assert truncate_for_log(huge, limit=0) == huge


def test_accepts_exception_objects_directly():
    err = ValueError("z" * 10_000)
    out = truncate_for_log(err)

    assert out.startswith("z" * 20)  # message body, not the class repr
    assert out.endswith("more chars)")
    assert len(out) < MAX_LOG_CHARS + 40
