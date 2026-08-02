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

"""Core utility modules for Victor.

This package contains shared utilities used across the codebase.
"""

from victor.core.utils.content_hasher import ContentHasher, HasherPresets
from victor.core.utils.math_utils import (
    clamp,
    coerce_float,
    coerce_int,
    lerp,
    normalize_score,
)
from victor.core.utils.text_normalizer import (
    TextNormalizationPresets,
    normalize_for_filename,
    normalize_for_git_branch,
    normalize_for_test_filename,
    sanitize_class_name,
    slugify,
)

__all__ = [
    "ContentHasher",
    "HasherPresets",
    "normalize_for_git_branch",
    "normalize_for_filename",
    "slugify",
    "sanitize_class_name",
    "normalize_for_test_filename",
    "TextNormalizationPresets",
    # Math utilities
    "clamp",
    "coerce_float",
    "coerce_int",
    "lerp",
    "normalize_score",
]
