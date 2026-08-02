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

"""Coverage parsers subpackage."""

# Re-export from parent parser module for convenience
from victor_coding.coverage.parser import (
    BaseCoverageParser,
    CloverParser,
    CoberturaParser,
    CoverageParser,
    GoCoverParser,
    JestCoverageParser,
    LcovParser,
    get_parser_for_file,
    parse_coverage_file,
)

__all__ = [
    "BaseCoverageParser",
    "CloverParser",
    "CoberturaParser",
    "CoverageParser",
    "GoCoverParser",
    "JestCoverageParser",
    "LcovParser",
    "get_parser_for_file",
    "parse_coverage_file",
]
