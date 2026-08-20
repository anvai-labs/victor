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

"""Framework Enhancement Proposal (FEP) system.

This module provides the FEP validation and management system for Victor.
FEPs are design documents describing framework-level changes.
"""

from victor.feps.schema import (
    FEPType,
    FEPStatus,
    FEPValidator,
    FEPMetadata,
    FEPValidationError,
    FEPValidationResult,
    FEPSection,
    parse_fep_metadata,
    validate_fep,
)
from victor.feps.manager import FEPManager, create_fep_manager

__all__ = [
    "FEPType",
    "FEPStatus",
    "FEPValidator",
    "FEPMetadata",
    "FEPValidationError",
    "FEPValidationResult",
    "FEPSection",
    "parse_fep_metadata",
    "validate_fep",
    "FEPManager",
    "create_fep_manager",
]
