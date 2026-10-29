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

"""Stage contract protocols for LSP compliance.

This module defines the StageContract protocol and related validation
to ensure Liskov Substitution Principle (LSP) compliance for vertical
stage definitions.

Design Pattern: Protocol + Strategy
- StageContract: Protocol defining valid stage contract
- StageValidator: Validates stage definitions against contract
- Enforces required stages, valid transitions, naming conventions

Integration Point:
    Update VerticalBase.get_stages() to validate against StageContract

Phase 2: Fix SOLID Violations via Coordinator Extraction
"""

from __future__ import annotations

# Stage contract types and validation are defined once, in the SDK (promoted
# from here). This module keeps the long-standing import path.
from victor_contracts.verticals.protocols.promoted import (
    StageContract as StageContract,
    StageContractMixin as StageContractMixin,
    StageValidator as StageValidator,
    validate_stage_contract as validate_stage_contract,
)
from victor_contracts.verticals.protocols.promoted_types import (
    StageValidationResult as StageValidationResult,
    ValidationError as ValidationError,
)

__all__ = [
    "StageContract",
    "StageValidator",
    "StageValidationResult",
    "ValidationError",
    "validate_stage_contract",
    "StageContractMixin",
]
