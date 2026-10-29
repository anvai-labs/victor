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

"""victor.core.verticals.protocols.stages re-exports the SDK stage contract (one definition)."""

import importlib

import pytest

MODULE = "victor.core.verticals.protocols.stages"

PAIRS = [
    ("StageContract", "victor_contracts.verticals.protocols.promoted", "StageContract"),
    ("StageValidator", "victor_contracts.verticals.protocols.promoted", "StageValidator"),
    ("StageContractMixin", "victor_contracts.verticals.protocols.promoted", "StageContractMixin"),
    (
        "validate_stage_contract",
        "victor_contracts.verticals.protocols.promoted",
        "validate_stage_contract",
    ),
    ("ValidationError", "victor_contracts.verticals.protocols.promoted_types", "ValidationError"),
    (
        "StageValidationResult",
        "victor_contracts.verticals.protocols.promoted_types",
        "StageValidationResult",
    ),
]


@pytest.mark.parametrize("core_name, sdk_mod, sdk_name", PAIRS)
def test_core_name_is_the_sdk_object(core_name, sdk_mod, sdk_name):
    core = getattr(importlib.import_module(MODULE), core_name)
    assert core is getattr(importlib.import_module(sdk_mod), sdk_name)


def test_stage_validation_behaves_identically_through_core():
    from victor.core.verticals.protocols.stages import ValidationError, validate_stage_contract

    good = {
        "INITIAL": {"name": "INITIAL", "description": "start", "next_stages": ["COMPLETION"]},
        "COMPLETION": {"name": "COMPLETION", "description": "done", "next_stages": []},
    }
    bad = {"_internal": {"description": ""}}
    assert validate_stage_contract(good).is_valid, validate_stage_contract(good).errors
    result = validate_stage_contract(bad)
    assert not result.is_valid
    assert {code for code, _ in result.errors} & {
        ValidationError.INVALID_STAGE_NAME,
        ValidationError.INVALID_DESCRIPTION,
    }
