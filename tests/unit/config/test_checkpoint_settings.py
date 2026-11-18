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

"""Mirror suite for ``victor/config/checkpoint_settings.py``.

Pins the conversation-checkpoint contract and the FEP-0047 workflow group
(durable-by-default graph checkpoints, FOU-D2): defaults are durable-on,
bounds reject nonsense, and the flat ``VICTOR_WORKFLOW_*`` env aliases
route through ``Settings`` construction.
"""

import pytest
from pydantic import ValidationError

from victor.config.checkpoint_settings import CheckpointSettings


def test_conversation_checkpoint_defaults_unchanged():
    cp = CheckpointSettings()
    assert cp.checkpoint_enabled is True
    assert cp.checkpoint_auto_interval == 5
    assert cp.checkpoint_max_per_session == 50
    assert cp.checkpoint_compression_enabled is True
    assert cp.checkpoint_compression_threshold == 1024


def test_workflow_group_defaults_are_durable_on():
    """FEP-0047: new installs are durable from first run; opt-out is one knob."""
    cp = CheckpointSettings()
    assert cp.workflow_checkpoints_enabled is True
    assert cp.workflow_checkpoint_max_per_thread == 20
    assert cp.workflow_checkpoint_retention_days == 7
    assert cp.workflow_checkpoint_db_path is None


def test_workflow_bounds_reject_nonsense():
    with pytest.raises(ValidationError):
        CheckpointSettings(workflow_checkpoint_max_per_thread=0)
    with pytest.raises(ValidationError):
        CheckpointSettings(workflow_checkpoint_retention_days=-1)


def test_retention_days_zero_disables_ttl_sweep_semantics():
    """0 is the documented 'no sweep' value and must be accepted."""
    cp = CheckpointSettings(workflow_checkpoint_retention_days=0)
    assert cp.workflow_checkpoint_retention_days == 0


def test_flat_env_aliases_route_to_checkpoint_group():
    from victor.config.settings import Settings

    routed = Settings(
        workflow_checkpoints_enabled=False,
        workflow_checkpoint_max_per_thread=5,
        workflow_checkpoint_retention_days=3,
        workflow_checkpoint_db_path="/tmp/cp.db",
    )
    assert routed.checkpoint.workflow_checkpoints_enabled is False
    assert routed.checkpoint.workflow_checkpoint_max_per_thread == 5
    assert routed.checkpoint.workflow_checkpoint_retention_days == 3
    assert routed.checkpoint.workflow_checkpoint_db_path == "/tmp/cp.db"
