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

"""Guardrails for service-owned runtime protocol imports."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path("victor")
LEGACY_RUNTIME_PROTOCOL_EXPORTS = frozenset(
    {
        "ChunkGeneratorProtocol",
        "CoordinationAdvisorRuntimeProtocol",
        "IntentClassifierProtocol",
        "RLCoordinatorProtocol",
        "ReminderManagerProtocol",
        "ResponseSanitizerProtocol",
        "StreamingRecoveryCoordinatorProtocol",
        "PromptCoordinatorProtocol",
        "StateCoordinatorProtocol",
        "StreamingConfidenceMonitorProtocol",
        "StreamingHandlerProtocol",
        "StreamingMetricsCollectorProtocol",
        "TaskCoordinatorProtocol",
        "ToolPlannerProtocol",
    }
)


def test_internal_code_does_not_import_legacy_runtime_protocol_names() -> None:
    violations: list[str] = []

    for path in sorted(ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module not in {
                "victor.agent.protocols.infrastructure_protocols",
                "victor.agent.protocols",
                "victor.agent.protocols.analysis_protocols",
                "victor.agent.protocols.coordination_protocols",
                "victor.agent.protocols.streaming_protocols",
            }:
                continue

            imported_names = {alias.name for alias in node.names}
            bad_names = sorted(imported_names & LEGACY_RUNTIME_PROTOCOL_EXPORTS)
            if bad_names or "*" in imported_names:
                violations.append(
                    f"{path}:{node.lineno} imports legacy runtime protocol names: "
                    f"{', '.join(bad_names or ['*'])}"
                )

    assert not violations, "\n".join(violations)
