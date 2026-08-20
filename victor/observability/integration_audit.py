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

"""Append-only JSONL audit log for vertical integration results.

Extracted from IntegrationResult.persist() to satisfy SRP — data classes
should not own I/O. Use IntegrationAuditService.record() explicitly where
audit logging is needed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from victor.framework.vertical_integration import IntegrationResult

logger = logging.getLogger(__name__)


class IntegrationAuditService:
    """Write vertical integration results to an append-only JSONL audit log.

    One file is created per UTC day under ``base_path``, named
    ``integration_YYYYMMDD.jsonl``.  Each line is a complete JSON record.

    Example::

        from victor.observability import IntegrationAuditService

        service = IntegrationAuditService()
        service.record(result)
    """

    DEFAULT_BASE_PATH: Path = Path.home() / ".victor" / "logs" / "integration"

    def __init__(self, base_path: Optional[Path] = None) -> None:
        self._base_path = base_path or self.DEFAULT_BASE_PATH

    def record(
        self,
        result: "IntegrationResult",
        base_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """Append *result* to the daily JSONL audit file.

        Args:
            result: The integration result to record.
            base_path: Override the base directory for this call only.

        Returns:
            Path to the JSONL file that was written, or ``None`` on failure.
        """
        effective_path = base_path or self._base_path
        try:
            effective_path.mkdir(parents=True, exist_ok=True)
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            filepath = effective_path / f"integration_{date_str}.jsonl"
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "vertical_name": result.vertical_name,
                "result": result.to_dict(),
            }
            with open(filepath, "a") as f:
                f.write(json.dumps(record) + "\n")
            logger.debug("IntegrationResult appended to: %s", filepath)
            return filepath
        except Exception as e:
            logger.warning("Failed to record IntegrationResult: %s", e)
            return None
