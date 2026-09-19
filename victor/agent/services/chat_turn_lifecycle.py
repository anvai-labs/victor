# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0 (the "License").

"""Per-turn setup and teardown callbacks bound as one lifecycle capability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class ChatTurnLifecycle:
    """Pair setup and teardown so their ownership cannot drift independently."""

    setup: Callable[..., Any] | None = None
    teardown: Callable[..., Any] | None = None
