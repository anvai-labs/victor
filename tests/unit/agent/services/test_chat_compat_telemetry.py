# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import pytest


def test_chat_compat_telemetry_module_removed():
    with pytest.raises(ImportError, match="chat_compat_telemetry"):
        import victor.agent.services.chat_compat_telemetry  # noqa: F401
