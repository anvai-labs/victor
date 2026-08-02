# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""File editing and transaction support.

This module re-exports from the core victor.processing.editing module for backward
compatibility. The file editing infrastructure is now part of victor-core
and available to all verticals.

For new code, prefer importing directly from victor.processing.editing:
    from victor.processing.editing import FileEditor, EditTransaction
"""

# Re-export from victor-core
from victor_contracts.processing_runtime import (
    OperationType,
    EditOperation,
    EditTransaction,
    FileEditor,
)

__all__ = [
    "OperationType",
    "EditOperation",
    "EditTransaction",
    "FileEditor",
]
