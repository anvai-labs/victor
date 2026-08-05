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

"""AST-based refactoring engine.

This module provides code refactoring capabilities using AST analysis
for safe and accurate code transformations.

Example usage:
    from victor_coding.refactor import get_refactor_manager, RefactorType
    from pathlib import Path

    # Get manager
    manager = get_refactor_manager()

    # Rename a symbol
    result = manager.rename_symbol(
        file_path=Path("main.py"),
        line=10,
        column=4,
        new_name="new_function_name",
    )

    # Extract code to function
    result = manager.extract_function(
        file_path=Path("main.py"),
        start_line=15,
        start_column=0,
        end_line=25,
        end_column=0,
        function_name="extracted_function",
    )

    # Get refactoring suggestions
    suggestions = manager.suggest_refactorings(Path("main.py"))
    for suggestion in suggestions:
        print(f"{suggestion.refactor_type.value}: {suggestion.reason}")

    # Undo if needed
    if result.success and result.can_undo():
        manager.undo(result)
"""

from victor_coding.refactor.protocol import (
    CodeEdit,
    CodePattern,
    RefactorCapabilities,
    RefactorPreview,
    RefactorRequest,
    RefactorResult,
    RefactorRisk,
    RefactorScope,
    RefactorSuggestion,
    RefactorType,
    SourceLocation,
    Symbol,
)
from victor_coding.refactor.analyzer import (
    BaseCodeAnalyzer,
    CodeAnalyzer,
    PythonAnalyzer,
    get_analyzer,
)
from victor_coding.refactor.transforms import (
    BaseTransform,
    CodeTransform,
    ExtractConstantTransform,
    ExtractFunctionTransform,
    ExtractVariableTransform,
    RenameFileTransform,
    RenameSymbolTransform,
)
from victor_coding.refactor.manager import (
    RefactorManager,
    get_refactor_manager,
    reset_refactor_manager,
)

__all__ = [
    # Protocol types
    "CodeEdit",
    "CodePattern",
    "RefactorCapabilities",
    "RefactorPreview",
    "RefactorRequest",
    "RefactorResult",
    "RefactorRisk",
    "RefactorScope",
    "RefactorSuggestion",
    "RefactorType",
    "SourceLocation",
    "Symbol",
    # Analyzers
    "BaseCodeAnalyzer",
    "CodeAnalyzer",
    "PythonAnalyzer",
    "get_analyzer",
    # Transforms
    "BaseTransform",
    "CodeTransform",
    "ExtractConstantTransform",
    "ExtractFunctionTransform",
    "ExtractVariableTransform",
    "RenameFileTransform",
    "RenameSymbolTransform",
    # Manager
    "RefactorManager",
    "get_refactor_manager",
    "reset_refactor_manager",
]
