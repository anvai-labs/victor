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

"""File type detection for Victor.

This module provides generic file type detection based on extensions,
filenames, and shebangs. Used across all verticals for file categorization.

Cross-Vertical Benefits:
- Coding vertical: Language detection for syntax highlighting, LSP
- DevOps vertical: Config file detection for validation
- Data Analysis vertical: Data format detection for parsing
- Research vertical: Document type detection for indexing

Example usage:
    from victor.processing.file_types import (
        detect_file_type,
        get_file_category,
        is_code_file,
        is_config_file,
        FileCategory,
        FileTypeRegistry,
    )

    # Detect file type
    file_type = detect_file_type(Path("main.py"))
    print(f"Type: {file_type.name}, Category: {file_type.category}")

    # Check category
    if is_code_file(Path("app.js")):
        print("This is code!")

    # Register custom type
    registry = FileTypeRegistry.get_instance()
    registry.register(FileType(
        name="proto",
        extensions=[".proto"],
        category=FileCategory.CODE,
    ))
"""

from victor.processing.file_types.detector import (
    # Types
    FileCategory,
    FileType,
    FileTypeDetector,
    # Registry
    FileTypeRegistry,
    # Functions
    detect_file_type,
    get_file_category,
    is_code_file,
    is_config_file,
    is_data_file,
)

__all__ = [
    # Types
    "FileCategory",
    "FileType",
    "FileTypeDetector",
    # Registry
    "FileTypeRegistry",
    # Functions
    "detect_file_type",
    "get_file_category",
    "is_code_file",
    "is_config_file",
    "is_data_file",
]
