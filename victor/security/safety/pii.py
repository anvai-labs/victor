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

"""Personally Identifiable Information (PII) detection and anonymization.

This module provides pattern-based detection of PII in data, along with
anonymization suggestions. Useful across all verticals for GDPR/CCPA
compliance and data privacy.

Example usage:
    from victor.security.safety.pii import (
        PIIScanner,
        detect_pii_columns,
        get_anonymization_suggestion,
    )

    # Detect PII columns in a dataframe
    columns = ["first_name", "email", "age", "ssn"]
    pii_cols = detect_pii_columns(columns)
    for col, pii_type in pii_cols:
        print(f"{col}: {pii_type} - {get_anonymization_suggestion(pii_type)}")

    # Scan text content
    scanner = PIIScanner()
    matches = scanner.scan_content(text_content)

The definitions live in the SDK (``victor_contracts.safety.pii``); this module
re-exports them so runtime code keeps one import path and one set of types.
"""

from victor_contracts.safety.pii import (
    ANONYMIZATION_SUGGESTIONS as ANONYMIZATION_SUGGESTIONS,
    PII_COLUMN_PATTERNS as PII_COLUMN_PATTERNS,
    PII_CONTENT_PATTERNS as PII_CONTENT_PATTERNS,
    PII_SEVERITY as PII_SEVERITY,
    PIIMatch as PIIMatch,
    PIIScanner as PIIScanner,
    PIISeverity as PIISeverity,
    PIIType as PIIType,
    detect_pii_columns as detect_pii_columns,
    detect_pii_in_content as detect_pii_in_content,
    get_anonymization_suggestion as get_anonymization_suggestion,
    get_pii_severity as get_pii_severity,
    get_pii_types as get_pii_types,
    get_safety_reminders as get_safety_reminders,
    has_pii as has_pii,
)

__all__ = [
    # Types
    "PIIType",
    "PIISeverity",
    "PIIMatch",
    # Patterns
    "PII_COLUMN_PATTERNS",
    "PII_CONTENT_PATTERNS",
    "PII_SEVERITY",
    "ANONYMIZATION_SUGGESTIONS",
    # Scanner
    "PIIScanner",
    # Functions
    "detect_pii_columns",
    "detect_pii_in_content",
    "get_anonymization_suggestion",
    "get_pii_severity",
    "has_pii",
    "get_pii_types",
    "get_safety_reminders",
]
