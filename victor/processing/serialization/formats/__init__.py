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

"""Format encoders for token-optimized serialization.

This module provides a plugin-based architecture for serialization formats.
New formats can be easily added by:

1. Creating a class that inherits from FormatEncoder
2. Registering it with the FormatRegistry

Example of adding a custom format:

    from victor.processing.serialization.formats import FormatEncoder, FormatRegistry

    class MyCustomEncoder(FormatEncoder):
        format_id = "my_custom"
        format_name = "My Custom Format"

        def can_encode(self, data, characteristics):
            return characteristics.is_flat()

        def encode(self, data, characteristics, config):
            # Custom encoding logic
            return "encoded data"

        def suitability_score(self, characteristics):
            return 0.8 if characteristics.is_flat() else 0.0

    # Register the encoder
    FormatRegistry.register(MyCustomEncoder())
"""

from victor.processing.serialization.formats.base import (
    FormatEncoder,
    FormatRegistry,
    EncodingResult,
    get_format_registry,
    reset_format_registry,
)
from victor.processing.serialization.formats.json_encoder import (
    JSONEncoder,
    MinifiedJSONEncoder,
)
from victor.processing.serialization.formats.toon_encoder import TOONEncoder
from victor.processing.serialization.formats.csv_encoder import CSVEncoder
from victor.processing.serialization.formats.markdown_encoder import (
    MarkdownTableEncoder,
)
from victor.processing.serialization.formats.reference_encoder import ReferenceEncoder

__all__ = [
    # Base classes
    "FormatEncoder",
    "FormatRegistry",
    "EncodingResult",
    "get_format_registry",
    "reset_format_registry",
    # Built-in encoders
    "JSONEncoder",
    "MinifiedJSONEncoder",
    "TOONEncoder",
    "CSVEncoder",
    "MarkdownTableEncoder",
    "ReferenceEncoder",
]


def _register_builtin_encoders() -> None:
    """Register all built-in format encoders."""
    registry = get_format_registry()

    # Skip if already registered
    if registry.get_encoder(JSONEncoder.format_id):
        return

    # Register in order of preference for auto-selection
    # JSON first as baseline, then progressively more optimized formats
    registry.register(JSONEncoder(), priority=0)
    registry.register(MinifiedJSONEncoder(), priority=1)
    registry.register(TOONEncoder(), priority=2)
    registry.register(CSVEncoder(), priority=3)
    registry.register(MarkdownTableEncoder(), priority=4)
    registry.register(ReferenceEncoder(), priority=5)


# Auto-register built-in encoders on module import
_register_builtin_encoders()
