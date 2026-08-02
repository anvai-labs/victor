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

"""Cache configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CacheConfig:
    """Cache configuration settings."""

    # Memory cache settings
    memory_max_size: int = 1000  # Max items in memory
    memory_ttl: int = 300  # 5 minutes TTL for memory cache

    # Disk cache settings
    disk_max_size: int = 1024 * 1024 * 1024  # 1GB max size
    disk_ttl: int = 86400 * 7  # 7 days TTL for disk cache
    disk_path: Optional[Path] = None  # Cache directory

    # Cache behavior
    enable_memory: bool = True  # Enable L1 memory cache
    enable_disk: bool = True  # Enable L2 disk cache
    auto_evict: bool = True  # Auto-evict old entries

    def __post_init__(self) -> None:
        """Set default cache path if not provided."""
        if self.disk_path is None:
            from victor.config.settings import get_project_paths

            self.disk_path = get_project_paths().global_cache_dir

        # Ensure cache directory exists
        if self.enable_disk:
            self.disk_path.mkdir(parents=True, exist_ok=True)
