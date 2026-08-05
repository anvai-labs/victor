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

"""
Product bundle definitions for vertical consolidation.

Product bundles group related verticals into convenient packages
for installation and licensing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class BundleTier(Enum):
    """Product bundle tiers."""

    FOUNDATION = "foundation"  # Core framework only
    ENGINEERING = "engineering"  # Coding + DevOps + Security
    DATA = "data"  # RAG + DataAnalysis + Research
    PROFESSIONAL = "professional"  # Engineering + Data
    ENTERPRISE = "enterprise"  # All verticals


@dataclass
class ProductBundle:
    """Definition of a product bundle."""

    name: str
    display_name: str
    description: str
    verticals: List[str]
    tier: BundleTier
    dependencies: List[str] = field(default_factory=list)
    optional_verticals: List[str] = field(default_factory=list)

    def get_all_verticals(self) -> Set[str]:
        """
        Get all verticals for this bundle including dependencies.

        Returns:
            Set of all required vertical names
        """
        all_verticals = set(self.verticals)

        # Resolve dependencies
        for dep_name in self.dependencies:
            dep_bundle = get_bundle(dep_name)
            if dep_bundle:
                all_verticals.update(dep_bundle.get_all_verticals())

        return all_verticals

    def get_install_command(self) -> str:
        """
        Get the pip install command for this bundle.

        Returns:
            Install command string
        """
        return f"pip install victor-ai[{self.name}]"


# Standard product bundles
FOUNDATION_BUNDLE = ProductBundle(
    name="foundation",
    display_name="Victor Foundation",
    description="Core framework with no domain verticals",
    verticals=[],
    tier=BundleTier.FOUNDATION,
)

ENGINEERING_BUNDLE = ProductBundle(
    name="engineering",
    display_name="Victor Engineering Suite",
    description="Complete software engineering toolkit: coding, DevOps, and security",
    verticals=["victor-coding", "victor-devops"],
    tier=BundleTier.ENGINEERING,
    optional_verticals=["security"],  # Built-in
)

DATA_BUNDLE = ProductBundle(
    name="data",
    display_name="Victor Data Suite",
    description="Data analysis and research toolkit: RAG, data analysis, and research",
    verticals=["victor-rag", "victor-dataanalysis", "victor-research"],
    tier=BundleTier.DATA,
)

PROFESSIONAL_BUNDLE = ProductBundle(
    name="professional",
    display_name="Victor Professional",
    description="Full engineering and data capabilities",
    verticals=[
        "victor-coding",
        "victor-devops",
        "victor-rag",
        "victor-dataanalysis",
        "victor-research",
    ],
    tier=BundleTier.PROFESSIONAL,
    dependencies=["engineering", "data"],
    optional_verticals=["security", "iac", "classification"],
)

ENTERPRISE_BUNDLE = ProductBundle(
    name="enterprise",
    display_name="Victor Enterprise",
    description="Complete Victor platform with all verticals",
    verticals=[
        "victor-coding",
        "victor-devops",
        "victor-rag",
        "victor-dataanalysis",
        "victor-research",
    ],
    tier=BundleTier.ENTERPRISE,
    optional_verticals=["security", "iac", "classification", "benchmark"],
)

# Bundle registry
_BUNDLE_REGISTRY: Dict[str, ProductBundle] = {
    "foundation": FOUNDATION_BUNDLE,
    "engineering": ENGINEERING_BUNDLE,
    "data": DATA_BUNDLE,
    "professional": PROFESSIONAL_BUNDLE,
    "enterprise": ENTERPRISE_BUNDLE,
}


def get_bundle(bundle_name: str) -> Optional[ProductBundle]:
    """
    Get product bundle by name.

    Args:
        bundle_name: Name of the bundle

    Returns:
        ProductBundle if found, None otherwise
    """
    return _BUNDLE_REGISTRY.get(bundle_name)


def list_bundles() -> List[ProductBundle]:
    """
    List all available product bundles.

    Returns:
        List of all product bundles
    """
    return list(_BUNDLE_REGISTRY.values())


def get_bundles_for_vertical(vertical_name: str) -> List[ProductBundle]:
    """
    Get all bundles that include a specific vertical.

    Args:
        vertical_name: Name of the vertical

    Returns:
        List of bundles that include this vertical
    """
    return [
        bundle
        for bundle in _BUNDLE_REGISTRY.values()
        if vertical_name in bundle.verticals or vertical_name in bundle.optional_verticals
    ]


def resolve_bundle_dependencies(bundle_name: str) -> Set[str]:
    """
    Resolve all verticals for a bundle including dependencies.

    Args:
        bundle_name: Name of the bundle

    Returns:
        Set of all required vertical names
    """
    bundle = get_bundle(bundle_name)
    if not bundle:
        return set()

    return bundle.get_all_verticals()


def get_bundle_tier(tier: BundleTier) -> Optional[ProductBundle]:
    """
    Get bundle by tier.

    Args:
        tier: Bundle tier

    Returns:
        ProductBundle if found, None otherwise
    """
    for bundle in _BUNDLE_REGISTRY.values():
        if bundle.tier == tier:
            return bundle
    return None


def get_install_command(bundle_name: str) -> Optional[str]:
    """
    Get the pip install command for a bundle.

    Args:
        bundle_name: Name of the bundle

    Returns:
        Install command string, or None if bundle not found
    """
    bundle = get_bundle(bundle_name)
    if not bundle:
        return None
    return bundle.get_install_command()
