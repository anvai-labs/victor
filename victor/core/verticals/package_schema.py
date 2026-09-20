"""
Vertical Package Metadata Schema.

This module defines the schema for victor-vertical.toml files,
which provide metadata for third-party vertical packages.

The schema uses Pydantic for validation and type safety.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from victor.core.verticals.base import VerticalBase


def normalize_victor_requirement(requirement: str) -> str:
    """Normalize Victor requirement strings to a full requirement expression."""
    req = requirement.strip()
    if req.startswith("victor-ai"):
        return req
    return f"victor-ai{req}"


def is_victor_version_compatible(current_version: str, requirement: str) -> bool:
    """Return True if the current Victor version satisfies the requirement."""
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version

        req = Requirement(normalize_victor_requirement(requirement))
        return Version(current_version) in req.specifier
    except Exception:
        # Preserve existing behavior: treat parsing issues as non-fatal.
        return True


class AuthorInfo(BaseModel):
    """Author information for a vertical package."""

    name: str
    email: Optional[str] = None


class VerticalClassSpec(BaseModel):
    """Specification for the vertical class entry point."""

    module: str
    class_name: str
    provides_tools: List[str] = Field(default_factory=list)
    provides_workflows: List[str] = Field(default_factory=list)
    provides_capabilities: List[str] = Field(default_factory=list)


class VerticalDependencies(BaseModel):
    """Dependencies for a vertical package."""

    python: List[str] = Field(default_factory=list)
    verticals: List[str] = Field(default_factory=list)


class VerticalCompatibility(BaseModel):
    """Compatibility requirements for a vertical package."""

    min_context_window: Optional[int] = None
    requires_tool_calling: bool = True
    preferred_providers: List[str] = Field(default_factory=list)
    platforms: List[str] = Field(default_factory=lambda: ["linux", "macos", "windows"])
    python_version: str = ">=3.12"


class VerticalSecurity(BaseModel):
    """Security metadata for a vertical package."""

    signed: bool = False
    signature_url: Optional[str] = None
    verified_author: bool = False
    permissions: List[str] = Field(default_factory=list)


class VerticalPackageMetadata(BaseModel):
    """
    Complete metadata for a vertical package.

    This model validates the structure of victor-vertical.toml files.
    It ensures all required fields are present and valid.

    Attributes:
        name: Vertical name (must be unique, lowercase, alphanumeric)
        version: Semantic version (e.g., "1.0.0")
        description: Brief description of the vertical
        authors: List of authors
        license: SPDX license identifier
        requires_victor: Minimum Victor version required
        python_package: Python package name on PyPI (optional)
        homepage: Homepage URL (optional)
        repository: Repository URL (optional)
        documentation: Documentation URL (optional)
        issues: Issue tracker URL (optional)
        category: Category for marketplace grouping (optional)
        tags: List of tags for search (optional)
        class_spec: Class specification (required)
        dependencies: Runtime dependencies (optional)
        compatibility: Compatibility requirements (optional)
        security: Security metadata (optional)
        installation: Installation hints (optional)
    """

    # Required fields
    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$")
    version: str
    description: str
    authors: List[AuthorInfo]
    license: str
    requires_victor: str
    class_spec: VerticalClassSpec

    # Optional fields
    python_package: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    documentation: Optional[str] = None
    issues: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    dependencies: VerticalDependencies = Field(default_factory=VerticalDependencies)
    compatibility: VerticalCompatibility = Field(default_factory=VerticalCompatibility)
    security: VerticalSecurity = Field(default_factory=VerticalSecurity)
    installation: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that the vertical name is not reserved."""
        reserved = {
            "victor",
            "core",
            "tools",
            "providers",
            "config",
            "ui",
            "tests",
            "framework",
            "agent",
            "workflows",
        }
        if v in reserved:
            raise ValueError(f"'{v}' is a reserved name")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate that the version is a valid semantic version."""
        try:
            from packaging.version import Version

            Version(v)
        except Exception as e:
            raise ValueError(f"Invalid version '{v}': {e}") from e
        return v

    @field_validator("requires_victor")
    @classmethod
    def validate_requires_victor(cls, v: str) -> str:
        """Validate that the Victor version requirement is valid."""
        try:
            from packaging.requirements import Requirement

            Requirement(normalize_victor_requirement(v))
        except Exception as e:
            raise ValueError(f"Invalid Victor requirement '{v}': {e}") from e
        return v

    @classmethod
    def from_toml(cls, path: Path) -> "VerticalPackageMetadata":
        """
        Load and validate metadata from a victor-vertical.toml file.

        Args:
            path: Path to the TOML file

        Returns:
            Validated VerticalPackageMetadata instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValidationError: If the TOML is invalid
        """
        try:
            import tomllib

            toml_data = tomllib.loads(path.read_text())
        except ImportError:
            # Python < 3.11, use tomli
            import tomli as tomllib

            toml_data = tomllib.loads(path.read_text())

        # Extract [vertical] section
        vertical_data = toml_data.get("vertical", {})

        # Map keys to match model
        if "class" in vertical_data:
            vertical_data["class_spec"] = vertical_data.pop("class")

        return cls(**vertical_data)

    def to_toml(self, path: Path) -> None:
        """
        Write metadata to a victor-vertical.toml file.

        Args:
            path: Path to write the TOML file
        """
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        # Build TOML structure
        data = {"vertical": self.dict(exclude_none=True, by_alias=True)}

        # Map class_spec back to class
        if "class_spec" in data["vertical"]:
            data["vertical"]["class"] = data["vertical"].pop("class_spec")

        # Write TOML
        try:
            import tomli_w

            toml_content = tomli_w.dumps(data)
        except ImportError:
            # Fallback: manual TOML generation
            toml_content = self._generate_toml_fallback()

        path.write_text(toml_content)

    def _generate_toml_fallback(self) -> str:
        """Generate TOML content without external library."""
        lines = ["[vertical]"]

        # Simple fields
        for field in ["name", "version", "description", "license", "requires_victor"]:
            value = getattr(self, field, None)
            if value:
                lines.append(f'{field} = "{value}"')

        # Authors
        if self.authors:
            lines.append("authors = [")
            for author in self.authors:
                if author.email:
                    lines.append(f'    {{name = "{author.name}", email = "{author.email}"}},')
                else:
                    lines.append(f'    {{name = "{author.name}"}},')
            lines.append("]")

        # Optional fields
        for field in [
            "python_package",
            "homepage",
            "repository",
            "documentation",
            "issues",
            "category",
        ]:
            value = getattr(self, field, None)
            if value:
                lines.append(f'{field} = "{value}"')

        # Tags
        if self.tags:
            tags_str = ", ".join(f'"{tag}"' for tag in self.tags)
            lines.append(f"tags = [{tags_str}]")

        # Class spec
        if self.class_spec:
            lines.append("\n[vertical.class]")
            lines.append(f'module = "{self.class_spec.module}"')
            lines.append(f'class_name = "{self.class_spec.class_name}"')

            if self.class_spec.provides_tools:
                tools_str = ", ".join(f'"{t}"' for t in self.class_spec.provides_tools)
                lines.append(f"provides_tools = [{tools_str}]")

            if self.class_spec.provides_workflows:
                workflows_str = ", ".join(f'"{w}"' for w in self.class_spec.provides_workflows)
                lines.append(f"provides_workflows = [{workflows_str}]")

            if self.class_spec.provides_capabilities:
                caps_str = ", ".join(f'"{c}"' for c in self.class_spec.provides_capabilities)
                lines.append(f"provides_capabilities = [{caps_str}]")

        return "\n".join(lines)


# Example victor-vertical.toml template
VICTOR_VERTICAL_TOML_TEMPLATE = """
# victor-vertical.toml - Vertical Package Metadata

[vertical]
# Required: Package identity
name = "myvertical"                   # Must be unique, lowercase, alphanumeric
version = "0.1.0"                     # Semantic versioning
description = "Brief description of your vertical"
authors = [{name = "Your Name", email = "you@example.com"}]
license = "Apache-2.0"

# Required: Victor compatibility
requires_victor = ">=0.5.0"           # Minimum Victor version

# Optional: Python package info
python_package = "victor-myvertical"  # PyPI package name
homepage = "https://github.com/yourusername/victor-myvertical"
repository = "https://github.com/yourusername/victor-myvertical"
documentation = "https://victor.dev/verticals/myvertical"
issues = "https://github.com/yourusername/victor-myvertical/issues"

# Optional: Categorization
category = "general"
tags = ["tag1", "tag2"]

[vertical.class]
# Required: Entry point specification
module = "victor_myvertical"
class_name = "MyVertical"

# Optional: Capability advertisement
provides_tools = ["tool1", "tool2"]
provides_workflows = ["workflow1"]
provides_capabilities = ["capability1"]

[vertical.dependencies]
# Optional: Runtime dependencies
python = [
    "requests>=2.0",
]
verticals = []

[vertical.compatibility]
# Optional: Provider requirements
requires_tool_calling = true
preferred_providers = ["anthropic", "openai"]
python_version = ">=3.12"

[vertical.security]
# Optional: Security metadata
signed = false
permissions = []

[vertical.installation]
# Optional: Installation hints
install_command = "pip install victor-myvertical"
"""
