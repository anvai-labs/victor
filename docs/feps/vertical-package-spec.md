# Victor Vertical Package Specification

**Status**: Draft
**Version**: 1.0.0
**Author**: Vijaykumar Singh <vijay@anvaiops.com>
**Created**: 2025-01-09

## Overview

This specification defines the format and structure of `victor-vertical.toml`, a metadata file that enables third-party developers to create and distribute vertical packages for Victor AI.

## Purpose

Victor verticals are domain-specific extensions that provide:
- Specialized tools for specific domains (e.g., security, ML, monitoring)
- Domain-specific workflows and agents
- Custom capabilities and integrations

The `victor-vertical.toml` file serves as the package manifest, enabling:
- Automatic discovery and installation
- Dependency management
- Compatibility checking
- Marketplace listing

## File Location

The `victor-vertical.toml` file must be placed in the root directory of the Python package, alongside `pyproject.toml`.

Example structure:
```
victor-security/
├── victor_security/
│   ├── __init__.py
│   └── security_assistant.py
├── victor-vertical.toml    # <-- This file
├── pyproject.toml
└── README.md
```

## TOML Structure

### Required Fields

```toml
[vertical]
name = "security"              # Unique identifier (lowercase, alphanumeric)
version = "1.0.0"              # Semantic version
description = "Security analysis and vulnerability scanning"
authors = [{name = "Author Name", email = "author@example.com"}]
license = "Apache-2.0"
requires_victor = ">=0.5.0"    # Minimum Victor version

[vertical.class]
module = "victor_security.security_assistant"
class_name = "SecurityAssistant"
```

### Optional Fields

```toml
[vertical]
# Package metadata
python_package = "victor-security"
homepage = "https://github.com/user/victor-security"
repository = "https://github.com/user/victor-security"
documentation = "https://victor.dev/verticals/security"
issues = "https://github.com/user/victor-security/issues"

# Categorization
category = "security"
tags = ["security", "sast", "vulnerability", "scanning"]

[vertical.class]
# Advertised capabilities
provides_tools = ["sast_scan", "dependency_check", "security_audit"]
provides_workflows = ["security_review", "vulnerability_scan"]
provides_capabilities = ["security_analysis"]

[vertical.dependencies]
# Python dependencies
python = ["bandit>=1.7.0", "safety>=2.0.0"]

# Vertical dependencies (must be installed first)
verticals = ["coding"]

[vertical.compatibility]
# Provider requirements
requires_tool_calling = true
preferred_providers = ["anthropic", "openai"]
min_context_window = 100000
python_version = ">=3.10"

[vertical.security]
# Security metadata
signed = false
verified_author = false
permissions = ["network", "filesystem"]

[vertical.installation]
# Installation hints
install_command = "pip install victor-security"
post_install = "victor vertical info security"
```

## Field Specifications

### [vertical] Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique vertical identifier (lowercase, alphanumeric, must start with letter) |
| `version` | string | Yes | Semantic version (e.g., "1.0.0") |
| `description` | string | Yes | Brief description of the vertical's purpose |
| `authors` | array | Yes | List of author objects with `name` and optional `email` |
| `license` | string | Yes | SPDX license identifier |
| `requires_victor` | string | Yes | Minimum Victor version (e.g., ">=0.5.0") |
| `python_package` | string | No | Python package name on PyPI |
| `homepage` | string (URL) | No | Project homepage |
| `repository` | string (URL) | No | Source code repository |
| `documentation` | string (URL) | No | Documentation URL |
| `issues` | string (URL) | No | Issue tracker URL |
| `category` | string | No | Category for marketplace grouping |
| `tags` | array of strings | No | Searchable tags |

### [vertical.class] Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `module` | string | Yes | Python module path (e.g., "victor_security.security_assistant") |
| `class_name` | string | Yes | Name of the VerticalBase subclass |
| `provides_tools` | array of strings | No | List of tools this vertical provides |
| `provides_workflows` | array of strings | No | List of workflows this vertical provides |
| `provides_capabilities` | array of strings | No | List of capabilities this vertical provides |

### [vertical.dependencies] Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `python` | array of strings | No | Python package dependencies (PEP 508 format) |
| `verticals` | array of strings | No | Other verticals this depends on |

### [vertical.compatibility] Section

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `requires_tool_calling` | boolean | No | true | Whether vertical requires tool-calling support |
| `preferred_providers` | array of strings | No | [] | Preferred LLM providers |
| `min_context_window` | integer | No | null | Minimum context window size |
| `python_version` | string | No | ">=3.10" | Minimum Python version |
| `platforms` | array of strings | No | ["linux", "macos", "windows"] | Supported platforms |

### [vertical.security] Section

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `signed` | boolean | No | false | Whether package is cryptographically signed |
| `signature_url` | string (URL) | No | null | URL to signature file |
| `verified_author` | boolean | No | false | Whether author identity is verified |
| `permissions` | array of strings | No | [] | Required permissions (network, filesystem, etc.) |

### [vertical.installation] Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `install_command` | string | No | Recommended installation command |
| `post_install` | string | No | Command to run after installation |

## Validation Rules

1. **Name Validation**:
   - Must be lowercase
   - Must start with a letter
   - Can contain letters, numbers, and underscores
   - Cannot use reserved names: `victor`, `core`, `tools`, `providers`, `config`, `ui`, `tests`, `framework`, `agent`, `workflows`

2. **Version Validation**:
   - Must follow semantic versioning (MAJOR.MINOR.PATCH)
   - Validated using `packaging.version.Version`

3. **Victor Requirement Validation**:
   - Must be a valid PEP 508 requirement
   - Auto-prepends "victor-ai" if not present

4. **URL Validation**:
   - Optional URL fields are validated for proper format if provided

## Entry Point Registration

Verticals must also register themselves in `pyproject.toml`:

```toml
[project.entry-points."victor.verticals"]
security = "victor_security.security_assistant:SecurityAssistant"
```

This allows Victor to discover and load the vertical automatically.

## Example Complete File

```toml
[vertical]
name = "security"
version = "1.0.0"
description = "Security analysis and vulnerability scanning for codebases"
authors = [
    {name = "Vijaykumar Singh", email = "vijay@anvaiops.com"}
]
license = "Apache-2.0"
requires_victor = ">=0.5.0"

python_package = "victor-security"
homepage = "https://github.com/vijay-singh/victor-security"
repository = "https://github.com/vijay-singh/victor-security"
documentation = "https://victor.dev/verticals/security"
issues = "https://github.com/vijay-singh/victor-security/issues"

category = "security"
tags = ["security", "sast", "vulnerability", "scanning", "audit"]

[vertical.class]
module = "victor_security.security_assistant"
class_name = "SecurityAssistant"
provides_tools = [
    "sast_scan",
    "dependency_check",
    "security_audit",
    "vulnerability_scan"
]
provides_workflows = [
    "security_review",
    "vulnerability_assessment",
    "compliance_check"
]
provides_capabilities = ["security_analysis"]

[vertical.dependencies]
python = [
    "bandit>=1.7.0",
    "safety>=2.0.0",
    "semgrep>=1.0.0"
]
verticals = ["coding"]

[vertical.compatibility]
requires_tool_calling = true
preferred_providers = ["anthropic", "openai"]
min_context_window = 100000
python_version = ">=3.10"
platforms = ["linux", "macos", "windows"]

[vertical.security]
signed = false
verified_author = false
permissions = ["network", "filesystem:read"]

[vertical.installation]
install_command = "pip install victor-security"
post_install = "victor vertical info security"
```

## Integration with Victor

### Installation

```bash
# From PyPI
victor vertical install victor-security

# From git
victor vertical install "git+https://github.com/user/victor-security.git"

# From local path
victor vertical install ./path/to/victor-security
```

### Listing

```bash
# List all verticals
victor vertical list

# List only installed
victor vertical list --source installed

# List only built-in
victor vertical list --source builtin

# List available from registry
victor vertical list --source available
```

### Search

```bash
# Search by name, description, or tags
victor vertical search security
victor vertical search "vulnerability scanning"
```

### Information

```bash
# Show detailed information
victor vertical info security
```

## Best Practices

1. **Naming**: Use descriptive, lowercase names for your vertical
2. **Versioning**: Follow semantic versioning strictly
3. **Dependencies**: Pin minimum versions for Python dependencies
4. **Documentation**: Provide comprehensive documentation URL
5. **Testing**: Test your vertical with multiple Victor versions
6. **Compatibility**: Specify realistic provider and platform requirements

## Migration from Legacy Format

If you have an existing vertical without `victor-vertical.toml`, create one using the template:

```bash
# Generate a template
victor vertical create my-vertical --description "My vertical"

# Edit the generated victor-vertical.toml
# Add to pyproject.toml entry points
# Test with victor vertical info my-vertical
```

## Appendix: Reserved Names

The following names are reserved and cannot be used as vertical names:
- `victor`
- `core`
- `tools`
- `providers`
- `config`
- `ui`
- `tests`
- `framework`
- `agent`
- `workflows`

## Changelog

### 1.0.0 (2025-01-09)
- Initial specification
- Define TOML structure and validation rules
- Document integration with Victor CLI
