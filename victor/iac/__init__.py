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

"""IaC Scanner Module - Security scanning for Infrastructure-as-Code.

This module provides comprehensive security scanning for IaC files,
detecting misconfigurations, hardcoded secrets, and security issues.

Supported Platforms:
- Terraform (.tf, .tfvars)
- Docker (Dockerfile, *.dockerfile)
- Docker Compose (docker-compose.yml)
- Kubernetes (*.yaml, *.yml manifests)

Usage:
    from victor.iac import IaCManager

    manager = IaCManager("/path/to/project")
    result = await manager.scan()

    print(f"Found {len(result.findings)} security findings")
    for finding in result.findings:
        print(f"  [{finding.severity.value}] {finding.rule_id}: {finding.message}")
"""

from .manager import IaCManager
from .protocol import (
    COMMON_SECRET_PATTERNS,
    Category,
    IaCConfig,
    IaCFinding,
    IaCPlatform,
    IaCResource,
    IaCScannerProtocol,
    ScanPolicy,
    ScanResult,
    SecretPattern,
    Severity,
)
from .scanners import (
    DockerComposeScanner,
    DockerScanner,
    KubernetesScanner,
    TerraformScanner,
    get_all_scanners,
    get_scanner,
)

__all__ = [
    # Manager
    "IaCManager",
    # Protocols
    "IaCScannerProtocol",
    # Data classes
    "IaCPlatform",
    "Severity",
    "Category",
    "IaCResource",
    "IaCFinding",
    "IaCConfig",
    "ScanResult",
    "ScanPolicy",
    "SecretPattern",
    # Scanners
    "TerraformScanner",
    "DockerScanner",
    "DockerComposeScanner",
    "KubernetesScanner",
    # Registry functions
    "get_scanner",
    "get_all_scanners",
    # Utilities
    "COMMON_SECRET_PATTERNS",
]
