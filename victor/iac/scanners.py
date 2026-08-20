from __future__ import annotations

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

"""IaC Scanners - Security scanners for different IaC platforms.

This module provides concrete implementations of the scanner protocol
for various Infrastructure-as-Code platforms.
"""

import logging
import re
from pathlib import Path

import yaml

from .protocol import (
    COMMON_SECRET_PATTERNS,
    Category,
    IaCConfig,
    IaCFinding,
    IaCPlatform,
    IaCResource,
    IaCScannerProtocol,
    ScanPolicy,
    Severity,
)

logger = logging.getLogger(__name__)


def scan_for_secrets(content: str, file_path: Path, findings: list[IaCFinding]) -> None:
    """Scan content for common secret patterns.

    Args:
        content: File content to scan
        file_path: Path for reporting
        findings: List to append findings to
    """
    for pattern in COMMON_SECRET_PATTERNS:
        for match in re.finditer(pattern.pattern, content):
            # Find line number
            line_num = content[: match.start()].count("\n") + 1

            findings.append(
                IaCFinding(
                    rule_id=f"SECRET-{pattern.name.upper().replace(' ', '-')}",
                    severity=pattern.severity,
                    category=Category.SECRETS,
                    message=pattern.description,
                    description=f"Found {pattern.name} at line {line_num}",
                    file_path=file_path,
                    line_number=line_num,
                    remediation="Use environment variables or secret management (Vault, AWS Secrets Manager)",
                    cwe_id="CWE-798",  # Use of Hard-coded Credentials
                )
            )


# =============================================================================
# Terraform Scanner
# =============================================================================


class TerraformScanner(IaCScannerProtocol):
    """Scanner for Terraform configurations."""

    @property
    def platform(self) -> IaCPlatform:
        return IaCPlatform.TERRAFORM

    async def detect_files(self, root_path: Path) -> list[Path]:
        """Find Terraform files."""
        files = []
        for pattern in ["**/*.tf", "**/*.tfvars"]:
            files.extend(root_path.glob(pattern))
        # Exclude .terraform directory
        files = [f for f in files if ".terraform" not in str(f)]
        return files

    async def parse_config(self, config_path: Path) -> IaCConfig:
        """Parse a Terraform file (simplified HCL parsing)."""
        content = config_path.read_text(encoding="utf-8")
        resources = []

        # Simple regex-based HCL parsing for resource blocks
        resource_pattern = r'resource\s+"([^"]+)"\s+"([^"]+)"\s*\{'
        for match in re.finditer(resource_pattern, content):
            resource_type = match.group(1)
            resource_name = match.group(2)
            line_num = content[: match.start()].count("\n") + 1

            resources.append(
                IaCResource(
                    resource_type=resource_type,
                    name=resource_name,
                    file_path=config_path,
                    line_number=line_num,
                )
            )

        # Parse variables
        variables = {}
        var_pattern = r'variable\s+"([^"]+)"\s*\{'
        for match in re.finditer(var_pattern, content):
            variables[match.group(1)] = None

        return IaCConfig(
            platform=self.platform,
            file_path=config_path,
            resources=resources,
            variables=variables,
            raw_content=content,
        )

    async def scan(self, config: IaCConfig, policy: ScanPolicy | None = None) -> list[IaCFinding]:
        """Scan Terraform configuration for security issues."""
        findings: list[IaCFinding] = []
        content = config.raw_content

        # Scan for secrets
        scan_for_secrets(content, config.file_path, findings)

        # AWS S3 Bucket without encryption
        if "aws_s3_bucket" in content and "server_side_encryption" not in content:
            findings.append(
                IaCFinding(
                    rule_id="TF-AWS-001",
                    severity=Severity.HIGH,
                    category=Category.ENCRYPTION,
                    message="S3 bucket without server-side encryption",
                    description="S3 buckets should have encryption enabled",
                    file_path=config.file_path,
                    remediation="Add server_side_encryption_configuration block",
                    cwe_id="CWE-311",
                )
            )

        # AWS S3 Bucket public access
        if re.search(r'acl\s*=\s*"public-read"', content):
            findings.append(
                IaCFinding(
                    rule_id="TF-AWS-002",
                    severity=Severity.CRITICAL,
                    category=Category.PERMISSIONS,
                    message="S3 bucket with public read access",
                    description="Public S3 buckets can expose sensitive data",
                    file_path=config.file_path,
                    remediation="Remove public-read ACL and use bucket policies",
                    cwe_id="CWE-284",
                )
            )

        # AWS Security Group with 0.0.0.0/0
        if re.search(r'cidr_blocks\s*=\s*\["0\.0\.0\.0/0"\]', content):
            findings.append(
                IaCFinding(
                    rule_id="TF-AWS-003",
                    severity=Severity.HIGH,
                    category=Category.NETWORK,
                    message="Security group allows traffic from all IPs (0.0.0.0/0)",
                    description="Overly permissive security groups increase attack surface",
                    file_path=config.file_path,
                    remediation="Restrict CIDR blocks to known IP ranges",
                    cwe_id="CWE-284",
                )
            )

        # RDS without encryption
        if "aws_db_instance" in content and "storage_encrypted" not in content:
            findings.append(
                IaCFinding(
                    rule_id="TF-AWS-004",
                    severity=Severity.HIGH,
                    category=Category.ENCRYPTION,
                    message="RDS instance without storage encryption",
                    description="Database storage should be encrypted at rest",
                    file_path=config.file_path,
                    remediation="Set storage_encrypted = true",
                    cwe_id="CWE-311",
                )
            )

        # CloudWatch Logs without retention
        if "aws_cloudwatch_log_group" in content and "retention_in_days" not in content:
            findings.append(
                IaCFinding(
                    rule_id="TF-AWS-005",
                    severity=Severity.LOW,
                    category=Category.LOGGING,
                    message="CloudWatch Log Group without retention policy",
                    description="Logs without retention can cause storage costs to grow",
                    file_path=config.file_path,
                    remediation="Set retention_in_days to appropriate value",
                )
            )

        # IAM policy with wildcard
        if re.search(r'"Action"\s*:\s*\[?\s*"\*"\s*\]?', content):
            findings.append(
                IaCFinding(
                    rule_id="TF-AWS-006",
                    severity=Severity.CRITICAL,
                    category=Category.PERMISSIONS,
                    message="IAM policy with wildcard action",
                    description="Wildcard actions grant excessive permissions",
                    file_path=config.file_path,
                    remediation="Use least-privilege principle with specific actions",
                    cwe_id="CWE-732",
                )
            )

        return findings


# =============================================================================
# Docker Scanner
# =============================================================================


class DockerScanner(IaCScannerProtocol):
    """Scanner for Dockerfiles."""

    @property
    def platform(self) -> IaCPlatform:
        return IaCPlatform.DOCKER

    async def detect_files(self, root_path: Path) -> list[Path]:
        """Find Dockerfiles."""
        files = []
        for pattern in ["**/Dockerfile", "**/Dockerfile.*", "**/*.dockerfile"]:
            files.extend(root_path.glob(pattern))
        return files

    async def parse_config(self, config_path: Path) -> IaCConfig:
        """Parse a Dockerfile."""
        content = config_path.read_text(encoding="utf-8")
        resources = []

        # Extract FROM images as resources
        from_pattern = r"^FROM\s+([^\s]+)(?:\s+AS\s+(\S+))?$"
        for match in re.finditer(from_pattern, content, re.MULTILINE | re.IGNORECASE):
            image = match.group(1)
            stage = match.group(2) or "main"
            line_num = content[: match.start()].count("\n") + 1

            resources.append(
                IaCResource(
                    resource_type="docker_image",
                    name=f"{stage}:{image}",
                    file_path=config_path,
                    line_number=line_num,
                    properties={"image": image, "stage": stage},
                )
            )

        return IaCConfig(
            platform=self.platform,
            file_path=config_path,
            resources=resources,
            raw_content=content,
        )

    async def scan(self, config: IaCConfig, policy: ScanPolicy | None = None) -> list[IaCFinding]:
        """Scan Dockerfile for security issues."""
        findings: list[IaCFinding] = []
        content = config.raw_content

        # Scan for secrets
        scan_for_secrets(content, config.file_path, findings)

        # Running as root
        if not re.search(r"^USER\s+(?!root)\S+", content, re.MULTILINE):
            findings.append(
                IaCFinding(
                    rule_id="DOCKER-001",
                    severity=Severity.MEDIUM,
                    category=Category.PERMISSIONS,
                    message="Container runs as root user",
                    description="Running containers as root increases security risk",
                    file_path=config.file_path,
                    remediation="Add USER instruction with non-root user",
                    cwe_id="CWE-250",
                )
            )

        # Using latest tag
        if re.search(r"^FROM\s+\S+:latest", content, re.MULTILINE | re.IGNORECASE):
            findings.append(
                IaCFinding(
                    rule_id="DOCKER-002",
                    severity=Severity.MEDIUM,
                    category=Category.BEST_PRACTICE,
                    message="Using 'latest' tag for base image",
                    description="Latest tag can lead to unpredictable builds",
                    file_path=config.file_path,
                    remediation="Pin base image to specific version",
                )
            )

        # ADD instead of COPY for files
        if re.search(r"^ADD\s+(?!https?://)[^\s]+\s+", content, re.MULTILINE):
            findings.append(
                IaCFinding(
                    rule_id="DOCKER-003",
                    severity=Severity.LOW,
                    category=Category.BEST_PRACTICE,
                    message="Using ADD instead of COPY for local files",
                    description="COPY is preferred for simple file operations",
                    file_path=config.file_path,
                    remediation="Use COPY for local files, ADD only for URLs/tarballs",
                )
            )

        # Healthcheck missing
        if "HEALTHCHECK" not in content:
            findings.append(
                IaCFinding(
                    rule_id="DOCKER-004",
                    severity=Severity.LOW,
                    category=Category.BEST_PRACTICE,
                    message="No HEALTHCHECK instruction",
                    description="Healthchecks help orchestrators manage containers",
                    file_path=config.file_path,
                    remediation="Add HEALTHCHECK instruction",
                )
            )

        # Hardcoded secrets in ENV
        env_pattern = r"^ENV\s+(\S+)\s*=?\s*['\"]?([^'\"\n]+)"
        for match in re.finditer(env_pattern, content, re.MULTILINE):
            var_name = match.group(1).upper()
            if any(kw in var_name for kw in ["PASSWORD", "SECRET", "KEY", "TOKEN"]):
                line_num = content[: match.start()].count("\n") + 1
                findings.append(
                    IaCFinding(
                        rule_id="DOCKER-005",
                        severity=Severity.HIGH,
                        category=Category.SECRETS,
                        message=f"Potential secret in ENV: {var_name}",
                        description="Secrets should not be hardcoded in Dockerfile",
                        file_path=config.file_path,
                        line_number=line_num,
                        remediation="Use build-time secrets or runtime environment variables",
                        cwe_id="CWE-798",
                    )
                )

        # Apt-get without cleanup
        if re.search(r"apt-get\s+install", content) and "rm -rf /var/lib/apt" not in content:
            findings.append(
                IaCFinding(
                    rule_id="DOCKER-006",
                    severity=Severity.LOW,
                    category=Category.BEST_PRACTICE,
                    message="apt-get without cleanup",
                    description="Apt cache should be cleaned to reduce image size",
                    file_path=config.file_path,
                    remediation="Add '&& rm -rf /var/lib/apt/lists/*' after apt-get",
                )
            )

        return findings


# =============================================================================
# Kubernetes Scanner
# =============================================================================


class KubernetesScanner(IaCScannerProtocol):
    """Scanner for Kubernetes manifests."""

    @property
    def platform(self) -> IaCPlatform:
        return IaCPlatform.KUBERNETES

    async def detect_files(self, root_path: Path) -> list[Path]:
        """Find Kubernetes manifest files."""
        files = []
        for pattern in ["**/*.yaml", "**/*.yml"]:
            files.extend(root_path.glob(pattern))

        # Filter to only K8s manifests
        k8s_files = []
        for f in files:
            try:
                content = f.read_text(encoding="utf-8")
                if "apiVersion:" in content and "kind:" in content:
                    k8s_files.append(f)
            except Exception:
                continue
        return k8s_files

    async def parse_config(self, config_path: Path) -> IaCConfig:
        """Parse a Kubernetes manifest."""
        content = config_path.read_text(encoding="utf-8")
        resources = []

        try:
            # Handle multi-document YAML
            docs = list(yaml.safe_load_all(content))
            for doc in docs:
                if not doc:
                    continue
                kind = doc.get("kind", "Unknown")
                metadata = doc.get("metadata", {})
                name = metadata.get("name", "unnamed")

                resources.append(
                    IaCResource(
                        resource_type=kind,
                        name=name,
                        file_path=config_path,
                        line_number=1,
                        properties=doc,
                        tags=metadata.get("labels", {}),
                    )
                )
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse K8s manifest {config_path}: {e}")

        return IaCConfig(
            platform=self.platform,
            file_path=config_path,
            resources=resources,
            raw_content=content,
        )

    async def scan(self, config: IaCConfig, policy: ScanPolicy | None = None) -> list[IaCFinding]:
        """Scan Kubernetes manifests for security issues."""
        findings: list[IaCFinding] = []

        # Scan for secrets in raw content
        scan_for_secrets(config.raw_content, config.file_path, findings)

        for resource in config.resources:
            props = resource.properties

            # Pod Security
            spec = props.get("spec", {})
            if resource.resource_type in [
                "Deployment",
                "Pod",
                "DaemonSet",
                "StatefulSet",
            ]:
                pod_spec = spec
                if "template" in spec:
                    pod_spec = spec.get("template", {}).get("spec", {})

                # Privileged containers
                containers = pod_spec.get("containers", [])
                for container in containers:
                    security_context = container.get("securityContext", {})
                    if security_context.get("privileged"):
                        findings.append(
                            IaCFinding(
                                rule_id="K8S-001",
                                severity=Severity.CRITICAL,
                                category=Category.PERMISSIONS,
                                message=f"Privileged container: {container.get('name')}",
                                description="Privileged containers have full host access",
                                file_path=config.file_path,
                                resource_type=resource.resource_type,
                                resource_name=resource.name,
                                remediation="Set privileged: false in securityContext",
                                cwe_id="CWE-250",
                            )
                        )

                    # Running as root
                    if security_context.get("runAsUser") == 0:
                        findings.append(
                            IaCFinding(
                                rule_id="K8S-002",
                                severity=Severity.MEDIUM,
                                category=Category.PERMISSIONS,
                                message=f"Container runs as root: {container.get('name')}",
                                description="Running as root increases security risk",
                                file_path=config.file_path,
                                resource_type=resource.resource_type,
                                resource_name=resource.name,
                                remediation="Set runAsNonRoot: true in securityContext",
                                cwe_id="CWE-250",
                            )
                        )

                    # No resource limits
                    if "resources" not in container or "limits" not in container.get(
                        "resources", {}
                    ):
                        findings.append(
                            IaCFinding(
                                rule_id="K8S-003",
                                severity=Severity.MEDIUM,
                                category=Category.BEST_PRACTICE,
                                message=f"No resource limits: {container.get('name')}",
                                description="Containers without limits can consume excessive resources",
                                file_path=config.file_path,
                                resource_type=resource.resource_type,
                                resource_name=resource.name,
                                remediation="Set resources.limits for CPU and memory",
                            )
                        )

                    # Image without tag
                    image = container.get("image", "")
                    if ":" not in image or image.endswith(":latest"):
                        findings.append(
                            IaCFinding(
                                rule_id="K8S-004",
                                severity=Severity.MEDIUM,
                                category=Category.BEST_PRACTICE,
                                message=f"Image without specific tag: {image}",
                                description="Using latest or untagged images is unpredictable",
                                file_path=config.file_path,
                                resource_type=resource.resource_type,
                                resource_name=resource.name,
                                remediation="Pin images to specific versions",
                            )
                        )

                # Host network
                if pod_spec.get("hostNetwork"):
                    findings.append(
                        IaCFinding(
                            rule_id="K8S-005",
                            severity=Severity.HIGH,
                            category=Category.NETWORK,
                            message="Pod uses host network",
                            description="Host network bypasses network policies",
                            file_path=config.file_path,
                            resource_type=resource.resource_type,
                            resource_name=resource.name,
                            remediation="Remove hostNetwork: true unless required",
                            cwe_id="CWE-284",
                        )
                    )

            # Service
            if resource.resource_type == "Service":
                if spec.get("type") == "LoadBalancer":
                    annotations = props.get("metadata", {}).get("annotations", {})
                    # Check for internal LB annotation
                    internal_annotations = [
                        "service.beta.kubernetes.io/aws-load-balancer-internal",
                        "networking.gke.io/load-balancer-type",
                        "service.beta.kubernetes.io/azure-load-balancer-internal",
                    ]
                    if not any(a in annotations for a in internal_annotations):
                        findings.append(
                            IaCFinding(
                                rule_id="K8S-006",
                                severity=Severity.MEDIUM,
                                category=Category.NETWORK,
                                message="LoadBalancer service without internal annotation",
                                description="Consider if this service should be internal-only",
                                file_path=config.file_path,
                                resource_type=resource.resource_type,
                                resource_name=resource.name,
                                remediation="Add internal load balancer annotation if not public",
                            )
                        )

            # Secret with plain text data
            if resource.resource_type == "Secret":
                if "stringData" in props:
                    findings.append(
                        IaCFinding(
                            rule_id="K8S-007",
                            severity=Severity.HIGH,
                            category=Category.SECRETS,
                            message="Secret contains unencrypted stringData",
                            description="Secrets in manifests should be externalized",
                            file_path=config.file_path,
                            resource_type=resource.resource_type,
                            resource_name=resource.name,
                            remediation="Use external secret management (Sealed Secrets, External Secrets)",
                            cwe_id="CWE-312",
                        )
                    )

        return findings


# =============================================================================
# Docker Compose Scanner
# =============================================================================


class DockerComposeScanner(IaCScannerProtocol):
    """Scanner for Docker Compose files."""

    @property
    def platform(self) -> IaCPlatform:
        return IaCPlatform.DOCKER_COMPOSE

    async def detect_files(self, root_path: Path) -> list[Path]:
        """Find Docker Compose files."""
        patterns = [
            "docker-compose.yml",
            "docker-compose.yaml",
            "docker-compose.*.yml",
            "docker-compose.*.yaml",
            "compose.yml",
            "compose.yaml",
        ]
        files = []
        for pattern in patterns:
            files.extend(root_path.glob(pattern))
            files.extend(root_path.glob(f"**/{pattern}"))
        return files

    async def parse_config(self, config_path: Path) -> IaCConfig:
        """Parse a Docker Compose file."""
        content = config_path.read_text(encoding="utf-8")
        resources = []

        try:
            data = yaml.safe_load(content)
            if not data:
                return IaCConfig(
                    platform=self.platform,
                    file_path=config_path,
                    raw_content=content,
                )

            services = data.get("services", {})
            for service_name, service_config in services.items():
                if not isinstance(service_config, dict):
                    continue

                resources.append(
                    IaCResource(
                        resource_type="docker_compose_service",
                        name=service_name,
                        file_path=config_path,
                        line_number=1,
                        properties=service_config,
                    )
                )

        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse Docker Compose {config_path}: {e}")

        return IaCConfig(
            platform=self.platform,
            file_path=config_path,
            resources=resources,
            raw_content=content,
        )

    async def scan(self, config: IaCConfig, policy: ScanPolicy | None = None) -> list[IaCFinding]:
        """Scan Docker Compose for security issues."""
        findings: list[IaCFinding] = []

        # Scan for secrets in raw content
        scan_for_secrets(config.raw_content, config.file_path, findings)

        for resource in config.resources:
            props = resource.properties

            # Privileged mode
            if props.get("privileged"):
                findings.append(
                    IaCFinding(
                        rule_id="DC-001",
                        severity=Severity.CRITICAL,
                        category=Category.PERMISSIONS,
                        message=f"Service {resource.name} runs privileged",
                        description="Privileged containers have full host access",
                        file_path=config.file_path,
                        resource_type=resource.resource_type,
                        resource_name=resource.name,
                        remediation="Remove privileged: true",
                        cwe_id="CWE-250",
                    )
                )

            # Host network mode
            if props.get("network_mode") == "host":
                findings.append(
                    IaCFinding(
                        rule_id="DC-002",
                        severity=Severity.HIGH,
                        category=Category.NETWORK,
                        message=f"Service {resource.name} uses host network",
                        description="Host networking bypasses container isolation",
                        file_path=config.file_path,
                        resource_type=resource.resource_type,
                        resource_name=resource.name,
                        remediation="Remove network_mode: host",
                        cwe_id="CWE-284",
                    )
                )

            # Environment secrets
            environment = props.get("environment", {})
            if isinstance(environment, dict):
                env_items = environment.items()
            elif isinstance(environment, list):
                env_items = []
                for item in environment:
                    if "=" in item:
                        k, v = item.split("=", 1)
                        env_items.append((k, v))
            else:
                env_items = []

            for key, value in env_items:
                key_upper = key.upper()
                if any(kw in key_upper for kw in ["PASSWORD", "SECRET", "KEY", "TOKEN"]):
                    if value and not value.startswith("${"):
                        findings.append(
                            IaCFinding(
                                rule_id="DC-003",
                                severity=Severity.HIGH,
                                category=Category.SECRETS,
                                message=f"Hardcoded secret in environment: {key}",
                                description="Secrets should use environment substitution",
                                file_path=config.file_path,
                                resource_type=resource.resource_type,
                                resource_name=resource.name,
                                remediation="Use ${VAR} syntax with .env file",
                                cwe_id="CWE-798",
                            )
                        )

            # Port 22 exposed
            ports = props.get("ports", [])
            for port in ports:
                port_str = str(port)
                if ":22" in port_str or port_str.startswith("22:"):
                    findings.append(
                        IaCFinding(
                            rule_id="DC-004",
                            severity=Severity.HIGH,
                            category=Category.NETWORK,
                            message=f"SSH port exposed on service {resource.name}",
                            description="SSH should not be exposed from containers",
                            file_path=config.file_path,
                            resource_type=resource.resource_type,
                            resource_name=resource.name,
                            remediation="Remove SSH port mapping, use docker exec instead",
                            cwe_id="CWE-284",
                        )
                    )

        return findings


# =============================================================================
# Scanner Registry
# =============================================================================


# Global registry of scanners
IAC_SCANNERS: dict[IaCPlatform, IaCScannerProtocol] = {
    IaCPlatform.TERRAFORM: TerraformScanner(),
    IaCPlatform.DOCKER: DockerScanner(),
    IaCPlatform.KUBERNETES: KubernetesScanner(),
    IaCPlatform.DOCKER_COMPOSE: DockerComposeScanner(),
}


def get_scanner(platform: IaCPlatform) -> IaCScannerProtocol | None:
    """Get a scanner for a specific IaC platform.

    Args:
        platform: The IaC platform

    Returns:
        The scanner or None if not supported
    """
    return IAC_SCANNERS.get(platform)


def get_all_scanners() -> list[IaCScannerProtocol]:
    """Get all registered IaC scanners."""
    return list(IAC_SCANNERS.values())
