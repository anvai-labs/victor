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

"""Tests for victor.security.safety.infrastructure module."""

import pytest

from victor.core.verticals.import_resolver import import_module_with_fallback
from victor.security.safety.infrastructure import (
    InfraPatternCategory,
    InfrastructureScanner,
    InfraScanResult,
    DESTRUCTIVE_PATTERNS,
    KUBERNETES_PATTERNS,
    DOCKER_PATTERNS,
    TERRAFORM_PATTERNS,
    CLOUD_PATTERNS,
    scan_infrastructure_command,
    validate_dockerfile,
    validate_kubernetes_manifest,
    get_all_infrastructure_patterns,
    get_safety_reminders,
)


def _load_vertical_attr(module_path: str, attr_name: str):
    """Load a vertical attribute or skip when unavailable."""
    module, _resolved = import_module_with_fallback(module_path)
    if module is None or not hasattr(module, attr_name):
        pytest.skip(f"Vertical module or attribute unavailable: {module_path}:{attr_name}")
    return getattr(module, attr_name)


class TestPatternLists:
    """Tests for pattern list definitions."""

    def test_destructive_patterns_not_empty(self):
        """DESTRUCTIVE_PATTERNS should have patterns."""
        assert len(DESTRUCTIVE_PATTERNS) > 0

    def test_kubernetes_patterns_not_empty(self):
        """KUBERNETES_PATTERNS should have patterns."""
        assert len(KUBERNETES_PATTERNS) > 0

    def test_docker_patterns_not_empty(self):
        """DOCKER_PATTERNS should have patterns."""
        assert len(DOCKER_PATTERNS) > 0

    def test_terraform_patterns_not_empty(self):
        """TERRAFORM_PATTERNS should have patterns."""
        assert len(TERRAFORM_PATTERNS) > 0

    def test_cloud_patterns_not_empty(self):
        """CLOUD_PATTERNS should have patterns."""
        assert len(CLOUD_PATTERNS) > 0

    def test_all_patterns_have_required_fields(self):
        """All patterns should have required fields."""
        all_patterns = get_all_infrastructure_patterns()
        for pattern in all_patterns:
            assert pattern.pattern, "Pattern should not be empty"
            assert pattern.description, "Description should not be empty"
            assert pattern.risk_level in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
            assert pattern.category, "Category should not be empty"


class TestInfraScanResult:
    """Tests for InfraScanResult dataclass."""

    def test_empty_result(self):
        """Empty InfraScanResult should have correct defaults."""
        result = InfraScanResult()
        assert result.matches == []
        assert result.risk_summary == {}
        assert result.has_critical is False
        assert result.has_high is False
        assert result.security_issues == []

    def test_add_match_updates_risk_summary(self):
        """Adding match should update risk_summary."""
        result = InfraScanResult()
        result.add_match(DESTRUCTIVE_PATTERNS[0])
        assert len(result.matches) == 1
        assert sum(result.risk_summary.values()) == 1

    def test_add_security_issue(self):
        """Adding security pattern should update security_issues."""
        # Find a security pattern
        security_pattern = None
        for pattern in KUBERNETES_PATTERNS:
            if pattern.category == "security":
                security_pattern = pattern
                break

        if security_pattern:
            result = InfraScanResult()
            result.add_match(security_pattern)
            assert len(result.security_issues) == 1


class TestInfrastructureScanner:
    """Tests for InfrastructureScanner class."""

    def test_scan_kubectl_delete_namespace(self):
        """Scanner should detect kubectl delete namespace."""
        scanner = InfrastructureScanner()
        result = scanner.scan_command("kubectl delete namespace production")
        assert result.has_critical is True

    def test_scan_kubectl_delete_deployment(self):
        """Scanner should detect kubectl delete deployment."""
        scanner = InfrastructureScanner()
        result = scanner.scan_command("kubectl delete deployment myapp")
        assert result.has_high is True

    def test_scan_terraform_destroy(self):
        """Scanner should detect terraform destroy."""
        scanner = InfrastructureScanner()
        result = scanner.scan_command("terraform destroy")
        assert result.has_critical is True

    def test_scan_terraform_destroy_auto_approve(self):
        """Scanner should detect terraform destroy --auto-approve."""
        scanner = InfrastructureScanner()
        result = scanner.scan_command("terraform destroy --auto-approve")
        assert result.has_critical is True

    def test_scan_docker_system_prune(self):
        """Scanner should detect docker system prune -a."""
        scanner = InfrastructureScanner()
        result = scanner.scan_command("docker system prune -a")
        assert result.has_high is True

    def test_scan_docker_volume_prune(self):
        """Scanner should detect docker volume prune."""
        scanner = InfrastructureScanner()
        result = scanner.scan_command("docker volume prune")
        assert result.has_high is True

    def test_scan_aws_terminate_instances(self):
        """Scanner should detect aws ec2 terminate-instances."""
        scanner = InfrastructureScanner()
        result = scanner.scan_command("aws ec2 terminate-instances --instance-ids i-123")
        assert result.has_high is True

    def test_scan_aws_delete_bucket(self):
        """Scanner should detect aws s3 rb --force."""
        scanner = InfrastructureScanner()
        result = scanner.scan_command("aws s3 rb --force s3://my-bucket")
        assert result.has_critical is True

    def test_scan_rm_rf_root(self):
        """Scanner should detect rm -rf /."""
        scanner = InfrastructureScanner()
        result = scanner.scan_command("rm -rf /home")
        assert result.has_critical is True

    def test_scan_drop_database(self):
        """Scanner should detect DROP DATABASE."""
        scanner = InfrastructureScanner()
        result = scanner.scan_command("DROP DATABASE production;")
        assert result.has_critical is True

    def test_scan_safe_command(self):
        """Scanner should return empty for safe commands."""
        scanner = InfrastructureScanner()
        result = scanner.scan_command("kubectl get pods")
        assert len(result.matches) == 0

    def test_scan_content_security_patterns(self):
        """Scanner should detect security patterns in content."""
        scanner = InfrastructureScanner()
        yaml_content = """
        securityContext:
          privileged: true
          runAsUser: 0
        """
        result = scanner.scan_content(yaml_content)
        assert len(result.security_issues) >= 2

    def test_get_patterns_by_category(self):
        """get_patterns_by_category should filter correctly."""
        scanner = InfrastructureScanner()
        k8s_patterns = scanner.get_patterns_by_category(InfraPatternCategory.KUBERNETES)
        assert len(k8s_patterns) > 0
        for pattern in k8s_patterns:
            assert pattern.category in ("kubernetes", "security")

    def test_get_patterns_by_risk(self):
        """get_patterns_by_risk should filter correctly."""
        scanner = InfrastructureScanner()
        critical_patterns = scanner.get_patterns_by_risk("CRITICAL")
        assert len(critical_patterns) > 0
        for pattern in critical_patterns:
            assert pattern.risk_level == "CRITICAL"

    def test_scanner_exclude_categories(self):
        """Scanner should exclude categories when configured."""
        scanner = InfrastructureScanner(
            include_kubernetes=False,
            include_docker=True,
            include_terraform=True,
        )
        result = scanner.scan_command("kubectl delete namespace production")
        # Should not detect k8s patterns
        assert len([m for m in result.matches if m.category == "kubernetes"]) == 0


class TestDockerfileValidation:
    """Tests for Dockerfile validation."""

    def test_validate_dockerfile_latest_tag(self):
        """Should warn about :latest tag."""
        dockerfile = "FROM python:latest"
        warnings = validate_dockerfile(dockerfile)
        assert any("latest" in w.lower() for w in warnings)

    def test_validate_dockerfile_no_user(self):
        """Should warn about no non-root user."""
        dockerfile = """
        FROM python:3.9
        RUN pip install app
        CMD ["python", "app.py"]
        """
        warnings = validate_dockerfile(dockerfile)
        assert any("user" in w.lower() for w in warnings)

    def test_validate_dockerfile_no_healthcheck(self):
        """Should warn about no HEALTHCHECK."""
        dockerfile = """
        FROM python:3.9
        USER app
        CMD ["python", "app.py"]
        """
        warnings = validate_dockerfile(dockerfile)
        assert any("healthcheck" in w.lower() for w in warnings)

    def test_validate_dockerfile_use_copy(self):
        """Should warn about ADD instead of COPY."""
        dockerfile = """
        FROM python:3.9
        ADD app.py /app/
        """
        warnings = validate_dockerfile(dockerfile)
        assert any("copy" in w.lower() and "add" in w.lower() for w in warnings)

    def test_validate_dockerfile_env_file(self):
        """Should warn about copying .env files."""
        dockerfile = """
        FROM python:3.9
        COPY .env /app/
        """
        warnings = validate_dockerfile(dockerfile)
        assert any(".env" in w for w in warnings)

    def test_validate_dockerfile_good(self):
        """Good Dockerfile should have minimal warnings."""
        dockerfile = """
        FROM python:3.9-slim
        RUN apt-get update && apt-get install -y curl && apt-get clean && rm -rf /var/lib/apt/lists/*
        USER app
        COPY --chown=app:app app.py /app/
        HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1
        CMD ["python", "app.py"]
        """
        warnings = validate_dockerfile(dockerfile)
        # Should have few warnings
        assert len(warnings) <= 2


class TestKubernetesValidation:
    """Tests for Kubernetes manifest validation."""

    def test_validate_k8s_privileged(self):
        """Should warn about privileged containers."""
        manifest = """
        spec:
          containers:
            - securityContext:
                privileged: true
        """
        warnings = validate_kubernetes_manifest(manifest)
        assert any("privileged" in w.lower() for w in warnings)

    def test_validate_k8s_no_limits(self):
        """Should warn about no resource limits."""
        manifest = """
        spec:
          containers:
            - name: app
              image: myapp:1.0
        """
        warnings = validate_kubernetes_manifest(manifest)
        assert any("limits" in w.lower() for w in warnings)

    def test_validate_k8s_no_probes(self):
        """Should warn about no probes."""
        manifest = """
        spec:
          containers:
            - name: app
              image: myapp:1.0
        """
        warnings = validate_kubernetes_manifest(manifest)
        assert any("liveness" in w.lower() for w in warnings)
        assert any("readiness" in w.lower() for w in warnings)

    def test_validate_k8s_latest_tag(self):
        """Should warn about :latest image tag."""
        manifest = """
        spec:
          containers:
            - name: app
              image: myapp:latest
        """
        warnings = validate_kubernetes_manifest(manifest)
        assert any("latest" in w.lower() for w in warnings)

    def test_validate_k8s_host_network(self):
        """Should warn about host network."""
        manifest = """
        spec:
          hostNetwork: true
        """
        warnings = validate_kubernetes_manifest(manifest)
        assert any("host network" in w.lower() for w in warnings)

    def test_validate_k8s_run_as_root(self):
        """Should warn about running as root."""
        manifest = """
        spec:
          securityContext:
            runAsUser: 0
        """
        warnings = validate_kubernetes_manifest(manifest)
        assert any("root" in w.lower() for w in warnings)

    def test_validate_k8s_good(self):
        """Good manifest should have minimal warnings."""
        manifest = """
        apiVersion: apps/v1
        kind: Deployment
        spec:
          containers:
            - name: app
              image: myapp:1.2.3
              securityContext:
                runAsNonRoot: true
                allowPrivilegeEscalation: false
              resources:
                limits:
                  cpu: 500m
                  memory: 512Mi
              livenessProbe:
                httpGet:
                  path: /health
                  port: 8080
              readinessProbe:
                httpGet:
                  path: /ready
                  port: 8080
        """
        warnings = validate_kubernetes_manifest(manifest)
        # Should have few warnings
        assert len(warnings) <= 2


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_scan_infrastructure_command_function(self):
        """scan_infrastructure_command convenience function should work."""
        matches = scan_infrastructure_command("terraform destroy")
        assert len(matches) > 0

    def test_get_all_infrastructure_patterns_function(self):
        """get_all_infrastructure_patterns should return combined list."""
        all_patterns = get_all_infrastructure_patterns()
        assert len(all_patterns) > 0
        # Should include patterns from all categories
        categories = {p.category for p in all_patterns}
        assert "kubernetes" in categories or "destructive" in categories

    def test_get_safety_reminders_function(self):
        """get_safety_reminders should return list of reminders."""
        reminders = get_safety_reminders()
        assert len(reminders) > 0
        assert all(isinstance(r, str) for r in reminders)


class TestVerticalIntegration:
    """Tests for vertical safety extension integration."""

    def test_coding_safety_extension_uses_scanner(self):
        """CodingSafetyExtension should delegate to CodePatternScanner."""
        CodingSafetyExtension = _load_vertical_attr(
            "victor.coding.safety",
            "CodingSafetyExtension",
        )

        ext = CodingSafetyExtension()
        patterns = ext.get_bash_patterns()
        assert len(patterns) > 0
        # Should include git patterns
        assert any("git" in p.category for p in patterns)

    def test_devops_safety_extension_uses_scanner(self):
        """DevOpsSafetyExtension should delegate to InfrastructureScanner."""
        DevOpsSafetyExtension = _load_vertical_attr(
            "victor.devops.safety",
            "DevOpsSafetyExtension",
        )

        ext = DevOpsSafetyExtension()
        patterns = ext.get_bash_patterns()
        assert len(patterns) > 0
        # Should include k8s/docker patterns
        assert any(p.category in ("kubernetes", "docker", "terraform") for p in patterns)

    def test_coding_extension_scan_command(self):
        """CodingSafetyExtension.scan_command should work."""
        CodingSafetyExtension = _load_vertical_attr(
            "victor.coding.safety",
            "CodingSafetyExtension",
        )

        ext = CodingSafetyExtension()
        matches = ext.scan_command("git push --force origin main")
        assert len(matches) > 0

    def test_devops_extension_scan_command(self):
        """DevOpsSafetyExtension.scan_command should work."""
        DevOpsSafetyExtension = _load_vertical_attr(
            "victor.devops.safety",
            "DevOpsSafetyExtension",
        )

        ext = DevOpsSafetyExtension()
        result = ext.scan_command("kubectl delete namespace prod")
        assert result.has_critical is True

    def test_devops_extension_validate_dockerfile(self):
        """DevOpsSafetyExtension.validate_dockerfile should work."""
        DevOpsSafetyExtension = _load_vertical_attr(
            "victor.devops.safety",
            "DevOpsSafetyExtension",
        )

        ext = DevOpsSafetyExtension()
        warnings = ext.validate_dockerfile("FROM python:latest")
        assert any("latest" in w.lower() for w in warnings)

    def test_devops_extension_validate_k8s(self):
        """DevOpsSafetyExtension.validate_kubernetes_manifest should work."""
        DevOpsSafetyExtension = _load_vertical_attr(
            "victor.devops.safety",
            "DevOpsSafetyExtension",
        )

        ext = DevOpsSafetyExtension()
        warnings = ext.validate_kubernetes_manifest("privileged: true")
        assert any("privileged" in w.lower() for w in warnings)
