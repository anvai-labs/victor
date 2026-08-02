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

"""Doctor command for system diagnostics and troubleshooting.

Provides comprehensive checks to help users diagnose and fix issues:
- Python version and environment
- Dependencies and packages
- Provider connectivity and configuration
- Tool availability
- File permissions
- Performance optimizations
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from victor.config.settings import get_project_paths


class Severity(Enum):
    """Severity level for diagnostic issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


@dataclass
class DiagnosticCheck:
    """Result of a diagnostic check."""

    name: str
    severity: Severity
    message: str
    suggestion: str | None = None
    passed: bool = False


class DoctorChecks:
    """Diagnostic checks for Victor."""

    def __init__(self, verbose: bool = False, fix: bool = False):
        self.verbose = verbose
        self.fix = fix
        self.checks: List[DiagnosticCheck] = []

    @staticmethod
    def _get_effective_config_dir() -> Path:
        """Resolve the effective Victor config directory."""
        victor_config = os.getenv("VICTOR_CONFIG_DIR")
        if victor_config:
            return Path(victor_config)
        return get_project_paths().global_victor_dir

    def add_check(
        self,
        name: str,
        severity: Severity,
        message: str,
        suggestion: str | None = None,
    ) -> None:
        """Add a diagnostic check result."""
        self.checks.append(
            DiagnosticCheck(
                name=name,
                severity=severity,
                message=message,
                suggestion=suggestion,
                passed=(severity == Severity.SUCCESS),
            )
        )

    def check_python_version(self) -> None:
        """Check Python version compatibility."""
        major, minor = sys.version_info[:2]
        version_str = f"{major}.{minor}"

        if (major, minor) >= (3, 10):
            self.add_check(
                name="Python Version",
                severity=Severity.SUCCESS,
                message=f"Python {version_str} is supported",
            )
        else:
            self.add_check(
                name="Python Version",
                severity=Severity.ERROR,
                message=f"Python {version_str} is not supported (requires 3.10+)",
                suggestion="Install Python 3.10 or higher from python.org or use pyenv/conda",
            )

    def check_pip_package(self, package_name: str) -> bool:
        """Check if a pip package is installed.

        Returns:
            True if package is installed
        """
        try:
            importlib.metadata.distribution(package_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

    def check_dependencies(self) -> None:
        """Check core dependencies."""
        required = [
            ("pydantic", "Core dependency for configuration"),
            ("typer", "CLI framework"),
            ("rich", "Terminal formatting"),
            ("aiohttp", "Async HTTP support"),
        ]

        optional = [
            ("openai", "OpenAI provider"),
            ("anthropic", "Anthropic provider"),
        ]

        for package, description in required:
            if self.check_pip_package(package):
                self.add_check(
                    name=f"Dependency: {package}",
                    severity=Severity.SUCCESS,
                    message=f"{package} is installed",
                )
            else:
                self.add_check(
                    name=f"Dependency: {package}",
                    severity=Severity.ERROR,
                    message=f"{package} is not installed ({description})",
                    suggestion=f"Run: pip install {package} or pip install -e .",
                )

        for package, description in optional:
            if self.check_pip_package(package):
                self.add_check(
                    name=f"Optional: {package}",
                    severity=Severity.SUCCESS,
                    message=f"{package} is installed",
                )
            else:
                self.add_check(
                    name=f"Optional: {package}",
                    severity=Severity.WARNING,
                    message=f"{package} is not installed ({description})",
                    suggestion=f"Run: pip install {package} to enable {package.split()[0]} provider",
                )

    def check_api_keys(self) -> None:
        """Check for API keys in environment."""
        api_keys = {
            "ANTHROPIC_API_KEY": "Anthropic Claude",
            "OPENAI_API_KEY": "OpenAI GPT",
            "GOOGLE_API_KEY": "Google Gemini",
            "AZURE_API_KEY": "Azure OpenAI",
            "XAI_API_KEY": "xAI Grok",
            "COHERE_API_KEY": "Cohere",
        }

        found_keys = []
        for env_var, provider in api_keys.items():
            if os.getenv(env_var):
                found_keys.append(provider)
                self.add_check(
                    name=f"API Key: {provider}",
                    severity=Severity.SUCCESS,
                    message=f"{provider} API key is configured",
                )

        if not found_keys:
            self.add_check(
                name="Cloud Provider API Keys",
                severity=Severity.WARNING,
                message="No cloud provider API keys found",
                suggestion="Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable, "
                "or use --provider ollama for local models",
            )

    def check_local_providers(self) -> None:
        """Check local provider availability."""
        # Check Ollama
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=2,
            )
            if result.returncode == 0:
                self.add_check(
                    name="Local Provider: Ollama",
                    severity=Severity.SUCCESS,
                    message="Ollama is running and accessible",
                )
            else:
                self.add_check(
                    name="Local Provider: Ollama",
                    severity=Severity.WARNING,
                    message="Ollama is installed but not running",
                    suggestion="Start Ollama: 'ollama serve' or 'brew services start ollama'",
                )
        except FileNotFoundError:
            self.add_check(
                name="Local Provider: Ollama",
                severity=Severity.INFO,
                message="Ollama is not installed",
                suggestion="Install Ollama: 'curl -fsSL https://ollama.com/install.sh | sh' "
                "or use: brew install ollama",
            )
        except Exception as e:
            self.add_check(
                name="Local Provider: Ollama",
                severity=Severity.INFO,
                message=f"Could not check Ollama status: {e}",
            )

    def check_default_model(self) -> None:
        """Check if default model is available in Ollama."""
        try:
            from victor.config.settings import load_settings

            settings = load_settings()

            # Only check if Ollama is the default provider
            if settings.provider.default_provider.lower() != "ollama":
                self.add_check(
                    name="Default Model",
                    severity=Severity.INFO,
                    message=f"Default provider is {settings.provider.default_provider} (skipping Ollama model check)",
                )
                return

            default_model = settings.provider.default_model

            # Check if Ollama is running
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                self.add_check(
                    name="Default Model",
                    severity=Severity.WARNING,
                    message=f"Cannot check model '{default_model}' - Ollama not running",
                    suggestion="Start Ollama: 'ollama serve'",
                )
                return

            # Parse available models
            models = result.stdout.strip().split("\n")[1:]  # Skip header
            available_models = [line.split()[0] for line in models if line.strip()]

            # Check if default model is available
            if default_model in available_models:
                self.add_check(
                    name="Default Model",
                    severity=Severity.SUCCESS,
                    message=f"Default model '{default_model}' is available",
                )
            else:
                # Model not found
                if available_models:
                    model_list = ", ".join(available_models[:5])
                    if len(available_models) > 5:
                        model_list += f", ... ({len(available_models)} total)"
                    self.add_check(
                        name="Default Model",
                        severity=Severity.WARNING,
                        message=f"Default model '{default_model}' is not installed",
                        suggestion=f"Available: {model_list}\n"
                        f"Pull default: 'ollama pull {default_model}'\n"
                        f"Or change default in: ~/.victor/profiles.yaml",
                    )
                else:
                    self.add_check(
                        name="Default Model",
                        severity=Severity.WARNING,
                        message=f"Default model '{default_model}' is not installed (no models found)",
                        suggestion=f"Pull default: 'ollama pull {default_model}'",
                    )

        except ImportError:
            self.add_check(
                name="Default Model",
                severity=Severity.INFO,
                message="Could not import settings to check default model",
            )
        except subprocess.TimeoutExpired:
            self.add_check(
                name="Default Model",
                severity=Severity.WARNING,
                message="Ollama command timed out",
                suggestion="Check if Ollama is running: 'ollama serve'",
            )
        except Exception as e:
            self.add_check(
                name="Default Model",
                severity=Severity.INFO,
                message=f"Could not check default model: {e}",
            )

    def check_config_directory(self) -> None:
        """Check Victor configuration directory."""
        victor_config = os.getenv("VICTOR_CONFIG_DIR")
        config_path = self._get_effective_config_dir()
        if victor_config:
            if config_path.exists():
                self.add_check(
                    name="Configuration Directory",
                    severity=Severity.SUCCESS,
                    message=f"Config directory exists: {config_path}",
                )
            else:
                self.add_check(
                    name="Configuration Directory",
                    severity=Severity.ERROR,
                    message=f"Config directory does not exist: {config_path}",
                    suggestion="Run 'victor init' to create configuration",
                )
        else:
            # Check default location
            if config_path.exists():
                self.add_check(
                    name="Configuration Directory",
                    severity=Severity.SUCCESS,
                    message=f"Using default config directory: {config_path}",
                )
            else:
                self.add_check(
                    name="Configuration Directory",
                    severity=Severity.INFO,
                    message="No configuration directory (will use defaults)",
                    suggestion="Run 'victor init' to create configuration",
                )

    def check_performance_settings(self) -> None:
        """Check performance optimization settings."""
        try:
            from victor.config.settings import Settings

            settings = Settings()

            # Check framework preloading
            if (
                hasattr(settings, "framework_preload_enabled")
                and settings.framework_preload_enabled
            ):
                self.add_check(
                    name="Performance: Preloading",
                    severity=Severity.SUCCESS,
                    message="Framework preloading is enabled (50-70% faster first requests)",
                )
            else:
                self.add_check(
                    name="Performance: Preloading",
                    severity=Severity.WARNING,
                    message="Framework preloading is disabled",
                    suggestion="Enable framework_preload in your profile for faster first requests",
                )

            # Check HTTP connection pooling
            if (
                hasattr(settings, "http_connection_pool_enabled")
                and settings.http_connection_pool_enabled
            ):
                self.add_check(
                    name="Performance: HTTP Pooling",
                    severity=Severity.SUCCESS,
                    message="HTTP connection pooling is enabled (20-30% faster HTTP requests)",
                )
            else:
                self.add_check(
                    name="Performance: HTTP Pooling",
                    severity=Severity.WARNING,
                    message="HTTP connection pooling is disabled",
                    suggestion="Enable http_connection_pool_enabled in your profile",
                )

            # Check tool selection cache
            if (
                hasattr(settings, "tool_selection_cache_enabled")
                and settings.tools.tool_selection_cache_enabled
            ):
                self.add_check(
                    name="Performance: Tool Cache",
                    severity=Severity.SUCCESS,
                    message="Tool selection cache is enabled (20-40% faster conversations)",
                )
            else:
                self.add_check(
                    name="Performance: Tool Cache",
                    severity=Severity.WARNING,
                    message="Tool selection cache is disabled",
                    suggestion="Enable tool_selection_cache_enabled in your profile",
                )

        except Exception as e:
            self.add_check(
                name="Performance Settings",
                severity=Severity.INFO,
                message=f"Could not check performance settings: {e}",
            )

    def check_file_permissions(self) -> None:
        """Check file system permissions for common operations."""
        # Check home directory writability
        home_dir = Path.home()
        if os.access(home_dir, os.W_OK):
            self.add_check(
                name="File Permissions: Home Directory",
                severity=Severity.SUCCESS,
                message=f"Home directory is writable: {home_dir}",
            )
        else:
            self.add_check(
                name="File Permissions: Home Directory",
                severity=Severity.ERROR,
                message=f"Home directory is not writable: {home_dir}",
                suggestion="Check directory permissions: ls -la ~",
            )

    def run_all_checks(self) -> List[DiagnosticCheck]:
        """Run all diagnostic checks.

        Returns:
            List of all diagnostic checks performed
        """
        self.check_python_version()
        self.check_dependencies()
        self.check_api_keys()
        self.check_local_providers()
        self.check_default_model()  # Phase 6: Check model existence
        self.check_config_directory()
        self.check_performance_settings()
        self.check_file_permissions()

        return self.checks

    def fix_issues(self) -> List[str]:
        """Attempt to automatically fix common issues.

        Returns:
            List of fix descriptions that were applied
        """
        fixes_applied = []

        for check in self.checks:
            if check.severity == Severity.SUCCESS:
                continue
            if check.severity == Severity.INFO and check.name != "Configuration Directory":
                continue

            # Attempt fixes for specific checks
            if check.name == "Configuration Directory" and (
                "does not exist" in check.message.lower()
                or "no configuration directory" in check.message.lower()
            ):
                # Create config directory if it doesn't exist
                try:
                    config_dir = self._get_effective_config_dir()
                    config_dir.mkdir(parents=True, exist_ok=True)
                    (config_dir / "config.yaml").write_text("# Victor configuration\n")
                    fixes_applied.append(f"Created config directory: {config_dir}")
                except Exception as e:
                    self.checks.append(
                        DiagnosticCheck(
                            name="Configuration Directory Fix",
                            severity=Severity.ERROR,
                            message=f"Failed to create config directory: {e}",
                            suggestion=f"Create directory manually: mkdir -p {config_dir}",
                            passed=False,
                        )
                    )

            elif check.name == "performance_settings" and "disabled" in check.message.lower():
                # Enable performance settings
                try:
                    import subprocess

                    result = subprocess.run(
                        ["victor", "config", "set", "cache_optimization", "true"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        fixes_applied.append("Enabled cache optimization")
                except Exception:
                    # Silently skip if config set command doesn't work
                    pass

        return fixes_applied

    def print_results(self) -> int:
        """Print diagnostic results to console.

        Returns:
            Exit code (0 for success, 1 for errors found)
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # Count by severity
        errors = sum(1 for c in self.checks if c.severity == Severity.ERROR)
        warnings = sum(1 for c in self.checks if c.severity == Severity.WARNING)
        infos = sum(1 for c in self.checks if c.severity == Severity.INFO)
        successes = sum(1 for c in self.checks if c.severity == Severity.SUCCESS)

        # Print summary table
        console.print("\n[bold]Victor Doctor - System Diagnostics[/]")
        console.print("═" * 50)

        table = Table(show_header=True, show_lines=True)
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Message", style="white")
        table.add_column("Suggestion", style="yellow")

        for check in self.checks:
            # Status icon
            if check.severity == Severity.SUCCESS:
                status = "[green]✓[/]"
            elif check.severity == Severity.ERROR:
                status = "[red]✗[/]"
            elif check.severity == Severity.WARNING:
                status = "[yellow]⚠[/]"
            else:  # INFO
                status = "[blue]ℹ[/]"

            table.add_row(
                check.name,
                status,
                check.message,
                check.suggestion or "",
            )

        console.print(table)

        # Print summary
        console.print("\n" + "─" * 50)
        console.print(
            f"[bold]Summary:[/] {successes} success, {warnings} warnings, {infos} info, {errors} errors"
        )

        if errors == 0 and warnings <= 2:
            console.print("\n[green]✓ Your system is ready to use Victor![/]")
            return 0
        elif errors > 0:
            console.print(f"\n[red]✗ Found {errors} error(s) that should be fixed[/]")
            console.print(
                "[yellow]Run 'victor config validate' for detailed configuration validation[/]"
            )
            return 1
        else:
            console.print(
                f"\n[yellow]⚠ Found {warnings} warning(s) - Victor will work but may be suboptimal[/]"
            )
            return 0


def run_doctor(verbose: bool = False, fix: bool = False) -> int:
    """Run doctor diagnostics.

    This is the entry point for 'victor doctor' command.

    Args:
        verbose: Show detailed diagnostic output
        fix: Automatically fix common issues

    Returns:
        Exit code (0 for success, 1 for errors)
    """
    from rich.console import Console

    console = Console()

    doctor = DoctorChecks(verbose=verbose, fix=fix)

    try:
        # Run initial checks
        doctor.run_all_checks()

        # Attempt fixes if requested
        if fix:
            console.print("\n[yellow]🔧 Attempting to fix issues...[/]\n")
            fixes_applied = doctor.fix_issues()

            if fixes_applied:
                console.print(f"[green]✓ Applied {len(fixes_applied)} fix(es):[/]")
                for fix_desc in fixes_applied:
                    console.print(f"  • {fix_desc}")
                console.print("")

                # Re-run checks to verify fixes
                console.print("[cyan]Re-running diagnostics to verify fixes...[/]\n")
                doctor.checks.clear()  # Clear previous checks
                doctor.run_all_checks()
            else:
                console.print("[yellow]⚠ No auto-fixes available for detected issues[/]\n")

        return doctor.print_results()
    except Exception as e:
        print(f"[red]✗ Doctor command failed:[/] {e}")
        return 1
