# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""First-run onboarding for Victor.

Sequences the existing setup primitives into one guided journey:

1. Welcome and confirmation
2. Provider + credential setup via :class:`AuthSetupWizard` (the ``victor
   auth setup`` core) — provider picker, key entry stored in the system
   keyring, and a real connection smoke test
3. Default-profile installation so bare ``victor`` starts with the chosen
   provider/model (the recommended experience profile is applied without
   extra questions)
4. Completion panel with copy-paste example prompts and next steps

Failures point at ``victor doctor`` inline instead of leaving the user
stranded. The flow is idempotent: success writes
``~/.victor/.onboarding_completed`` which gates future first-run triggers,
and ``victor onboarding --force`` re-runs it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from victor.config.profiles import (
    ProfileManager,
    get_recommended_profile,
    install_profile,
)
from victor.config.settings import get_project_paths

#: Copy-paste prompts offered on the completion screen.
EXAMPLE_PROMPTS = [
    "Explain what this repository does and how it is structured",
    "Write a Python script that renames every .txt file in a folder to .md",
    "Review this function for bugs: <paste code>",
]


class OnboardingWizard:
    """First-run wizard: auth setup → default profile → next steps."""

    def __init__(self, console: Optional[Console] = None, offer_chat: bool = True):
        """Initialize the wizard.

        Args:
            console: Optional Rich console instance (creates default if None)
            offer_chat: Whether the completion screen offers to start a chat.
                The bare ``victor`` first-run path passes False because it
                always drops into interactive chat right after onboarding.
        """
        self.console = console or Console()
        self.offer_chat = offer_chat
        self.config_dir = get_project_paths().global_victor_dir

    def run(self) -> int:
        """Run the complete onboarding flow.

        Returns:
            Exit code (0 for success/cancellation, 1 for error)
        """
        try:
            self._show_welcome()

            if not self._confirm_start():
                self.console.print("\n[yellow]Onboarding cancelled.[/]")
                return 0

            # Provider + credentials + connection smoke test. The auth wizard
            # is the single owner of provider selection, key persistence
            # (keyring + accounts config), and validation — onboarding only
            # frames it and finishes the job.
            from victor.ui.commands.auth import AuthSetupWizard

            auth_wizard = AuthSetupWizard(self.console, first_run=True)
            exit_code = auth_wizard.run()
            if exit_code != 0:
                self._show_doctor_hint()
                return exit_code

            provider = auth_wizard.state.get("selected_provider")
            model = auth_wizard.state.get("selected_model")
            account = auth_wizard.state.get("saved_account")
            if not provider or account is None:
                # User backed out inside the auth wizard; nothing was saved.
                self.console.print("\n[yellow]Onboarding cancelled.[/]")
                return 0

            self._install_default_profile(provider, model, account)
            self._write_completion_marker(provider, model)
            self._show_completion(provider, model)

            return 0

        except KeyboardInterrupt:
            self.console.print("\n\n[yellow]Onboarding cancelled.[/]")
            return 0
        except Exception as e:
            self.console.print(f"\n[red]✗[/] An error occurred: {e}")
            self._show_doctor_hint()
            return 1

    def _show_welcome(self) -> None:
        """Display welcome screen."""
        welcome_text = Text()
        welcome_text.append("Welcome to ", style="white")
        welcome_text.append("Victor", style="bold cyan")
        welcome_text.append("! ", style="white")
        welcome_text.append("Open-source Agentic AI Framework\n\n", style="dim")

        features = Table(show_header=False, box=None, padding=(0, 2))
        features.add_column("", style="cyan")
        features.add_column("", style="white")

        features.add_row("✦", "24 LLM provider adapters")
        features.add_row("✦", "34 tool modules")
        features.add_row("✦", "Multi-agent coordination")
        features.add_row("✦", "Domain-specific verticals")
        features.add_row("✦", "YAML workflow engine")

        # Group, not Text + Table: rich renderables don't support "+", and
        # the resulting TypeError used to kill the whole wizard at step 0.
        panel = Panel.fit(
            Group(welcome_text, features),
            title="[bold cyan]Victor Setup Wizard[/]",
            border_style="cyan",
            padding=(1, 2),
        )

        self.console.print(panel)
        self.console.print()

    def _confirm_start(self) -> bool:
        """Ask user to confirm starting onboarding.

        Returns:
            True if user wants to continue, False otherwise
        """
        # Check if config already exists
        profiles_path = self.config_dir / "profiles.yaml"
        if profiles_path.exists():
            self.console.print("[yellow]⚠[/] Configuration already exists!")
            self.console.print(f"[dim]Found: {profiles_path}[/]")
            self.console.print()

            if not Confirm.ask("Would you like to reconfigure Victor?", default=False):
                return False

        return Confirm.ask("Ready to set up Victor?", default=True)

    def _install_default_profile(self, provider: str, model: Optional[str], account: Any) -> None:
        """Point bare ``victor`` at the configured provider.

        The auth wizard persists the account (keyring + config.yaml) and an
        account-named chat profile, but startup reads the ``default`` profile
        from profiles.yaml — without this step a first-run cloud user would
        still boot on the built-in Ollama default.
        """
        recommended = get_recommended_profile()
        # Local providers report a placeholder model; let the profile
        # template pick its own default in that case.
        model_override = None if model in (None, "", "default") else model

        profiles_path = install_profile(
            recommended,
            config_dir=self.config_dir,
            provider_override=provider,
            model_override=model_override,
        )
        # install_profile rewrites profiles.yaml wholesale; restore the
        # account-named chat profile the auth wizard just synced.
        try:
            ProfileManager.for_config_dir(self.config_dir).upsert_account_profile(account)
        except Exception:
            pass  # the default profile alone is enough to start chatting

        self.console.print()
        self.console.print(f"[green]✓[/] Applied '{recommended.name}' profile — saved to:")
        self.console.print(f"  [dim]{profiles_path}[/]")
        self.console.print("  [dim]Change later with: victor config profiles list[/]")

    def _write_completion_marker(self, provider: str, model: Optional[str]) -> None:
        """Record completion so first-run detection never re-triggers."""
        marker_file = self.config_dir / ".onboarding_completed"
        completed_at = datetime.now().isoformat()
        try:
            marker_file.write_text(
                f"# Onboarding completed successfully\n"
                f"# Completed at: {completed_at}\n"
                f"# Provider: {provider}\n"
                f"# Model: {model}\n"
            )
        except Exception:
            pass  # marker is an optimization; setup itself succeeded

    def _show_doctor_hint(self) -> None:
        """Point at ``victor doctor``, with a quick inline snapshot."""
        self.console.print()
        try:
            from victor.ui.commands.doctor import DoctorChecks, Severity

            checks = DoctorChecks()
            checks.check_api_keys()
            checks.check_local_providers()
            for check in checks.checks:
                style = "green" if check.severity == Severity.SUCCESS else "yellow"
                self.console.print(f"  [{style}]{check.name}:[/] {check.message}")
        except Exception:
            pass
        self.console.print("[dim]Run [cyan]victor doctor[/] for a full diagnosis.[/]")

    def _show_completion(self, provider: str, model: Optional[str]) -> None:
        """Show completion screen and next steps."""
        self.console.print("\n[bold cyan]✓ You're ready![/]")
        self.console.print("═" * 50)

        # Summary table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("", style="yellow")
        table.add_column("", style="white")

        table.add_row("Provider", provider)
        table.add_row("Model", str(model or "auto"))
        table.add_row("Config", str(self.config_dir / "profiles.yaml"))

        self.console.print("\n[bold]Your Configuration:[/]")
        self.console.print(table)

        # Example prompts to get going immediately
        self.console.print("\n[bold]Try asking:[/]")
        for prompt in EXAMPLE_PROMPTS:
            self.console.print(f'  [cyan]"{prompt}"[/]')

        # Next steps
        self.console.print("\n[bold]Next Steps:[/]")
        self.console.print("  [cyan]victor[/] - Start chatting")
        self.console.print("  [cyan]victor examples[/] - Browse runnable examples")
        self.console.print("  [cyan]victor doctor[/] - Run diagnostics")

        # Offer to start chat (skipped when the caller starts chat itself)
        if self.offer_chat:
            self.console.print()
            if Confirm.ask("Start your first chat now?", default=True):
                self._start_first_chat()

    def _start_first_chat(self) -> None:
        """Start the first chat session."""
        self.console.print("\n[yellow]Starting Victor chat...[/]\n")
        self.console.print("[dim]Type your message and press Enter. Type 'quit' to exit.[/]\n")

        try:
            from victor.ui.commands.chat import _run_default_interactive

            _run_default_interactive()
        except Exception as e:
            self.console.print(f"\n[yellow]Chat ended: {e}[/]")


def run_onboarding(offer_chat: bool = True) -> int:
    """Run the onboarding wizard.

    Entry point for bare ``victor`` first-run detection, ``victor
    onboarding``, and ``victor init --wizard``.

    Args:
        offer_chat: Whether the completion screen offers to start a chat.
            Pass False when the caller drops into chat itself.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    console = Console()
    try:
        wizard = OnboardingWizard(console, offer_chat=offer_chat)
        return wizard.run()
    except Exception as e:
        console.print(f"\n[red]✗[/] Onboarding failed: {e}")
        return 1
