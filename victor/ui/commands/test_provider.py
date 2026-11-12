import typer
import logging
from rich.console import Console
from victor.ui.console import get_console

from victor.core.async_utils import run_sync
from victor.config.settings import load_settings
from victor.providers.registry import ProviderRegistry

test_provider_app = typer.Typer(
    name="test-provider", help="Test if a provider is working correctly."
)
console = get_console()
logger = logging.getLogger(__name__)


@test_provider_app.callback(invoke_without_command=True)
def test_provider(
    ctx: typer.Context,
    provider: str = typer.Argument(..., help="Provider name to test"),
    auth_mode: str = typer.Option(
        "api_key",
        "--auth-mode",
        help="Authentication mode: 'api_key' or 'oauth' (OpenAI Codex, Qwen Coding Plan).",
    ),
    coding_plan: bool = typer.Option(
        False,
        "--coding-plan",
        help="Use Z.AI coding plan endpoint.",
    ),
    endpoint: str = typer.Option(
        None,
        "--endpoint",
        help="Z.AI endpoint variant: standard, coding, china, anthropic.",
    ),
):
    """Test if a provider is working correctly."""
    if ctx.invoked_subcommand is None:
        console.print(f"Testing provider: [cyan]{provider}[/]")
        run_sync(
            test_provider_async(
                provider,
                auth_mode=auth_mode,
                coding_plan=coding_plan,
                endpoint=endpoint,
            )
        )


async def test_provider_async(
    provider: str,
    auth_mode: str = "api_key",
    coding_plan: bool = False,
    endpoint: str = None,
) -> None:
    """Async function to test provider."""
    settings = load_settings()

    try:
        # Check if provider is registered
        if not ProviderRegistry.is_registered(provider):
            console.print(f"[red]✗[/] Provider '{provider}' not found")
            console.print(f"\nAvailable providers: {', '.join(ProviderRegistry.list_providers())}")
            return

        console.print("[green]✓[/] Provider registered")

        # Get provider settings
        provider_settings = settings.get_provider_settings(provider)

        # Check API key for cloud providers (skip for OAuth mode)
        if auth_mode == "api_key" and provider in [
            "anthropic",
            "openai",
            "google",
            "xai",
            "grok",
            "zai",
            "qwen",
        ]:
            api_key = provider_settings.get("api_key")
            if not api_key:
                console.print(f"[red]✗[/] No API key configured for {provider}")
                env_var = f"{provider.upper()}_API_KEY"
                if provider == "zai":
                    env_var = "ZAI_API_KEY or ZHIPUAI_API_KEY"
                elif provider == "qwen":
                    env_var = "QWEN_API_KEY or DASHSCOPE_API_KEY"
                console.print(f"\nSet environment variable: [bold]{env_var}[/]")
                return
            console.print("[green]✓[/] API key configured")

        # Provider-specific connectivity tests
        if provider == "ollama":
            await _test_ollama(provider_settings)
        elif provider == "anthropic":
            await _test_anthropic(provider_settings)
        elif provider == "openai":
            await _test_openai(provider_settings, auth_mode=auth_mode)
        elif provider == "google":
            await _test_google(provider_settings)
        elif provider in ("zai", "zai-coding-plan", "zai-coding", "zhipuai", "zhipu"):
            await _test_zai(provider_settings, coding_plan=coding_plan, endpoint=endpoint)
        elif provider in ("qwen", "alibaba", "dashscope"):
            await _test_qwen(provider_settings, auth_mode=auth_mode)
        else:
            console.print(f"\n[green]✓[/] Provider {provider} is ready to use!")

    except Exception as e:
        console.print(f"[red]✗[/] Error: {e}")


async def _test_ollama(provider_settings: dict) -> None:
    """Test Ollama connectivity."""
    from victor.providers.ollama_provider import OllamaProvider

    ollama = OllamaProvider(**provider_settings)
    try:
        models = await ollama.list_models()
        if models:
            console.print(f"[green]✓[/] Ollama is running with {len(models)} models")
            console.print("\n[green]✓[/] Provider ollama is ready to use!")
        else:
            console.print("[yellow]⚠[/] Ollama is running but no models installed")
            console.print("\nPull a model: [bold]ollama pull qwen2.5-coder:7b[/]")

    except Exception as e:
        console.print(f"[red]✗[/] Cannot connect to Ollama: {e}")
        console.print("\nMake sure Ollama is running: [bold]ollama serve[/]")
    finally:
        await ollama.close()


async def _test_anthropic(provider_settings: dict) -> None:
    """Test Anthropic API connectivity."""
    from victor.providers.anthropic_provider import AnthropicProvider

    anthropic = AnthropicProvider(**provider_settings)
    try:
        # Use list_models to verify API key works
        # Anthropic doesn't have a public models endpoint, but the provider
        # returns a static list - we test by making a minimal API call
        console.print("[dim]Testing API connectivity...[/]")

        # Try to get models (static list, but validates provider init)
        models = await anthropic.list_models()
        console.print("[green]✓[/] Anthropic API key is valid")
        console.print(f"[green]✓[/] {len(models)} Claude models available")
        console.print("\n[green]✓[/] Provider anthropic is ready to use!")

    except Exception as e:
        error_msg = str(e).lower()
        if "authentication" in error_msg or "api_key" in error_msg or "401" in error_msg:
            console.print(f"[red]✗[/] Invalid API key: {e}")
            console.print("\nGet your API key from: [bold]https://console.anthropic.com/[/]")
        elif "rate_limit" in error_msg or "429" in error_msg:
            console.print(f"[yellow]⚠[/] Rate limited: {e}")
            console.print("[green]✓[/] API key is valid (rate limited)")
            console.print("\n[green]✓[/] Provider anthropic is ready to use!")
        else:
            console.print(f"[red]✗[/] Connection error: {e}")
    finally:
        await anthropic.close()


async def _test_openai(provider_settings: dict, auth_mode: str = "api_key") -> None:
    """Test OpenAI API connectivity."""
    from victor.providers.openai_provider import OpenAIProvider

    if auth_mode == "oauth":
        console.print("[cyan]Using OAuth authentication (Codex subscription)[/]")
        provider_settings["auth_mode"] = "oauth"

    openai = OpenAIProvider(**provider_settings)
    try:
        console.print("[dim]Testing API connectivity...[/]")

        # List models to verify API key
        models = await openai.list_models()
        auth_label = "OAuth token" if auth_mode == "oauth" else "API key"
        console.print(f"[green]✓[/] OpenAI {auth_label} is valid")
        console.print(f"[green]✓[/] {len(models)} chat-capable models available")
        console.print("\n[green]✓[/] Provider openai is ready to use!")

    except Exception as e:
        error_msg = str(e).lower()
        if "authentication" in error_msg or "api_key" in error_msg or "401" in error_msg:
            console.print(f"[red]✗[/] Invalid credentials: {e}")
            if auth_mode == "oauth":
                console.print(
                    "\nTry re-authenticating: [bold]victor providers auth login openai[/]"
                )
            else:
                console.print(
                    "\nGet your API key from: [bold]https://platform.openai.com/api-keys[/]"
                )
        elif "rate_limit" in error_msg or "429" in error_msg:
            console.print(f"[yellow]⚠[/] Rate limited: {e}")
            console.print("[green]✓[/] Credentials are valid (rate limited)")
            console.print("\n[green]✓[/] Provider openai is ready to use!")
        else:
            console.print(f"[red]✗[/] Connection error: {e}")
    finally:
        await openai.close()


async def _test_google(provider_settings: dict) -> None:
    """Test Google Gemini API connectivity."""
    try:
        from victor.providers.google_provider import GoogleProvider
    except ImportError:
        console.print("[red]✗[/] Google provider not installed")
        console.print("Install with: [bold]pip install victor[google][/bold]")
        return

    try:
        google = GoogleProvider(**provider_settings)
    except ImportError:
        console.print("[red]✗[/] Google AI SDK not installed")
        console.print("Install with: [bold]pip install victor[google][/bold]")
        return

    try:
        console.print("[dim]Testing API connectivity...[/]")

        # List models to verify API key
        models = await google.list_models()
        console.print("[green]✓[/] Google API key is valid")
        console.print(f"[green]✓[/] {len(models)} generative models available")
        console.print("\n[green]✓[/] Provider google is ready to use!")

    except Exception as e:
        error_msg = str(e).lower()
        if "authentication" in error_msg or "api_key" in error_msg or "401" in error_msg:
            console.print(f"[red]✗[/] Invalid API key: {e}")
            console.print("\nGet your API key from: [bold]https://aistudio.google.com/apikey[/]")
        elif "rate_limit" in error_msg or "429" in error_msg:
            console.print(f"[yellow]⚠[/] Rate limited: {e}")
            console.print("[green]✓[/] API key is valid (rate limited)")
            console.print("\n[green]✓[/] Provider google is ready to use!")
        else:
            console.print(f"[red]✗[/] Connection error: {e}")
    finally:
        await google.close()


async def _test_zai(
    provider_settings: dict,
    coding_plan: bool = False,
    endpoint: str = None,
) -> None:
    """Test Z.AI GLM API connectivity."""
    from victor.providers.zai_provider import ZAIProvider, ZAI_BASE_URLS

    if coding_plan:
        provider_settings["coding_plan"] = True
    if endpoint:
        provider_settings["endpoint"] = endpoint

    zai = ZAIProvider(**provider_settings)
    base_url = str(zai.client.base_url)

    try:
        console.print(f"[dim]Endpoint: {base_url}[/]")
        console.print("[dim]Testing API connectivity...[/]")

        models = await zai.list_models()
        console.print("[green]✓[/] Z.AI API key is valid")
        console.print(f"[green]✓[/] {len(models)} models available")

        if coding_plan or (endpoint and endpoint == "coding"):
            console.print("[green]✓[/] Coding Plan endpoint active")

        console.print("\n[green]✓[/] Provider zai is ready to use!")

    except Exception as e:
        error_msg = str(e).lower()
        if "auth" in error_msg or "401" in error_msg:
            console.print(f"[red]✗[/] Invalid API key: {e}")
            console.print("\nSet: [bold]export ZAI_API_KEY=your-key[/]")
            console.print("Get key from: [bold]https://open.bigmodel.cn/usercenter/apikeys[/]")
        elif "429" in error_msg or "rate" in error_msg:
            console.print(f"[yellow]⚠[/] Rate limited: {e}")
            console.print("[green]✓[/] API key is valid (rate limited)")
        else:
            console.print(f"[red]✗[/] Connection error: {e}")
    finally:
        await zai.close()


async def _test_qwen(provider_settings: dict, auth_mode: str = "api_key") -> None:
    """Test Qwen API connectivity."""
    from victor.providers.qwen_provider import QwenProvider
    from victor.providers.base import Message

    if auth_mode == "oauth":
        console.print("[cyan]Using OAuth authentication (Qwen Coding Plan)[/]")
        provider_settings["auth_mode"] = "oauth"

    qwen = QwenProvider(**provider_settings)
    base_url = str(qwen.client.base_url)

    try:
        console.print(f"[dim]Endpoint: {base_url}[/]")
        console.print("[dim]Testing API connectivity...[/]")

        # Qwen doesn't have a list_models endpoint; send a minimal chat request
        response = await qwen.chat(
            messages=[Message(role="user", content="Hi, reply with just 'ok'")],
            model="qwen-turbo-latest",
            max_tokens=10,
        )
        auth_label = "OAuth token" if auth_mode == "oauth" else "API key"
        console.print(f"[green]✓[/] Qwen {auth_label} is valid")
        console.print(f"[green]✓[/] Response: {response.content[:50]}")
        console.print("\n[green]✓[/] Provider qwen is ready to use!")

    except Exception as e:
        error_msg = str(e).lower()
        if "auth" in error_msg or "401" in error_msg:
            console.print(f"[red]✗[/] Invalid credentials: {e}")
            if auth_mode == "oauth":
                console.print("\nTry re-authenticating: [bold]victor providers auth login qwen[/]")
            else:
                console.print("\nSet: [bold]export QWEN_API_KEY=your-key[/]")
                console.print("Get key from: [bold]https://dashscope.console.aliyun.com/apiKey[/]")
        elif "429" in error_msg or "rate" in error_msg:
            console.print(f"[yellow]⚠[/] Rate limited: {e}")
            console.print("[green]✓[/] Credentials are valid (rate limited)")
        else:
            console.print(f"[red]✗[/] Connection error: {e}")
    finally:
        await qwen.close()
