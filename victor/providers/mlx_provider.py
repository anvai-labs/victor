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

"""MLX LM provider for Apple Silicon-optimized inference.

MLX is Apple's machine learning framework designed specifically for Silicon chips.
This provider loads models in-process for best performance using unified memory.

Usage:
    # Install MLX LM
    pip install mlx-lm

    # Use with Victor
    victor chat --provider mlx --model mlx-community/Qwen2.5-7B-Instruct-4bit
    victor chat --profile mlx-qwen

Recommended MLX models (quantized for efficiency):
    - mlx-community/Qwen2.5-7B-Instruct-4bit (4GB RAM)
    - mlx-community/Qwen2.5-14B-Instruct-4bit (8GB RAM)
    - mlx-community/Qwen2.5-Coder-7B-Instruct-4bit (coding optimized)
    - mlx-community/Llama-3.2-3B-Instruct-4bit (2GB RAM, fast)
    - mlx-community/Mistral-7B-Instruct-v0.3-4bit (4GB RAM)

Download models from: https://huggingface.co/mlx-community
"""

import asyncio
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from victor.providers.base import (
    CacheCostModel,
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.logging import ProviderLogger

# NOTE: Keep mlx_lm imports out of module top-level.
# Importing mlx can crash process on some environments with broken Metal/MPS.
_MLX_IMPORT_ATTEMPTED = False
_MLX_AVAILABLE = False
_MLX_IMPORT_ERROR: Optional[Exception] = None
_mlx_load = None
_mlx_stream_generate = None


def _ensure_mlx_imported() -> None:
    """Import mlx_lm lazily and cache availability state."""
    global _MLX_IMPORT_ATTEMPTED
    global _MLX_AVAILABLE
    global _MLX_IMPORT_ERROR
    global _mlx_load
    global _mlx_stream_generate

    if _MLX_IMPORT_ATTEMPTED:
        if not _MLX_AVAILABLE:
            detail = f": {_MLX_IMPORT_ERROR}" if _MLX_IMPORT_ERROR else ""
            raise ImportError(f"mlx-lm is not available{detail}")
        return

    _MLX_IMPORT_ATTEMPTED = True
    try:
        from mlx_lm import load as _load
        from mlx_lm import stream_generate as _stream_generate

        _mlx_load = _load
        _mlx_stream_generate = _stream_generate
        _MLX_AVAILABLE = True
    except Exception as exc:
        _MLX_IMPORT_ERROR = exc
        _MLX_AVAILABLE = False
        raise ImportError(f"mlx-lm is not available: {exc}") from exc


# Models that support tool calling (instruction-tuned)
TOOL_CAPABLE_PATTERNS = [
    "instruct",
    "chat",
    "coder",
    "-it",
    "qwen",
    "llama-3",
    "mistral",
    "deepseek",
]


def _model_supports_tools(model: str) -> bool:
    """Check if model likely supports tool calling.

    Args:
        model: Model name/path

    Returns:
        True if model likely supports tools
    """
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in TOOL_CAPABLE_PATTERNS)


class MLXProvider(BaseProvider):
    """Provider for MLX LM (Apple Silicon optimized inference).

    MLX provides efficient CPU+GPU inference using Apple's unified memory architecture.
    Models are loaded in-process for best performance.

    Attributes:
        model_path: Path or HuggingFace ID of MLX model
        model_kwargs: Additional arguments for model loading
    """

    def __init__(
        self,
        model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize MLX provider.

        Args:
            model: Model path or HuggingFace ID (default: small fast model)
            non_interactive: Non-interactive mode flag (for consistency)
            **kwargs: Additional configuration (ignored for MLX)
        """
        try:
            _ensure_mlx_imported()
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is not available in this runtime. "
                "Install and verify with: python -c 'import mlx_lm'"
            ) from exc

        super().__init__(api_key="not-needed", base_url="in-process", **kwargs)

        self.model_path = model
        self.model_kwargs = {
            "trust_remote_code": kwargs.get("trust_remote_code", False),
        }
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._load_lock = asyncio.Lock()

        # Initialize provider logger
        self._provider_logger = ProviderLogger("mlx", __name__)
        self._provider_logger.log_provider_init(
            model=model,
            key_source=None,  # MLX is local, no API key
            non_interactive=non_interactive or False,
            config=self.model_kwargs,
        )

    async def _ensure_model_loaded(self) -> None:
        """Load MLX model if not already loaded (thread-safe).

        This is called automatically on first request.
        """
        if self._model is not None:
            return

        async with self._load_lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return

            try:
                self._provider_logger.logger.info(f"Loading MLX model: {self.model_path}")
                if _mlx_load is None:
                    raise RuntimeError("mlx-lm loader unavailable")

                # Try loading with model_kwargs, filter out unsupported kwargs
                load_kwargs = self.model_kwargs.copy()
                loop = asyncio.get_event_loop()

                # First attempt: with all kwargs
                try:
                    self._model, self._tokenizer = await loop.run_in_executor(
                        None,
                        lambda: _mlx_load(self.model_path, **load_kwargs),
                    )
                except TypeError as e:
                    # If load() doesn't accept some kwargs (like trust_remote_code),
                    # try without the problematic ones
                    if "trust_remote_code" in str(e) and "trust_remote_code" in load_kwargs:
                        self._provider_logger.logger.debug(
                            "MLX load() doesn't accept trust_remote_code, retrying without it"
                        )
                        load_kwargs = {
                            k: v for k, v in load_kwargs.items() if k != "trust_remote_code"
                        }
                        self._model, self._tokenizer = await loop.run_in_executor(
                            None,
                            lambda: _mlx_load(self.model_path, **load_kwargs),
                        )
                    else:
                        raise

                self._provider_logger.logger.info(f"MLX model loaded: {self.model_path}")
            except Exception as e:
                # Convert specific exception types to provider errors
                if isinstance(e, ProviderError):
                    raise
                raise ProviderConnectionError(f"Failed to load MLX model {self.model_path}: {e}")

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming.

        Returns:
            True (MLX supports streaming via stream_generate)
        """
        return True

    def supports_prompt_caching(self) -> bool:
        """MLX has no API-level prompt caching (no billing discount)."""
        return False

    def supports_kv_prefix_caching(self) -> bool:
        """MLX reuses KV cache for pure-attention models with matching prefixes."""
        return True

    def kv_cache_cost_model(self) -> CacheCostModel:
        """KV-prefix caching (FEP-0011): latency-only, no billing discount."""
        return CacheCostModel(
            supported=True,
            read_discount=0.0,
            ttl_seconds=0.0,
            prefix_granularity="system_block",
        )

    def context_window(self, model: Optional[str] = None) -> int:
        from victor.providers.context_windows import OLLAMA, MLX_DEFAULT, lookup

        target = model or getattr(self, "_current_model", None)
        return lookup(OLLAMA, target, MLX_DEFAULT)

    def supports_tools(self) -> bool:
        """Check if provider supports tool calling.

        Returns:
            True if model likely supports tools based on name
        """
        return _model_supports_tools(self.model_path)

    @property
    def name(self) -> str:
        """Provider name.

        Returns:
            Provider identifier
        """
        return "mlx"

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send a chat completion request.

        Args:
            messages: List of conversation messages
            model: Model identifier
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: Available tools for the model to use
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResponse with generated content
        """
        return await self._make_request(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        )

    async def close(self) -> None:
        """Close any open connections or resources.

        For MLX, this just releases model references to allow GC.
        """
        self._model = None
        self._tokenizer = None
        self._provider_logger.logger.debug(
            f"MLX provider resources released for model: {self.model_path}"
        )

    async def _make_request(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a chat completion (non-streaming).

        Args:
            messages: Conversation messages
            model: Model identifier (uses self.model_path if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools (not parsed by MLX, extracted from content)
            **kwargs: Additional parameters

        Returns:
            CompletionResponse with generated content
        """
        await self._ensure_model_loaded()

        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)

        # Generate response in thread pool with structured logging
        loop = asyncio.get_event_loop()
        with self._provider_logger.log_api_call(
            endpoint="local",
            model=model or self.model_path,
            operation="chat",
        ):
            try:
                response_text = await loop.run_in_executor(
                    None,
                    lambda: self._sync_generate(
                        prompt=prompt,
                        temp=temperature,
                        max_tokens=max_tokens,
                    ),
                )
            except ProviderError:
                # Re-raise provider errors as-is
                raise
            except Exception as e:
                # Convert generic exceptions to ProviderError
                raise ProviderError(f"MLX generation failed: {e}")

        # Extract tool calls if present
        from victor.providers.openai_compat import extract_tool_calls_from_content

        tool_calls, content = extract_tool_calls_from_content(response_text, id_prefix="mlx")

        return CompletionResponse(
            content=content,
            model=model or self.model_path,
            tool_calls=tool_calls if tool_calls else None,
        )

    def _sync_generate(
        self,
        prompt: str,
        temp: float,
        max_tokens: int,
    ) -> str:
        """Synchronous generation wrapper for thread pool execution.

        Args:
            prompt: Formatted prompt
            temp: Temperature
            max_tokens: Maximum tokens

        Returns:
            Generated text
        """
        if _mlx_stream_generate is None:
            raise RuntimeError("mlx-lm not properly installed")

        # Use stream_generate and collect all chunks
        text_chunks = []
        for response in _mlx_stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=max_tokens,
            temp=temp,
            verbose=False,
        ):
            text_chunks.append(response.text)

        return "".join(text_chunks)

    def _messages_to_prompt(self, messages: List[Message]) -> str:
        """Convert messages to prompt using tokenizer's chat template.

        Args:
            messages: List of Message objects

        Returns:
            Formatted prompt string
        """
        # Convert to MLX format
        mlx_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Apply chat template if available
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                mlx_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            # Fallback: simple concatenation
            prompt = "\n".join(f"{msg.role}: {msg.content}" for msg in messages)
            prompt += "\nassistant:"

        return prompt

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion response.

        Args:
            messages: Conversation messages
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional parameters

        Yields:
            StreamChunk objects with incremental content
        """
        await self._ensure_model_loaded()

        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)

        # Stream in thread pool with structured logging
        loop = asyncio.get_event_loop()

        def generate_sync():
            """Synchronous generator for streaming."""
            try:
                if _mlx_stream_generate is None:
                    raise RuntimeError("mlx-lm stream generator unavailable")
                for response in _mlx_stream_generate(
                    self._model,
                    self._tokenizer,
                    prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                    verbose=False,
                ):
                    yield response.text
            except ProviderError:
                # Re-raise provider errors as-is
                raise
            except Exception as e:
                # Convert generic exceptions to ProviderError
                self._provider_logger.logger.error(f"MLX streaming error: {e}")
                raise ProviderError(f"MLX streaming failed: {e}")

        # Create async generator from sync generator
        stream_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        async def stream_producer():
            """Run sync generator in thread pool and feed queue."""
            try:
                for chunk in await loop.run_in_executor(None, lambda: list(generate_sync())):
                    await stream_queue.put(chunk)
                await stream_queue.put(None)  # Sentinel
            except ProviderError:
                # Re-raise provider errors as-is
                raise
            except Exception as e:
                await stream_queue.put(e)

        # Start producer task with structured logging
        with self._provider_logger.log_api_call(
            endpoint="local",
            model=model or self.model_path,
            operation="stream",
        ):
            producer_task = asyncio.create_task(stream_producer())

            try:
                while True:
                    item = await stream_queue.get()
                    if item is None:  # Sentinel
                        break
                    if isinstance(item, Exception):
                        # Convert specific exception types to provider errors
                        if isinstance(item, ProviderError):
                            raise item
                        raise ProviderError(f"MLX streaming failed: {item}")

                    yield StreamChunk(content=item, model=model or self.model_path)
            finally:
                await producer_task

    async def check_connection(self) -> bool:
        """Check if MLX is available and model can be loaded.

        Returns:
            True if MLX is available
        """
        if not _MLX_AVAILABLE:
            return False

        try:
            await self._ensure_model_loaded()
            return True
        except Exception as e:
            self._provider_logger.logger.warning(f"MLX connection check failed: {e}")
            return False

    async def list_models(self) -> List[str]:
        """List available MLX models.

        Note: This returns a small curated list of recommended models.
        In practice, any HuggingFace model compatible with MLX can be used.

        Returns:
            List of model identifiers
        """
        return [
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "mlx-community/Qwen2.5-14B-Instruct-4bit",
            "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "mlx-community/Phi-3.5-mini-4bit-instruct",
        ]

    def __repr__(self) -> str:
        """String representation of provider.

        Returns:
            Provider description
        """
        return f"MLXProvider(model='{self.model_path}')"
