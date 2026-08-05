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

"""AWS Bedrock provider for enterprise model access.

Amazon Bedrock provides a fully managed service for accessing foundation
models from Amazon, Anthropic, Meta, Mistral, and others.

Authentication:
- AWS IAM credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- AWS IAM Role (for EC2, Lambda, ECS)
- AWS SSO / Identity Center
- Environment variables or ~/.aws/credentials

Features:
- Access to Claude, Llama, Mistral, Amazon Titan models
- Enterprise security (VPC, PrivateLink, KMS encryption)
- Provisioned throughput for guaranteed capacity
- Fine-tuning and custom models
- Native tool/function calling

References:
- https://docs.aws.amazon.com/bedrock/
- https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
"""

from victor.core.json_utils import json_loads
from json import JSONDecodeError
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

from victor.providers.base import (
    CacheCostModel,
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.logging import ProviderLogger

logger = logging.getLogger(__name__)

# Bedrock model IDs
BEDROCK_MODELS = {
    # Anthropic Claude models
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "description": "Claude 3.5 Sonnet v2 - Latest, best quality",
        "context_window": 200000,
        "max_output": 8192,
        "supports_tools": True,
    },
    "anthropic.claude-3-5-haiku-20241022-v1:0": {
        "description": "Claude 3.5 Haiku - Fast, efficient",
        "context_window": 200000,
        "max_output": 8192,
        "supports_tools": True,
    },
    "anthropic.claude-3-opus-20240229-v1:0": {
        "description": "Claude 3 Opus - Most capable",
        "context_window": 200000,
        "max_output": 4096,
        "supports_tools": True,
    },
    "anthropic.claude-3-sonnet-20240229-v1:0": {
        "description": "Claude 3 Sonnet - Balanced",
        "context_window": 200000,
        "max_output": 4096,
        "supports_tools": True,
    },
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "description": "Claude 3 Haiku - Fast, affordable",
        "context_window": 200000,
        "max_output": 4096,
        "supports_tools": True,
    },
    # Meta Llama models
    "meta.llama3-2-90b-instruct-v1:0": {
        "description": "Llama 3.2 90B - Large multimodal",
        "context_window": 128000,
        "max_output": 4096,
        "supports_tools": True,
    },
    "meta.llama3-2-11b-instruct-v1:0": {
        "description": "Llama 3.2 11B - Multimodal",
        "context_window": 128000,
        "max_output": 4096,
        "supports_tools": True,
    },
    "meta.llama3-1-405b-instruct-v1:0": {
        "description": "Llama 3.1 405B - Largest open model",
        "context_window": 128000,
        "max_output": 4096,
        "supports_tools": True,
    },
    "meta.llama3-1-70b-instruct-v1:0": {
        "description": "Llama 3.1 70B - High quality",
        "context_window": 128000,
        "max_output": 4096,
        "supports_tools": True,
    },
    "meta.llama3-1-8b-instruct-v1:0": {
        "description": "Llama 3.1 8B - Efficient",
        "context_window": 128000,
        "max_output": 4096,
        "supports_tools": True,
    },
    # Mistral models
    "mistral.mistral-large-2407-v1:0": {
        "description": "Mistral Large - Most capable Mistral",
        "context_window": 128000,
        "max_output": 8192,
        "supports_tools": True,
    },
    "mistral.mistral-small-2402-v1:0": {
        "description": "Mistral Small - Efficient",
        "context_window": 32000,
        "max_output": 8192,
        "supports_tools": True,
    },
    "mistral.mixtral-8x7b-instruct-v0:1": {
        "description": "Mixtral 8x7B MoE - Balanced",
        "context_window": 32000,
        "max_output": 4096,
        "supports_tools": True,
    },
    # Amazon Titan models
    "amazon.titan-text-premier-v1:0": {
        "description": "Titan Text Premier - Amazon's best",
        "context_window": 32000,
        "max_output": 8192,
        "supports_tools": False,
    },
    "amazon.titan-text-express-v1": {
        "description": "Titan Text Express - Fast, affordable",
        "context_window": 8192,
        "max_output": 8192,
        "supports_tools": False,
    },
    "amazon.titan-text-lite-v1": {
        "description": "Titan Text Lite - Cost effective",
        "context_window": 4096,
        "max_output": 4096,
        "supports_tools": False,
    },
    # Cohere models
    "cohere.command-r-plus-v1:0": {
        "description": "Command R+ - RAG optimized, large",
        "context_window": 128000,
        "max_output": 4096,
        "supports_tools": True,
    },
    "cohere.command-r-v1:0": {
        "description": "Command R - RAG optimized",
        "context_window": 128000,
        "max_output": 4096,
        "supports_tools": True,
    },
}


class BedrockProvider(BaseProvider):
    """Provider for AWS Bedrock.

    Features:
    - Access to Claude, Llama, Mistral, Titan models
    - AWS IAM authentication
    - Enterprise security features
    - Tool calling support (model dependent)
    """

    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        region: Optional[str] = None,
        profile_name: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize Bedrock provider.

        Args:
            region: AWS region (default: us-east-1 or AWS_DEFAULT_REGION)
            profile_name: AWS profile name from ~/.aws/credentials
            timeout: Request timeout
            non_interactive: Force non-interactive mode (None = auto-detect)
            **kwargs: Additional configuration (passed to boto3)
        """
        # Initialize structured logger
        self._provider_logger = ProviderLogger("bedrock", __name__)

        self._region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self._profile_name = profile_name
        self._client = None
        self._runtime_client = None

        # Determine authentication source for logging
        auth_source = []
        if profile_name:
            auth_source.append(f"AWS profile: {profile_name}")
        if os.environ.get("AWS_ACCESS_KEY_ID"):
            auth_source.append("AWS_ACCESS_KEY_ID environment variable")
        if os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE"):
            auth_source.append("AWS IAM role (ECS/EKS/Lambda)")
        if not auth_source:
            auth_source.append("Default AWS credential chain")

        # Log provider initialization
        self._provider_logger.log_provider_init(
            model="bedrock",  # Will be set on chat()
            key_source=", ".join(auth_source) if auth_source else None,
            non_interactive=non_interactive if non_interactive is not None else True,
            config={
                "region": self._region,
                "profile_name": profile_name,
                "timeout": timeout,
                **kwargs,
            },
        )

        super().__init__(
            base_url=f"bedrock.{self._region}.amazonaws.com", timeout=timeout, **kwargs
        )

    async def _get_client(self):
        """Get or create boto3 Bedrock runtime client."""
        if self._runtime_client is None:
            try:
                import boto3
                from botocore.config import Config

                config = Config(
                    read_timeout=self.timeout,
                    connect_timeout=30,
                    retries={"max_attempts": 3},
                )

                session_kwargs = {}
                if self._profile_name:
                    session_kwargs["profile_name"] = self._profile_name

                session = boto3.Session(**session_kwargs)
                self._runtime_client = session.client(
                    "bedrock-runtime",
                    region_name=self._region,
                    config=config,
                )
            except ImportError:
                raise ProviderError(
                    message="boto3 is required for AWS Bedrock. Install with: pip install boto3",
                    provider=self.name,
                )
            except Exception as e:
                raise ProviderError(
                    message=f"Failed to initialize Bedrock client: {e}",
                    provider=self.name,
                )
        return self._runtime_client

    @property
    def name(self) -> str:
        return "bedrock"

    def supports_tools(self) -> bool:
        return True  # Depends on model

    def supports_streaming(self) -> bool:
        return True

    def supports_prompt_caching(self) -> bool:
        """AWS Bedrock explicit caching for Claude/Nova (90% read, 1.25x write, 5m TTL)."""
        return True

    def supports_kv_prefix_caching(self) -> bool:
        """Bedrock reuses KV cache for matching prompt prefixes."""
        return True

    def cache_cost_model(self) -> CacheCostModel:
        """Characterized API caching (FEP-0011): 90% read, 1.25x write, 5m TTL (Claude/Nova)."""
        return CacheCostModel(
            supported=True,
            read_discount=0.9,
            write_overhead=1.25,
            ttl_seconds=300.0,
            min_prefix_tokens=1024,
            prefix_granularity="system_block",
        )

    def context_window(self, model: Optional[str] = None) -> int:
        from victor.providers.context_windows import (
            ANTHROPIC,
            ANTHROPIC_DEFAULT,
            lookup,
        )

        target = model or getattr(self, "_current_model", None)
        return lookup(ANTHROPIC, target, ANTHROPIC_DEFAULT)

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
        """Send chat completion request to Bedrock."""
        # Use structured logging context manager
        with self._provider_logger.log_api_call(
            endpoint=f"bedrock.{self._region}.amazonaws.com",
            model=model,
            operation="chat",
            num_messages=len(messages),
            has_tools=tools is not None,
        ) as log_success:
            try:
                client = await self._get_client()

                # Determine model family for request format
                if model.startswith("anthropic."):
                    response = await self._chat_anthropic(
                        client,
                        messages,
                        model,
                        temperature,
                        max_tokens,
                        tools,
                        **kwargs,
                    )
                elif model.startswith("meta."):
                    response = await self._chat_meta(
                        client,
                        messages,
                        model,
                        temperature,
                        max_tokens,
                        tools,
                        **kwargs,
                    )
                elif model.startswith("mistral."):
                    response = await self._chat_mistral(
                        client,
                        messages,
                        model,
                        temperature,
                        max_tokens,
                        tools,
                        **kwargs,
                    )
                else:
                    response = await self._chat_converse(
                        client,
                        messages,
                        model,
                        temperature,
                        max_tokens,
                        tools,
                        **kwargs,
                    )

                # Log success with usage info
                tokens = response.usage.get("total_tokens") if response.usage else None
                log_success(tokens=tokens)

                return response

            except Exception as e:
                raise self.classify_error(e) from e

    async def _chat_converse(
        self, client, messages, model, temperature, max_tokens, tools, **kwargs
    ) -> CompletionResponse:
        """Use Bedrock Converse API (unified format for all models)."""
        import asyncio

        # Build Converse API request
        converse_messages = []
        system_prompt = None

        for msg in messages:
            if msg.role == "system":
                system_prompt = [{"text": msg.content}]
            elif msg.role == "user":
                converse_messages.append({"role": "user", "content": [{"text": msg.content}]})
            elif msg.role == "assistant":
                content = []
                if msg.content:
                    content.append({"text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append(
                            {
                                "toolUse": {
                                    "toolUseId": tc.get("id", ""),
                                    "name": tc.get("name", ""),
                                    "input": tc.get("arguments", {}),
                                }
                            }
                        )
                converse_messages.append({"role": "assistant", "content": content})
            elif msg.role == "tool":
                converse_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": getattr(msg, "tool_call_id", ""),
                                    "content": [{"text": msg.content}],
                                }
                            }
                        ],
                    }
                )

        request_params = {
            "modelId": model,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }

        if system_prompt:
            request_params["system"] = system_prompt

        if tools:
            request_params["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": {"json": tool.parameters},
                        }
                    }
                    for tool in tools
                ]
            }
            # INFO log of the serialized tool payload (first-principles debugging).
            _tool_names = ", ".join(
                t["toolSpec"]["name"] for t in request_params["toolConfig"]["tools"]
            )
            logger.info(
                "[ToolSchemas→LLM] Bedrock converse payload: %d tools: %s",
                len(request_params["toolConfig"]["tools"]),
                _tool_names,
            )

        # Run synchronous boto3 call in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: client.converse(**request_params))

        return self._parse_converse_response(response, model)

    async def _chat_anthropic(
        self, client, messages, model, temperature, max_tokens, tools, **kwargs
    ) -> CompletionResponse:
        """Use native Anthropic format for Claude models (fallback)."""
        # Use Converse API for consistency
        return await self._chat_converse(
            client, messages, model, temperature, max_tokens, tools, **kwargs
        )

    async def _chat_meta(
        self, client, messages, model, temperature, max_tokens, tools, **kwargs
    ) -> CompletionResponse:
        """Handle Meta Llama models."""
        return await self._chat_converse(
            client, messages, model, temperature, max_tokens, tools, **kwargs
        )

    async def _chat_mistral(
        self, client, messages, model, temperature, max_tokens, tools, **kwargs
    ) -> CompletionResponse:
        """Handle Mistral models."""
        return await self._chat_converse(
            client, messages, model, temperature, max_tokens, tools, **kwargs
        )

    def _parse_converse_response(self, response: Dict[str, Any], model: str) -> CompletionResponse:
        """Parse Bedrock Converse API response."""
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        text_content = ""
        tool_calls = []

        for block in content_blocks:
            if "text" in block:
                text_content += block["text"]
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(
                    {
                        "id": tu.get("toolUseId", ""),
                        "name": tu.get("name", ""),
                        "arguments": tu.get("input", {}),
                    }
                )

        # NOT routed through victor.providers.usage_parsing (ADR-0047 D10a wave 2):
        # victor holds the Converse API's camelCase inputTokens/outputTokens block,
        # while sandhi's "bedrock" parser expects the invoke-model shapes
        # (Anthropic-on-Bedrock snake_case usage / Titan). Routing this block parses
        # "successfully" to all-zero counts — a silent mis-meter, not a clean
        # fallback — so the native parse stays until sandhi learns the Converse shape.
        usage = None
        if usage_data := response.get("usage"):
            usage = {
                "prompt_tokens": usage_data.get("inputTokens", 0),
                "completion_tokens": usage_data.get("outputTokens", 0),
                "total_tokens": usage_data.get("inputTokens", 0)
                + usage_data.get("outputTokens", 0),
            }

        return CompletionResponse(
            content=text_content,
            role="assistant",
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=response.get("stopReason", "").lower(),
            usage=usage,
            model=model,
            raw_response=response,
        )

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
        """Stream chat completion from Bedrock using Converse Stream API."""
        # Log stream start
        num_tools = len(tools) if tools else 0
        self._provider_logger.logger.debug(
            f"Starting Bedrock stream: model={model}, msgs={len(messages)}, tools={num_tools}"
        )

        try:
            import asyncio

            client = await self._get_client()

            # Build request same as chat
            converse_messages = []
            system_prompt = None

            for msg in messages:
                if msg.role == "system":
                    system_prompt = [{"text": msg.content}]
                elif msg.role == "user":
                    converse_messages.append({"role": "user", "content": [{"text": msg.content}]})
                elif msg.role == "assistant":
                    content = []
                    if msg.content:
                        content.append({"text": msg.content})
                    converse_messages.append({"role": "assistant", "content": content})

            request_params = {
                "modelId": model,
                "messages": converse_messages,
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                },
            }

            if system_prompt:
                request_params["system"] = system_prompt

            if tools:
                request_params["toolConfig"] = {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": {"json": tool.parameters},
                            }
                        }
                        for tool in tools
                    ]
                }
                # INFO log of the serialized tool payload (first-principles debugging).
                _tool_names = ", ".join(
                    t["toolSpec"]["name"] for t in request_params["toolConfig"]["tools"]
                )
                logger.info(
                    "[ToolSchemas→LLM] Bedrock converse_stream payload: %d tools: %s",
                    len(request_params["toolConfig"]["tools"]),
                    _tool_names,
                )

            # Run synchronous streaming call
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: client.converse_stream(**request_params)
            )

            accumulated_tool_calls: List[Dict[str, Any]] = []
            current_tool_use: Optional[Dict[str, Any]] = None

            for event in response.get("stream", []):
                if "contentBlockStart" in event:
                    start = event["contentBlockStart"]
                    if "toolUse" in start.get("start", {}):
                        tu = start["start"]["toolUse"]
                        current_tool_use = {
                            "id": tu.get("toolUseId", ""),
                            "name": tu.get("name", ""),
                            "arguments": "",
                        }

                elif "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        yield StreamChunk(
                            content=delta["text"],
                            is_final=False,
                        )
                    elif "toolUse" in delta and current_tool_use:
                        current_tool_use["arguments"] += delta["toolUse"].get("input", "")

                elif "contentBlockStop" in event:
                    if current_tool_use and current_tool_use.get("name"):
                        # Parse accumulated JSON arguments
                        args = current_tool_use.get("arguments", "{}")
                        try:
                            args = json_loads(args) if isinstance(args, str) else args
                        except JSONDecodeError:
                            args = {}
                        accumulated_tool_calls.append(
                            {
                                "id": current_tool_use["id"],
                                "name": current_tool_use["name"],
                                "arguments": args,
                            }
                        )
                        current_tool_use = None

                elif "messageStop" in event:
                    yield StreamChunk(
                        content="",
                        tool_calls=(accumulated_tool_calls if accumulated_tool_calls else None),
                        stop_reason=event["messageStop"].get("stopReason", "stop").lower(),
                        is_final=True,
                    )

        except Exception as e:
            raise self.classify_error(e) from e

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Bedrock foundation models."""
        try:
            import boto3
            import asyncio

            session_kwargs = {}
            if self._profile_name:
                session_kwargs["profile_name"] = self._profile_name

            session = boto3.Session(**session_kwargs)
            bedrock_client = session.client("bedrock", region_name=self._region)

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, bedrock_client.list_foundation_models)

            return [
                {
                    "id": model["modelId"],
                    "name": model.get("modelName", model["modelId"]),
                    "provider": model.get("providerName", "unknown"),
                    "input_modalities": model.get("inputModalities", []),
                    "output_modalities": model.get("outputModalities", []),
                }
                for model in response.get("modelSummaries", [])
            ]

        except Exception as e:
            self._provider_logger.logger.debug(f"Failed to list Bedrock models: {e}")
            return [
                {"id": model_id, **model_info} for model_id, model_info in BEDROCK_MODELS.items()
            ]

    async def close(self) -> None:
        """Close the Bedrock client."""
        self._runtime_client = None
        self._client = None
