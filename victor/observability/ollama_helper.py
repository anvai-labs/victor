"""Ollama model detection helper.

This module provides utilities to detect available ollama models
and select the simplest one for testing.
"""

import json
import socket
from typing import List, Optional


def get_ollama_models() -> List[str]:
    """Get list of available ollama LLM models from localhost API.

    Filters out embedding models to only return text generation models.

    Returns:
        List of LLM model names (e.g., ["llama3.2", "phi3", "gemma2:2b"])

    Example:
        >>> models = get_ollama_models()
        >>> print(models)
        ['llama3.2', 'phi3', 'gemma2:2b']
    """
    try:
        import urllib.request
        import urllib.error

        # Try to connect to ollama API
        url = "http://localhost:11434/api/tags"
        request = urllib.request.Request(url, method="GET")
        request.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(request, timeout=2) as response:
            data = json.loads(response.read().decode())

            # Filter out embedding models
            models = []
            for model in data.get("models", []):
                model_name = model["name"].split(":")[0]  # Remove tag

                # Skip embedding models
                if "embed" in model_name.lower():
                    continue

                models.append(model["name"])

            return models

    except (urllib.error.URLError, socket.error, json.JSONDecodeError, KeyError):
        # Ollama not available or error
        return []


def select_simplest_model(models: List[str]) -> Optional[str]:
    """Select the simplest model from a list of ollama models.

    Prioritizes:
    1. Models with "2b" or "1b" in name (smallest parameter count)
    2. Models with "tiny" or "mini" in name
    3. Models with "phi" (usually small)
    4. Models with "gemma" (usually efficient)
    5. First available model

    Args:
        models: List of model names

    Returns:
        Simplest model name or None if no models available

    Example:
        >>> models = ["llama3.2", "phi3", "gemma2:2b"]
        >>> select_simplest_model(models)
        'gemma2:2b'
    """
    if not models:
        return None

    # Priority 1: Smallest parameter count (2b, 1b)
    for model in models:
        if "2b" in model.lower() or "1b" in model.lower():
            return model.split(":")[0]  # Remove tag if present

    # Priority 2: Models with "tiny" or "mini"
    for model in models:
        if "tiny" in model.lower() or "mini" in model.lower():
            return model.split(":")[0]

    # Priority 3: phi models (usually small and fast)
    for model in models:
        if model.lower().startswith("phi"):
            return model.split(":")[0]

    # Priority 4: gemma models (efficient)
    for model in models:
        if model.lower().startswith("gemma"):
            return model.split(":")[0]

    # Priority 5: qwen models (often have small versions)
    for model in models:
        if model.lower().startswith("qwen"):
            return model.split(":")[0]

    # Fallback: First available model (without tag)
    return models[0].split(":")[0]


def get_default_ollama_model() -> str:
    """Get the default ollama model for testing.

    Tries to:
    1. Detect available models from ollama
    2. Select the simplest one
    3. Fall back to "gpt-oss" if ollama not available

    Returns:
        Model name to use for testing

    Example:
        >>> model = get_default_ollama_model()
        >>> print(f"Using model: {model}")
        Using model: phi3
    """
    models = get_ollama_models()

    if models:
        simplest = select_simplest_model(models)
        if simplest:
            return simplest

    # Fallback if ollama not available
    return "gpt-oss"


def is_ollama_available() -> bool:
    """Check if ollama is available on localhost.

    Returns:
        True if ollama API is accessible

    Example:
        >>> if is_ollama_available():
        ...     print("Ollama is ready!")
        ... else:
        ...     print("Ollama not available, using fallback")
    """
    models = get_ollama_models()
    return len(models) > 0
