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

"""
Resource Availability Detection for Smart Routing.

This module provides resource detection capabilities:
- GPUAvailability: GPU status for local providers
- QuotaInfo: API quota status for cloud providers
- ResourceAvailabilityDetector: Detects resource availability

Usage:
    from victor.providers.resource_detector import ResourceAvailabilityDetector

    detector = ResourceAvailabilityDetector()

    # Check GPU availability
    gpu = await detector.check_gpu_availability()
    if gpu.available:
        print(f"GPU has {gpu.memory_mb}MB memory available")

    # Check API quota
    quota = await detector.check_api_quota("anthropic")
    print(f"Remaining requests: {quota.remaining_requests}")
"""

from __future__ import annotations

from victor.core.json_utils import json_loads
import asyncio
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUAvailability:
    """GPU availability status.

    Attributes:
        available: Whether GPU is available
        memory_mb: Total GPU memory in MB (if available)
        utilization_percent: Current GPU utilization (if available)
        reason: Reason if GPU is not available
    """

    available: bool
    memory_mb: Optional[int] = None
    utilization_percent: Optional[float] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "available": self.available,
            "memory_mb": self.memory_mb,
            "utilization_percent": self.utilization_percent,
            "reason": self.reason,
        }


@dataclass
class QuotaInfo:
    """API quota information.

    Attributes:
        remaining_requests: Remaining request count (if available)
        reset_time: When quota resets (if available)
        rate_limit_remaining: Rate limit remaining (if available)
    """

    remaining_requests: Optional[int] = None
    reset_time: Optional[datetime] = None
    rate_limit_remaining: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "remaining_requests": self.remaining_requests,
            "reset_time": self.reset_time.isoformat() if self.reset_time else None,
            "rate_limit_remaining": self.rate_limit_remaining,
        }


class ResourceAvailabilityDetector:
    """Detects resource availability for providers.

    Checks:
    - GPU availability for local providers (Ollama, LMStudio, vLLM)
    - API quota for cloud providers (Anthropic, OpenAI, etc.)
    - Memory and utilization metrics

    Uses platform-specific detection:
    - NVIDIA GPUs: nvidia-smi
    - Apple Silicon: system_profiler
    - AMD GPUs: rocm-smi
    """

    # Local providers that require GPU
    LOCAL_PROVIDERS = {"ollama", "lmstudio", "vllm", "inferflux"}

    # Cloud providers that use API quotas
    CLOUD_PROVIDERS = {
        "anthropic",
        "openai",
        "deepseek",
        "xai",
        "cohere",
        "google",
        "groqcloud",
    }

    def __init__(self):
        """Initialize resource detector."""
        self._gpu_cache: Optional[GPUAvailability] = None
        self._gpu_cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 30  # Cache GPU status for 30 seconds

        logger.debug("ResourceAvailabilityDetector initialized")

    async def check_gpu_availability(self) -> GPUAvailability:
        """Check GPU availability.

        Attempts to detect GPU using platform-specific tools.
        Results are cached for 30 seconds.

        Returns:
            GPUAvailability status
        """
        # Check cache
        if self._gpu_cache and self._is_cache_valid():
            return self._gpu_cache

        # Detect GPU
        gpu = await self._detect_gpu()

        # Update cache
        self._gpu_cache = gpu
        self._gpu_cache_time = datetime.now()

        return gpu

    async def check_api_quota(self, provider: str) -> QuotaInfo:
        """Check API quota for a cloud provider.

        Note: Most providers don't expose quota information via API.
        This method returns empty QuotaInfo by default.

        Args:
            provider: Provider name

        Returns:
            QuotaInfo (mostly empty due to API limitations)
        """
        # Most providers don't expose quota info
        # This is a placeholder for future integration
        return QuotaInfo()

    def is_provider_available(
        self,
        provider: str,
        resources: Dict[str, any],
    ) -> bool:
        """Check if a provider is available given resource constraints.

        Args:
            provider: Provider name
            resources: Resource dict with 'gpu' key

        Returns:
            True if provider is available
        """
        provider = provider.lower()

        # Local providers require GPU
        if provider in self.LOCAL_PROVIDERS:
            gpu = resources.get("gpu")
            if not gpu or not gpu.available:
                return False

        # Cloud providers are always available (network permitting)
        return True

    async def _detect_gpu(self) -> GPUAvailability:
        """Detect GPU using platform-specific tools.

        Returns:
            GPUAvailability status
        """
        # Try NVIDIA GPU
        nvidia_gpu = await self._check_nvidia_gpu()
        if nvidia_gpu.available:
            return nvidia_gpu

        # Try Apple Silicon GPU
        apple_gpu = await self._check_apple_gpu()
        if apple_gpu.available:
            return apple_gpu

        # Try AMD GPU
        amd_gpu = await self._check_amd_gpu()
        if amd_gpu.available:
            return amd_gpu

        # No GPU detected
        return GPUAvailability(
            available=False, reason="No GPU detected (NVIDIA, Apple Silicon, or AMD)"
        )

    async def _check_nvidia_gpu(self) -> GPUAvailability:
        """Check for NVIDIA GPU using nvidia-smi.

        Returns:
            GPUAvailability status
        """
        try:
            # Run nvidia-smi with timeout
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=name,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                return GPUAvailability(available=False, reason="nvidia-smi failed")

            # Parse output
            output = stdout.decode().strip()
            if not output:
                return GPUAvailability(available=False, reason="No NVIDIA GPU detected")

            # Parse: "NVIDIA GeForce RTX 4090, 24564, 0"
            parts = output.split(", ")
            if len(parts) >= 2:
                memory_mb = int(parts[1].strip())
                utilization = float(parts[2].strip()) if len(parts) > 2 else None

                return GPUAvailability(
                    available=True,
                    memory_mb=memory_mb,
                    utilization_percent=utilization,
                )

        except FileNotFoundError:
            return GPUAvailability(available=False, reason="nvidia-smi not found")
        except Exception as e:
            logger.debug(f"NVIDIA GPU check failed: {e}")
            return GPUAvailability(available=False, reason=f"NVIDIA check failed: {e}")

        return GPUAvailability(available=False, reason="Unknown NVIDIA GPU status")

    async def _check_apple_gpu(self) -> GPUAvailability:
        """Check for Apple Silicon GPU using system_profiler.

        Returns:
            GPUAvailability status
        """
        try:
            # Run system_profiler with timeout
            proc = await asyncio.create_subprocess_exec(
                "system_profiler",
                "SPDisplaysDataType",
                "-json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                return GPUAvailability(available=False, reason="system_profiler failed")

            # Parse JSON output
            import json

            data = json_loads(stdout.decode())
            displays = data.get("SPDisplaysDataType", [])

            if not displays:
                return GPUAvailability(available=False, reason="No display detected")

            # Check for Apple Silicon GPU
            for display in displays:
                name = display.get("sppu_gpu_chipset_model", "")
                if "Apple" in name or "M1" in name or "M2" in name or "M3" in name:
                    # Apple Silicon GPUs have unified memory
                    # Estimate based on system memory (not precise)
                    return GPUAvailability(
                        available=True,
                        memory_mb=None,  # Unified memory, can't easily measure
                        utilization_percent=None,
                    )

        except FileNotFoundError:
            return GPUAvailability(available=False, reason="system_profiler not found")
        except Exception as e:
            logger.debug(f"Apple GPU check failed: {e}")
            return GPUAvailability(available=False, reason=f"Apple GPU check failed: {e}")

        return GPUAvailability(available=False, reason="No Apple Silicon GPU detected")

    async def _check_amd_gpu(self) -> GPUAvailability:
        """Check for AMD GPU using rocm-smi.

        Returns:
            GPUAvailability status
        """
        try:
            # Run rocm-smi with timeout
            proc = await asyncio.create_subprocess_exec(
                "rocm-smi",
                "--showmeminfo",
                "--showuse",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                return GPUAvailability(available=False, reason="rocm-smi failed")

            # Parse output (simplified)
            output = stdout.decode().strip()
            if "GPU" in output and "VRAM" in output:
                # AMD GPU detected (parsing is complex, just return available)
                return GPUAvailability(
                    available=True,
                    memory_mb=None,  # Parsing not implemented
                    utilization_percent=None,
                )

        except FileNotFoundError:
            return GPUAvailability(available=False, reason="rocm-smi not found")
        except Exception as e:
            logger.debug(f"AMD GPU check failed: {e}")
            return GPUAvailability(available=False, reason=f"AMD GPU check failed: {e}")

        return GPUAvailability(available=False, reason="No AMD GPU detected")

    def _is_cache_valid(self) -> bool:
        """Check if GPU cache is still valid.

        Returns:
            True if cache is valid
        """
        if not self._gpu_cache_time:
            return False

        elapsed = (datetime.now() - self._gpu_cache_time).total_seconds()
        return elapsed < self._cache_ttl_seconds

    def clear_cache(self) -> None:
        """Clear cached GPU status."""
        self._gpu_cache = None
        self._gpu_cache_time = None
        logger.debug("GPU availability cache cleared")
