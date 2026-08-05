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
Unit tests for resource availability detector.
"""

import pytest
from datetime import datetime

from victor.providers.resource_detector import (
    GPUAvailability,
    QuotaInfo,
    ResourceAvailabilityDetector,
)


class TestGPUAvailability:
    """Tests for GPUAvailability dataclass."""

    def test_gpu_available(self):
        """Test GPU available status."""
        gpu = GPUAvailability(
            available=True,
            memory_mb=24564,
            utilization_percent=50.0,
        )

        assert gpu.available is True
        assert gpu.memory_mb == 24564
        assert gpu.utilization_percent == 50.0
        assert gpu.reason is None

    def test_gpu_unavailable(self):
        """Test GPU unavailable status."""
        gpu = GPUAvailability(
            available=False,
            reason="No GPU detected",
        )

        assert gpu.available is False
        assert gpu.reason == "No GPU detected"
        assert gpu.memory_mb is None
        assert gpu.utilization_percent is None

    def test_to_dict(self):
        """Test converting GPU availability to dictionary."""
        gpu = GPUAvailability(
            available=True,
            memory_mb=16384,
            utilization_percent=75.0,
        )

        data = gpu.to_dict()

        assert data["available"] is True
        assert data["memory_mb"] == 16384
        assert data["utilization_percent"] == 75.0
        assert data["reason"] is None


class TestQuotaInfo:
    """Tests for QuotaInfo dataclass."""

    def test_quota_info(self):
        """Test quota information."""
        quota = QuotaInfo(
            remaining_requests=1000,
            rate_limit_remaining=50,
        )

        assert quota.remaining_requests == 1000
        assert quota.rate_limit_remaining == 50
        assert quota.reset_time is None

    def test_quota_with_reset_time(self):
        """Test quota with reset time."""
        reset_time = datetime.now()
        quota = QuotaInfo(
            remaining_requests=500,
            reset_time=reset_time,
        )

        assert quota.remaining_requests == 500
        assert quota.reset_time == reset_time

    def test_to_dict(self):
        """Test converting quota info to dictionary."""
        reset_time = datetime.now()
        quota = QuotaInfo(
            remaining_requests=100,
            reset_time=reset_time,
        )

        data = quota.to_dict()

        assert data["remaining_requests"] == 100
        assert data["reset_time"] == reset_time.isoformat()
        assert data["rate_limit_remaining"] is None


class TestResourceAvailabilityDetector:
    """Tests for ResourceAvailabilityDetector."""

    @pytest.fixture
    def detector(self):
        """Create a resource detector instance."""
        return ResourceAvailabilityDetector()

    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector is not None
        assert detector._cache_ttl_seconds == 30
        assert detector._gpu_cache is None

    def test_local_providers_set(self, detector):
        """Test that local providers are defined."""
        assert "ollama" in detector.LOCAL_PROVIDERS
        assert "lmstudio" in detector.LOCAL_PROVIDERS
        assert "vllm" in detector.LOCAL_PROVIDERS

    def test_cloud_providers_set(self, detector):
        """Test that cloud providers are defined."""
        assert "anthropic" in detector.CLOUD_PROVIDERS
        assert "openai" in detector.CLOUD_PROVIDERS
        assert "deepseek" in detector.CLOUD_PROVIDERS

    @pytest.mark.asyncio
    async def test_check_gpu_availability(self, detector):
        """Test checking GPU availability."""
        gpu = await detector.check_gpu_availability()

        # Should return GPUAvailability regardless of result
        assert isinstance(gpu, GPUAvailability)

        # If GPU is available, should have memory info
        if gpu.available:
            # Just verify it's a valid response
            assert gpu.memory_mb is not None or gpu.utilization_percent is not None
        else:
            # Should have reason
            assert gpu.reason is not None

    @pytest.mark.asyncio
    async def test_check_gpu_caching(self, detector):
        """Test that GPU status is cached."""
        # First call
        gpu1 = await detector.check_gpu_availability()
        cache_time = detector._gpu_cache_time

        # Second call should use cache
        gpu2 = await detector.check_gpu_availability()

        assert gpu1.available == gpu2.available
        assert detector._gpu_cache_time == cache_time

    @pytest.mark.asyncio
    async def test_check_api_quota(self, detector):
        """Test checking API quota."""
        quota = await detector.check_api_quota("anthropic")

        # Should return QuotaInfo (mostly empty due to API limitations)
        assert isinstance(quota, QuotaInfo)

    def test_is_provider_available_local_with_gpu(self, detector):
        """Test local provider availability with GPU."""
        gpu = GPUAvailability(available=True, memory_mb=16384)
        resources = {"gpu": gpu}

        # Local provider should be available
        assert detector.is_provider_available("ollama", resources) is True
        assert detector.is_provider_available("lmstudio", resources) is True

    def test_is_provider_available_local_without_gpu(self, detector):
        """Test local provider availability without GPU."""
        gpu = GPUAvailability(available=False, reason="No GPU")
        resources = {"gpu": gpu}

        # Local provider should not be available
        assert detector.is_provider_available("ollama", resources) is False
        assert detector.is_provider_available("lmstudio", resources) is False

    def test_is_provider_available_cloud(self, detector):
        """Test cloud provider availability."""
        # Cloud providers don't require GPU
        resources = {}

        assert detector.is_provider_available("anthropic", resources) is True
        assert detector.is_provider_available("openai", resources) is True

    def test_clear_cache(self, detector):
        """Test clearing GPU cache."""
        # Populate cache
        detector._gpu_cache = GPUAvailability(available=True)
        detector._gpu_cache_time = datetime.now()

        detector.clear_cache()

        assert detector._gpu_cache is None
        assert detector._gpu_cache_time is None

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, detector):
        """Test that cache expires after TTL."""
        # Set cache time to past
        detector._gpu_cache = GPUAvailability(available=True)
        detector._gpu_cache_time = datetime.now()

        # Modify TTL to 0 for testing
        detector._cache_ttl_seconds = 0

        # Cache should be invalid
        assert not detector._is_cache_valid()

    @pytest.mark.asyncio
    async def test_check_nvidia_gpu(self, detector):
        """Test NVIDIA GPU detection."""
        # This will fail on systems without NVIDIA GPU
        # Just verify it doesn't crash
        gpu = await detector._check_nvidia_gpu()

        assert isinstance(gpu, GPUAvailability)

    @pytest.mark.asyncio
    async def test_check_apple_gpu(self, detector):
        """Test Apple Silicon GPU detection."""
        # This will fail on non-Apple systems
        # Just verify it doesn't crash
        gpu = await detector._check_apple_gpu()

        assert isinstance(gpu, GPUAvailability)

    @pytest.mark.asyncio
    async def test_check_amd_gpu(self, detector):
        """Test AMD GPU detection."""
        # This will fail on systems without AMD GPU
        # Just verify it doesn't crash
        gpu = await detector._check_amd_gpu()

        assert isinstance(gpu, GPUAvailability)

    def test_is_provider_available_local_without_gpu(self, detector):
        """Test local provider availability without GPU."""
        gpu = GPUAvailability(available=False, reason="No GPU")
        resources = {"gpu": gpu}

        # Local provider should not be available
        assert detector.is_provider_available("ollama", resources) is False
        assert detector.is_provider_available("lmstudio", resources) is False
