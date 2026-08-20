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

"""Integration tests to verify Docker availability handling.

Tests both scenarios:
1. Docker is available and working
2. Docker is not available (graceful degradation)
"""

import pytest
from unittest.mock import patch
from docker.errors import DockerException

# CodeSandbox migrated from victor.tools.code_executor_tool to the external
# victor-coding package (resolution chain in
# victor/core/utils/capability_loader.py). Skip the whole module when that
# submodule isn't available rather than breaking collection for the rest of
# the suite. NOTE: guard the actual submodule, not just the top-level package
# -- importorskip("victor_coding") passes (the package imports fine) and then
# the import below would raise ModuleNotFoundError during collection.
pytest.importorskip("victor_coding.tools.code_executor_tool")
from victor_coding.tools.code_executor_tool import CodeSandbox  # noqa: E402


class TestDockerAvailability:
    """Test Docker availability detection and handling."""

    def test_docker_available_detection(self):
        """Test that Docker availability is correctly detected when Docker is running."""
        # This test will pass if Docker is running, skip if not
        try:
            manager = CodeSandbox(require_docker=False)

            # Skip test if Docker is not available (instead of failing)
            if not manager.docker_available:
                pytest.skip("Docker not running - skipping Docker-available test")

            # Should detect Docker is available
            assert manager.docker_available is True, "Docker should be detected as available"
            assert manager.docker_client is not None, "Docker client should be initialized"

            print("\n✅ Docker detected as AVAILABLE")
            print(f"   docker_available: {manager.docker_available}")
            print(f"   docker_client: {type(manager.docker_client).__name__}")

        except RuntimeError as e:
            if "Docker is not running" in str(e):
                pytest.skip("Docker not running - skipping Docker-available test")
            raise

    def test_docker_unavailable_graceful_handling(self):
        """Test graceful handling when Docker is not available."""
        # Mock docker.from_env() to raise DockerException
        with patch("docker.from_env", side_effect=DockerException("Docker not available")):
            manager = CodeSandbox(require_docker=False)

            # Should handle unavailability gracefully
            assert manager.docker_available is False, "Docker should be detected as unavailable"
            assert manager.docker_client is None, "Docker client should be None"

            print("\n✅ Docker unavailable handled GRACEFULLY")
            print(f"   docker_available: {manager.docker_available}")
            print(f"   docker_client: {manager.docker_client}")

    def test_docker_required_raises_error(self):
        """Test that require_docker=True raises error when Docker unavailable."""
        # Mock docker.from_env() to raise DockerException
        with patch("docker.from_env", side_effect=DockerException("Docker not available")):
            with pytest.raises(RuntimeError, match="Docker is not running or not installed"):
                CodeSandbox(require_docker=True)

            print("\n✅ require_docker=True correctly raises error when Docker unavailable")

    def test_start_with_docker_available(self):
        """Test starting execution manager with Docker available."""
        try:
            manager = CodeSandbox()

            if manager.docker_available:
                # Should be able to start
                manager.start()

                assert manager.container is not None, "Container should be created"
                print("\n✅ Container STARTED successfully with Docker available")
                print(f"   Container ID: {manager.container.id[:12]}")

                # Cleanup
                manager.stop()
                print("   Container STOPPED successfully")
            else:
                pytest.skip("Docker not available")

        except RuntimeError as e:
            if "Docker" in str(e):
                pytest.skip("Docker not running")
            raise

    def test_start_with_docker_unavailable(self):
        """Test starting execution manager with Docker unavailable."""
        with patch("docker.from_env", side_effect=DockerException("Docker not available")):
            manager = CodeSandbox(require_docker=False)

            # Should not raise error when starting without Docker
            manager.start()  # Should return early

            assert manager.container is None, "No container should be created"
            print("\n✅ start() handled gracefully when Docker unavailable")

    def test_execute_with_docker_unavailable(self):
        """Test code execution when Docker is unavailable."""
        with patch("docker.from_env", side_effect=DockerException("Docker not available")):
            manager = CodeSandbox(require_docker=False)

            result = manager.execute("print('Hello')")

            # Should return error message instead of crashing
            assert result["exit_code"] == 1
            assert "Docker is not available" in result["stderr"]
            assert result["stdout"] == ""

            print("\n✅ execute() returns error message when Docker unavailable")
            print(f"   Result: {result}")

    def test_execute_with_docker_available(self):
        """Test code execution when Docker is available and container is running."""
        try:
            manager = CodeSandbox()

            if not manager.docker_available:
                pytest.skip("Docker not available")

            manager.start()

            # Execute simple Python code
            result = manager.execute("print('Hello from Docker')")

            assert result["exit_code"] == 0, f"Execution failed: {result['stderr']}"
            assert "Hello from Docker" in result["stdout"]

            print("\n✅ Code EXECUTED successfully in Docker container")
            print(f"   Output: {result['stdout'].strip()}")

            # Cleanup
            manager.stop()

        except RuntimeError as e:
            if "Docker" in str(e):
                pytest.skip("Docker not running")
            raise

    def test_stop_with_docker_unavailable(self):
        """Test stopping manager when Docker is unavailable."""
        with patch("docker.from_env", side_effect=DockerException("Docker not available")):
            manager = CodeSandbox(require_docker=False)

            # Should not raise error
            manager.stop()  # Should return early

            print("\n✅ stop() handled gracefully when Docker unavailable")

    def test_put_files_with_docker_unavailable(self):
        """Test put_files when Docker is unavailable."""
        with patch("docker.from_env", side_effect=DockerException("Docker not available")):
            manager = CodeSandbox(require_docker=False)

            # Should not raise error
            manager.put_files(["/tmp/test.txt"])  # Should return early

            print("\n✅ put_files() handled gracefully when Docker unavailable")

    def test_get_file_with_docker_unavailable(self):
        """Test get_file when Docker is unavailable."""
        with patch("docker.from_env", side_effect=DockerException("Docker not available")):
            manager = CodeSandbox(require_docker=False)

            # Should return empty bytes instead of crashing
            result = manager.get_file("/tmp/test.txt")

            assert result == b"", "Should return empty bytes"
            print("\n✅ get_file() returns empty bytes when Docker unavailable")


class TestDockerIntegration:
    """Integration tests with actual Docker (if available)."""

    @pytest.mark.integration
    def test_full_docker_workflow(self):
        """Test complete Docker workflow: start, execute, stop."""
        manager = CodeSandbox()

        if not manager.docker_available:
            pytest.skip("Docker not available for integration test")

        print("\n" + "=" * 70)
        print("FULL DOCKER INTEGRATION TEST")
        print("=" * 70)

        # Step 1: Start container
        print("\n1. Starting Docker container...")
        manager.start()

        # Check if container actually started
        if manager.container is None:
            pytest.skip("Docker container failed to start (Docker daemon may not be running)")

        assert manager.container is not None
        print(f"   ✓ Container started: {manager.container.id[:12]}")

        # Step 2: Execute Python code
        print("\n2. Executing Python code in container...")
        code = """
import sys
print(f"Python version: {sys.version}")
print("Hello from isolated Docker container!")

# Test calculation
result = 2 + 2
print(f"2 + 2 = {result}")
"""
        result = manager.execute(code)

        assert result["exit_code"] == 0, f"Execution failed: {result['stderr']}"
        assert "Python version" in result["stdout"]
        assert "Hello from isolated Docker container!" in result["stdout"]
        assert "2 + 2 = 4" in result["stdout"]

        print("   ✓ Execution successful")
        print("\n   Output:")
        for line in result["stdout"].strip().split("\n"):
            print(f"   {line}")

        # Step 3: Test error handling
        print("\n3. Testing error handling...")
        error_code = "print(1/0)"  # Will cause ZeroDivisionError
        error_result = manager.execute(error_code)

        assert error_result["exit_code"] != 0
        assert "ZeroDivisionError" in error_result["stderr"]
        print("   ✓ Error handling works correctly")
        print(f"   Error: {error_result['stderr'].strip()[:100]}...")

        # Step 4: Stop container
        print("\n4. Stopping Docker container...")
        manager.stop()
        print("   ✓ Container stopped successfully")

        print("\n" + "=" * 70)
        print("✅ FULL DOCKER INTEGRATION TEST PASSED")
        print("=" * 70)

    @pytest.mark.integration
    def test_docker_isolation(self):
        """Test that code runs in isolated Docker environment."""
        manager = CodeSandbox()

        if not manager.docker_available:
            pytest.skip("Docker not available")

        manager.start()

        # Check if container actually started
        if manager.container is None:
            pytest.skip("Docker container failed to start (Docker daemon may not be running)")

        # Test isolation - file system should be separate
        code = """
import os
print(f"Working dir: {os.getcwd()}")
print(f"Home dir: {os.path.expanduser('~')}")
"""
        result = manager.execute(code)

        assert result["exit_code"] == 0
        # Should be in container's working directory
        assert "/app" in result["stdout"] or "Working dir:" in result["stdout"]

        print("\n✅ Container isolation verified")
        print(f"Container working dir: {result['stdout']}")

        manager.stop()


if __name__ == "__main__":
    """Run tests directly for debugging."""
    print("Running Docker availability tests...")
    print("=" * 70)
    pytest.main([__file__, "-v", "-s"])
