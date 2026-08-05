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

"""Docker service provider for local container management.

Uses the Docker SDK to manage containers as workflow services.

Example:
    provider = DockerServiceProvider()

    config = ServicePresets.postgres()
    handle = await provider.start(config)

    # Do work...

    await provider.stop(handle)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from victor.workflows.services.definition import (
    PortMapping,
    ServiceConfig,
    ServiceHandle,
    ServiceStartError,
    ServiceState,
    VolumeMount,
)
from victor.workflows.services.providers.base import BaseServiceProvider

logger = logging.getLogger(__name__)

# Optional Docker SDK import
try:
    import docker
    from docker.errors import APIError, ContainerError, ImageNotFound, NotFound
    from docker.models.containers import Container

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None  # type: ignore


class DockerServiceProvider(BaseServiceProvider):
    """Service provider using local Docker daemon.

    Manages containers with proper port mapping, volume mounts,
    and health checking.

    Attributes:
        client: Docker client instance
        network_name: Default network for services
        label_prefix: Label prefix for managed containers
    """

    def __init__(
        self,
        docker_host: Optional[str] = None,
        network_name: str = "victor-services",
        label_prefix: str = "victor.service",
    ):
        """Initialize Docker provider.

        Args:
            docker_host: Docker daemon URL (default: from environment)
            network_name: Network name for service containers
            label_prefix: Label prefix for identifying managed containers
        """
        if not DOCKER_AVAILABLE:
            raise ImportError("Docker SDK not available. Install with: pip install docker")

        self._docker_host = docker_host
        self._network_name = network_name
        self._label_prefix = label_prefix
        self._client: Optional[docker.DockerClient] = None

    @property
    def client(self) -> docker.DockerClient:
        """Get or create Docker client."""
        if self._client is None:
            if self._docker_host:
                self._client = docker.DockerClient(base_url=self._docker_host)
            else:
                self._client = docker.from_env()
        return self._client

    def _ensure_network(self) -> None:
        """Ensure the service network exists."""
        try:
            self.client.networks.get(self._network_name)
        except NotFound:
            logger.info(f"Creating Docker network: {self._network_name}")
            self.client.networks.create(
                self._network_name,
                driver="bridge",
                labels={f"{self._label_prefix}.managed": "true"},
            )

    def _build_container_config(
        self,
        config: ServiceConfig,
        handle: ServiceHandle,
    ) -> Dict[str, Any]:
        """Build Docker container configuration."""
        container_config: Dict[str, Any] = {
            "image": config.image,
            "name": handle.service_id,
            "detach": True,
            "environment": config.environment,
            "labels": {
                f"{self._label_prefix}.name": config.name,
                f"{self._label_prefix}.id": handle.service_id,
                f"{self._label_prefix}.managed": "true",
                **{f"{self._label_prefix}.{k}": v for k, v in config.labels.items()},
            },
        }

        # Port mappings
        if config.ports:
            ports = {}
            for pm in config.ports:
                port_key = f"{pm.container_port}/{pm.protocol}"
                if pm.host_port == 0:
                    # Auto-assign port
                    ports[port_key] = None
                else:
                    ports[port_key] = (pm.host_ip, pm.host_port)
            container_config["ports"] = ports

        # Volume mounts
        if config.volumes:
            volumes = {}
            for vm in config.volumes:
                volumes[vm.source] = {
                    "bind": vm.target,
                    "mode": "ro" if vm.read_only else "rw",
                }
            container_config["volumes"] = volumes

        # Command and entrypoint
        if config.command:
            container_config["command"] = config.command
        if config.entrypoint:
            container_config["entrypoint"] = config.entrypoint

        # Working directory
        if config.working_dir:
            container_config["working_dir"] = config.working_dir

        # User
        if config.user:
            container_config["user"] = config.user

        # Resource limits
        if config.memory_limit:
            container_config["mem_limit"] = config.memory_limit
        if config.cpu_limit:
            container_config["cpu_period"] = 100000
            container_config["cpu_quota"] = int(config.cpu_limit * 100000)

        # Network
        if config.network:
            container_config["network"] = config.network
        elif config.network_mode:
            container_config["network_mode"] = config.network_mode
        else:
            self._ensure_network()
            container_config["network"] = self._network_name

        # Security
        if config.privileged:
            container_config["privileged"] = True
        if config.cap_add:
            container_config["cap_add"] = config.cap_add
        if config.cap_drop:
            container_config["cap_drop"] = config.cap_drop

        return container_config

    async def _do_start(self, config: ServiceConfig) -> ServiceHandle:
        """Start a Docker container."""
        if not config.image:
            raise ServiceStartError(config.name, "No image specified")

        handle = ServiceHandle.create(config)

        try:
            # Pull image if needed
            try:
                await asyncio.to_thread(self.client.images.get, config.image)
            except ImageNotFound:
                logger.info(f"Pulling image: {config.image}")
                await asyncio.to_thread(self.client.images.pull, config.image)

            # Build container config
            container_config = await asyncio.to_thread(self._build_container_config, config, handle)

            # Create and start container
            container: Container = await asyncio.to_thread(
                self.client.containers.run,
                **container_config,
            )

            handle.container_id = container.id
            handle.state = ServiceState.STARTING

            # Get actual port mappings
            await asyncio.to_thread(container.reload)

            ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
            for pm in config.ports:
                port_key = f"{pm.container_port}/{pm.protocol}"
                port_info = ports.get(port_key)
                if port_info and len(port_info) > 0:
                    host_port = int(port_info[0]["HostPort"])
                    handle.ports[pm.container_port] = host_port

            # Get container IP if on custom network
            networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})
            if self._network_name in networks:
                handle.metadata["container_ip"] = networks[self._network_name].get("IPAddress")

            logger.info(f"Started container {handle.container_id[:12]} for '{config.name}'")
            return handle

        except (APIError, ContainerError, ImageNotFound) as e:
            handle.state = ServiceState.FAILED
            handle.error = str(e)
            raise ServiceStartError(config.name, str(e))

    async def _do_stop(self, handle: ServiceHandle, grace_period: float) -> None:
        """Stop a Docker container gracefully."""
        if not handle.container_id:
            return

        try:
            container = await asyncio.to_thread(
                self.client.containers.get,
                handle.container_id,
            )

            logger.debug(
                f"Stopping container {handle.container_id[:12]} " f"(grace={grace_period}s)"
            )

            await asyncio.to_thread(container.stop, timeout=int(grace_period))
            await asyncio.to_thread(container.remove)

            logger.info(f"Stopped and removed container {handle.container_id[:12]}")

        except NotFound:
            logger.debug(f"Container {handle.container_id} already removed")
        except Exception as e:
            logger.error(f"Failed to stop container {handle.container_id}: {e}")
            raise

    async def _do_cleanup(self, handle: ServiceHandle) -> None:
        """Force remove a Docker container."""
        if not handle.container_id:
            return

        try:
            container = await asyncio.to_thread(
                self.client.containers.get,
                handle.container_id,
            )

            await asyncio.to_thread(container.remove, force=True)

            logger.info(f"Force removed container {handle.container_id[:12]}")

        except NotFound:
            pass
        except Exception as e:
            logger.warning(f"Cleanup failed for container {handle.container_id}: {e}")

    async def get_logs(self, handle: ServiceHandle, tail: int = 100) -> str:
        """Get container logs."""
        if not handle.container_id:
            return ""

        try:
            container = await asyncio.to_thread(
                self.client.containers.get,
                handle.container_id,
            )

            logs = await asyncio.to_thread(container.logs, tail=tail, timestamps=True)

            return logs.decode("utf-8", errors="replace")

        except NotFound:
            return "[Container not found]"
        except Exception as e:
            return f"[Error getting logs: {e}]"

    async def _run_command_in_service(
        self,
        handle: ServiceHandle,
        command: str,
    ) -> Tuple[int, str]:
        """Execute a command inside the container."""
        if not handle.container_id:
            raise RuntimeError("No container ID")

        try:
            container = await asyncio.to_thread(
                self.client.containers.get,
                handle.container_id,
            )

            result = await asyncio.to_thread(container.exec_run, command, demux=True)

            exit_code = result.exit_code
            stdout = result.output[0] or b""
            stderr = result.output[1] or b""
            output = (stdout + stderr).decode("utf-8", errors="replace")

            return exit_code, output

        except Exception as e:
            logger.debug(f"Command execution failed: {e}")
            return 1, str(e)

    async def cleanup_all(self, label_filter: Optional[str] = None) -> int:
        """Clean up all managed containers.

        Args:
            label_filter: Additional label filter

        Returns:
            Number of containers removed
        """
        filters = {"label": f"{self._label_prefix}.managed=true"}
        if label_filter:
            filters["label"] = [filters["label"], label_filter]

        try:
            containers = await asyncio.to_thread(
                self.client.containers.list,
                filters=filters,
                all=True,
            )

            count = 0
            for container in containers:
                try:
                    await asyncio.to_thread(container.remove, force=True)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove container {container.id}: {e}")

            logger.info(f"Cleaned up {count} containers")
            return count

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0

    def list_running(self) -> List[Dict[str, Any]]:
        """List all running managed containers.

        Returns:
            List of container info dicts
        """
        filters = {"label": f"{self._label_prefix}.managed=true"}

        try:
            containers = self.client.containers.list(filters=filters)

            return [
                {
                    "id": c.id[:12],
                    "name": c.name,
                    "service": c.labels.get(f"{self._label_prefix}.name"),
                    "status": c.status,
                    "created": c.attrs.get("Created"),
                }
                for c in containers
            ]
        except Exception as e:
            logger.error(f"Failed to list containers: {e}")
            return []
