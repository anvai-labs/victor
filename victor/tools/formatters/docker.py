"""Docker operations formatter with Rich markup."""

from typing import Dict, Any

from .base import ToolFormatter, FormattedOutput


class DockerFormatter(ToolFormatter):
    """Format Docker operations output with Rich markup.

    Produces formatted output for:
    - Container status (green=running, red=stopped, yellow=paused)
    - Image information
    - Volume and network status
    - Service status for Docker Compose
    """

    def validate_input(self, data: Dict) -> bool:
        """Validate Docker result has required fields."""
        return isinstance(data, dict) and (
            "containers" in data
            or "images" in data
            or "volumes" in data
            or "services" in data
            or "output" in data
        )

    def format(
        self, data: Dict[str, Any], operation: str = "ps", max_items: int = 20, **kwargs
    ) -> FormattedOutput:
        """Format Docker operations output with Rich markup.

        Args:
            data: Docker result dict with containers, images, volumes, etc.
            operation: Docker operation type (ps, images, volumes, services, etc.)
            max_items: Maximum items to display (default: 20)

        Returns:
            FormattedOutput with Rich markup
        """
        # Route to appropriate formatter based on operation/data structure
        if "containers" in data or operation == "ps":
            return self._format_containers(data, max_items)
        elif "images" in data or operation == "images":
            return self._format_images(data, max_items)
        elif "volumes" in data or operation == "volumes":
            return self._format_volumes(data, max_items)
        elif "services" in data or operation == "services":
            return self._format_services(data, max_items)
        else:
            # Generic output formatting
            output = data.get("output", "")
            return FormattedOutput(
                content=output,
                format_type="plain",
                summary=f"docker {operation}",
                line_count=len(output.splitlines()) if output else 0,
                contains_markup=False,
            )

    def _format_containers(self, data: Dict, max_items: int) -> FormattedOutput:
        """Format Docker containers list."""
        containers = data.get("containers", [])
        lines = []

        if not containers:
            lines.append("[dim]No containers[/]")
        else:
            running_count = sum(1 for c in containers if c.get("state") == "running")
            lines.append(
                f"[bold]Containers:[/] [dim]{running_count} running, {len(containers)} total[/]"
            )
            lines.append("")

            for container in containers[:max_items]:
                container_id = container.get("id", "")[:12]
                name = container.get("name", "")
                image = container.get("image", "")
                state = container.get("state", "unknown")
                ports = container.get("ports", "")

                # Color-code state
                if state == "running":
                    state_color = "green"
                    icon = "●"
                elif state == "stopped":
                    state_color = "red"
                    icon = "○"
                elif state == "paused":
                    state_color = "yellow"
                    icon = "◐"
                else:
                    state_color = "white"
                    icon = "?"

                lines.append(f"  [{state_color}]{icon}[/] [bold]{name}[/] [dim]({container_id})[/]")
                lines.append(f"    [dim cyan]{image}[/] [dim]• {state}[/]")
                if ports:
                    lines.append(f"    [dim]Ports: {ports}[/]")
                lines.append("")  # Blank line between containers

            if len(containers) > max_items:
                lines.append(f"[dim]... and {len(containers) - max_items} more containers[/]")

        return FormattedOutput(
            content="\n".join(lines),
            format_type="rich",
            summary=f"{len(containers)} containers",
            line_count=len(lines),
            contains_markup=True,
        )

    def _format_images(self, data: Dict, max_items: int) -> FormattedOutput:
        """Format Docker images list."""
        images = data.get("images", [])
        lines = []

        if not images:
            lines.append("[dim]No images[/]")
        else:
            lines.append(f"[bold]Images:[/] [dim]{len(images)} total[/]")
            lines.append("")

            for image in images[:max_items]:
                image_id = image.get("id", "")[:12]
                tags = image.get("tags", [])
                size = image.get("size", "")
                created = image.get("created", "")

                tag_str = tags[0] if tags else "<none>"
                lines.append(f"  [bold]{tag_str}[/] [dim]({image_id})[/]")
                lines.append(f"    [dim]Size: {size} • Created: {created}[/]")
                lines.append("")  # Blank line between images

            if len(images) > max_items:
                lines.append(f"[dim]... and {len(images) - max_items} more images[/]")

        return FormattedOutput(
            content="\n".join(lines),
            format_type="rich",
            summary=f"{len(images)} images",
            line_count=len(lines),
            contains_markup=True,
        )

    def _format_volumes(self, data: Dict, max_items: int) -> FormattedOutput:
        """Format Docker volumes list."""
        volumes = data.get("volumes", [])
        lines = []

        if not volumes:
            lines.append("[dim]No volumes[/]")
        else:
            lines.append(f"[bold]Volumes:[/] [dim]{len(volumes)} total[/]")
            lines.append("")

            for volume in volumes[:max_items]:
                name = volume.get("name", "")
                driver = volume.get("driver", "local")
                mountpoint = volume.get("mountpoint", "")

                lines.append(f"  [bold cyan]{name}[/] [dim]({driver})[/]")
                lines.append(f"    [dim]{mountpoint}[/]")
                lines.append("")  # Blank line between volumes

            if len(volumes) > max_items:
                lines.append(f"[dim]... and {len(volumes) - max_items} more volumes[/]")

        return FormattedOutput(
            content="\n".join(lines),
            format_type="rich",
            summary=f"{len(volumes)} volumes",
            line_count=len(lines),
            contains_markup=True,
        )

    def _format_services(self, data: Dict, max_items: int) -> FormattedOutput:
        """Format Docker Compose services status."""
        services = data.get("services", [])
        lines = []

        if not services:
            lines.append("[dim]No services[/]")
        else:
            lines.append(f"[bold]Services:[/] [dim]{len(services)} total[/]")
            lines.append("")

            for service in services[:max_items]:
                name = service.get("name", "")
                state = service.get("state", "unknown")
                replicas = service.get("replicas", 0)

                # Color-code state
                if state == "running":
                    state_color = "green"
                    icon = "●"
                elif state == "stopped":
                    state_color = "red"
                    icon = "○"
                else:
                    state_color = "yellow"
                    icon = "?"

                lines.append(f"  [{state_color}]{icon}[/] [bold]{name}[/] [dim]• {state}[/]")
                if replicas > 0:
                    lines.append(f"    [dim]{replicas} replica{'s' if replicas != 1 else ''}[/]")
                lines.append("")  # Blank line between services

            if len(services) > max_items:
                lines.append(f"[dim]... and {len(services) - max_items} more services[/]")

        return FormattedOutput(
            content="\n".join(lines),
            format_type="rich",
            summary=f"{len(services)} services",
            line_count=len(lines),
            contains_markup=True,
        )
