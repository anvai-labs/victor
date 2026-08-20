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

"""Network operations formatter for Rich console output."""

from typing import Any, Dict

from .base import ToolFormatter, FormattedOutput


class NetworkFormatter(ToolFormatter):
    """Formatter for network operations (ping, traceroute, dns).

    Provides color-coded output for:
    - Ping statistics (green=good, red=timeout, yellow=slow)
    - Traceroute hops
    - DNS lookup results
    - Network latency (ms)
    - Packet loss percentages
    """

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate network operation data.

        Args:
            data: Network operation result

        Returns:
            True if data is valid, False otherwise
        """
        return isinstance(data, dict) and (
            "operation" in data
            or "host" in data
            or "ping" in data
            or "traceroute" in data
            or "dns" in data
            or "latency_ms" in data
        )

    def format(self, data: Dict[str, Any], **kwargs) -> FormattedOutput:
        """Format network operation results with Rich markup.

        Args:
            data: Network operation result
            **kwargs: Additional options (max_hops, show_details, etc.)

        Returns:
            FormattedOutput with Rich markup
        """
        operation = data.get("operation", "").lower()

        if "ping" in operation or "latency_ms" in data:
            return self._format_ping(data, **kwargs)
        elif "traceroute" in operation or "hops" in data:
            return self._format_traceroute(data, **kwargs)
        elif "dns" in operation or "lookup" in operation:
            return self._format_dns(data, **kwargs)
        else:
            return self._format_generic(data, **kwargs)

    def _format_ping(self, data: Dict[str, Any], **kwargs) -> FormattedOutput:
        """Format ping results.

        Args:
            data: Ping operation result
            **kwargs: Additional options

        Returns:
            FormattedOutput with Rich markup
        """
        lines = []

        host = data.get("host", data.get("target", "unknown"))
        latency = data.get("latency_ms", data.get("avg_latency_ms", 0))
        packet_loss = data.get("packet_loss", 0)
        packets_sent = data.get("packets_sent", data.get("count", 0))
        packets_received = data.get("packets_received", 0)

        # Header with host and status
        lines.append(f"[bold cyan]PING {host}[/]")
        lines.append("")

        # Overall status
        if packet_loss == 0:
            lines.append(f"[green]✓ Host is up[/] - [cyan]{latency:.1f}ms[/] latency")
        elif packet_loss < 50:
            lines.append(f"[yellow]⚠ Intermittent connectivity[/] - {packet_loss:.1f}% packet loss")
        else:
            lines.append(f"[red]✗ Host is down[/] - {packet_loss:.1f}% packet loss")

        lines.append("")

        # Statistics
        if packets_sent > 0:
            lines.append("[bold]Statistics:[/]")
            lines.append(f"  Packets: [green]{packets_sent} sent[/], {packets_received} received")
            lines.append(f"  Loss: [red]{packet_loss:.1f}%[/]")

            if "min_latency_ms" in data:
                lines.append(
                    f"  Latency: min [cyan]{data['min_latency_ms']:.1f}ms[/], "
                    f"avg [cyan]{data['avg_latency_ms']:.1f}ms[/], "
                    f"max [cyan]{data['max_latency_ms']:.1f}ms[/]"
                )

        content = "\n".join(lines)
        summary = f"{host}: {latency:.1f}ms, {packet_loss:.1f}% loss"

        return FormattedOutput(
            content=content,
            format_type="rich",
            summary=summary,
            contains_markup=True,
        )

    def _format_traceroute(self, data: Dict[str, Any], **kwargs) -> FormattedOutput:
        """Format traceroute results.

        Args:
            data: Traceroute operation result
            **kwargs: Additional options

        Returns:
            FormattedOutput with Rich markup
        """
        max_hops = kwargs.get("max_hops", 30)

        lines = []

        target = data.get("host", data.get("target", "unknown"))
        hops = data.get("hops", data.get("traceroute", []))

        # Header
        lines.append(f"[bold cyan]TRACEROUTE {target}[/]")
        lines.append("")

        # Show each hop
        for i, hop in enumerate(hops[:max_hops]):
            hop_num = hop.get("hop", i + 1)
            host = hop.get("host", "*")
            ip = hop.get("ip", "")
            latency_ms = hop.get("latency_ms", [])

            # Color-code based on latency
            if latency_ms and len(latency_ms) > 0:
                avg_latency = sum(latency_ms) / len(latency_ms)
                if avg_latency < 50:
                    color = "green"
                elif avg_latency < 150:
                    color = "yellow"
                else:
                    color = "red"

                latency_str = ", ".join(f"[{color}]{ms:.1f}ms[/]" for ms in latency_ms[:3])
            else:
                latency_str = "[red]*[/]"
                color = "red"

            lines.append(f"  [{color}]{hop_num:2d}[/] {host} ({ip}) {latency_str}")

        if len(hops) > max_hops:
            lines.append(f"  [dim]... and {len(hops) - max_hops} more hops[/]")

        content = "\n".join(lines)
        summary = f"{target}: {len(hops)} hops"

        return FormattedOutput(
            content=content,
            format_type="rich",
            summary=summary,
            contains_markup=True,
        )

    def _format_dns(self, data: Dict[str, Any], **kwargs) -> FormattedOutput:
        """Format DNS lookup results.

        Args:
            data: DNS lookup result
            **kwargs: Additional options

        Returns:
            FormattedOutput with Rich markup
        """
        lines = []

        domain = data.get("domain", data.get("host", "unknown"))
        records = data.get("records", data.get("results", []))

        # Header
        lines.append(f"[bold cyan]DNS Lookup: {domain}[/]")
        lines.append("")

        # Show records
        if records:
            lines.append("[bold]Records:[/]")
            for record in records:
                record_type = record.get("type", "A")
                record_value = record.get("value", record.get("data", ""))
                ttl = record.get("ttl", "")

                if ttl:
                    lines.append(f"  [{record_type}] {record_value} [dim](TTL: {ttl}s)[/]")
                else:
                    lines.append(f"  [{record_type}] {record_value}[/]")
        else:
            lines.append("[dim]No records found[/]")

        content = "\n".join(lines)
        summary = f"{domain}: {len(records)} records"

        return FormattedOutput(
            content=content,
            format_type="rich",
            summary=summary,
            contains_markup=True,
        )

    def _format_generic(self, data: Dict[str, Any], **kwargs) -> FormattedOutput:
        """Format generic network operation results.

        Args:
            data: Network operation result
            **kwargs: Additional options

        Returns:
            FormattedOutput with Rich markup
        """
        lines = []

        operation = data.get("operation", "Network operation")
        lines.append(f"[bold cyan]{operation}[/]")
        lines.append("")

        for key, value in data.items():
            if key not in ("operation", "success"):
                lines.append(f"  [bold]{key}:[/] {value}")

        content = "\n".join(lines)
        summary = operation

        return FormattedOutput(
            content=content,
            format_type="rich",
            summary=summary,
            contains_markup=True,
        )

    def _extract_summary(self, data: Dict[str, Any]) -> str:
        """Extract summary from network operation data.

        Args:
            data: Network operation result

        Returns:
            Summary string
        """
        host = data.get("host", data.get("target", "network operation"))
        latency = data.get("latency_ms", data.get("avg_latency_ms"))

        if latency is not None:
            return f"{host}: {latency:.1f}ms"

        return host

    def get_fallback(self) -> "ToolFormatter":
        """Return fallback formatter.

        Returns:
            GenericFormatter instance
        """
        from .generic import GenericFormatter

        return GenericFormatter()
