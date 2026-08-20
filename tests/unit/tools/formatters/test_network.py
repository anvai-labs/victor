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

"""Tests for Network formatter."""

import pytest

from victor.tools.formatters.network import NetworkFormatter
from victor.tools.formatters.base import FormattedOutput


class TestNetworkFormatter:
    """Test NetworkFormatter for network operations."""

    def test_validate_input_valid_with_operation(self):
        """Test validation with operation field."""
        formatter = NetworkFormatter()
        data = {"operation": "ping"}
        assert formatter.validate_input(data) is True

    def test_validate_input_valid_with_host(self):
        """Test validation with host field."""
        formatter = NetworkFormatter()
        data = {"host": "example.com"}
        assert formatter.validate_input(data) is True

    def test_validate_input_valid_with_latency(self):
        """Test validation with latency_ms field."""
        formatter = NetworkFormatter()
        data = {"latency_ms": 45.5}
        assert formatter.validate_input(data) is True

    def test_validate_input_invalid(self):
        """Test validation with invalid data."""
        formatter = NetworkFormatter()
        data = {"invalid": "data"}
        assert formatter.validate_input(data) is False

    def test_format_ping_success(self):
        """Test formatting successful ping."""
        formatter = NetworkFormatter()
        data = {
            "host": "example.com",
            "latency_ms": 45.5,
            "packet_loss": 0,
            "packets_sent": 4,
            "packets_received": 4,
        }

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert result.format_type == "rich"
        assert "[green]✓ Host is up[/]" in result.content
        assert "45.5ms" in result.content

    def test_format_ping_packet_loss(self):
        """Test formatting ping with packet loss."""
        formatter = NetworkFormatter()
        data = {
            "host": "example.com",
            "latency_ms": 0,
            "packet_loss": 25.0,  # Changed from 50.0 to be < 50
            "packets_sent": 4,
            "packets_received": 3,
        }

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "[yellow]⚠ Intermittent connectivity[/]" in result.content
        assert "25.0%" in result.content

    def test_format_ping_host_down(self):
        """Test formatting ping when host is down."""
        formatter = NetworkFormatter()
        data = {
            "operation": "ping",
            "host": "unreachable.com",
            "packet_loss": 100.0,
            "latency_ms": 0.0,
        }

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        # When packet_loss is 100%, it shows "Host is down"
        assert "Host is down" in result.content
        assert "100.0%" in result.content

    def test_format_ping_with_statistics(self):
        """Test formatting ping with detailed statistics."""
        formatter = NetworkFormatter()
        data = {
            "host": "example.com",
            "latency_ms": 50.0,
            "min_latency_ms": 45.0,
            "avg_latency_ms": 50.0,
            "max_latency_ms": 55.0,
            "packet_loss": 0,
            "packets_sent": 10,
        }

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "min" in result.content
        assert "avg" in result.content
        assert "max" in result.content

    def test_format_traceroute(self):
        """Test formatting traceroute results."""
        formatter = NetworkFormatter()
        data = {
            "operation": "traceroute",
            "host": "example.com",
            "hops": [
                {
                    "hop": 1,
                    "host": "router1",
                    "ip": "192.168.1.1",
                    "latency_ms": [10.5, 11.2, 10.8],
                },
                {
                    "hop": 2,
                    "host": "router2",
                    "ip": "10.0.0.1",
                    "latency_ms": [25.3, 26.1, 25.5],
                },
            ],
        }

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "TRACEROUTE example.com" in result.content
        assert "router1" in result.content
        assert "192.168.1.1" in result.content

    def test_format_dns_lookup(self):
        """Test formatting DNS lookup results."""
        formatter = NetworkFormatter()
        data = {
            "operation": "dns",
            "domain": "example.com",
            "records": [
                {"type": "A", "value": "93.184.216.34", "ttl": 300},
                {
                    "type": "AAAA",
                    "value": "2606:2800:220:1:248:1893:25c8:1946",
                    "ttl": 300,
                },
            ],
        }

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "DNS Lookup: example.com" in result.content
        assert "93.184.216.34" in result.content
        assert "[A]" in result.content

    def test_format_traceroute_with_max_hops(self):
        """Test formatting traceroute with max_hops limit."""
        formatter = NetworkFormatter()
        many_hops = [
            {"hop": i, "host": f"router{i}", "ip": f"10.0.0.{i}", "latency_ms": [10.0]}
            for i in range(50)
        ]
        data = {"operation": "traceroute", "host": "example.com", "hops": many_hops}

        result = formatter.format(data, max_hops=10)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "more hops" in result.content

    def test_summary_extraction_ping(self):
        """Test summary extraction for ping."""
        formatter = NetworkFormatter()
        data = {"host": "example.com", "latency_ms": 45.5}

        result = formatter.format(data)

        assert "example.com" in result.summary
        assert "45.5ms" in result.summary

    def test_fallback_formatter(self):
        """Test fallback formatter is returned."""
        formatter = NetworkFormatter()
        fallback = formatter.get_fallback()

        assert fallback is not None
        assert hasattr(fallback, "format")
