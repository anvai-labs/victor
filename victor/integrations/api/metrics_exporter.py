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

"""Event Bridge Metrics Exporter for Prometheus.

Exports EventBridge reliability metrics in Prometheus format for external
monitoring and alerting.

Usage:
    from victor.integrations.api.metrics_exporter import export_prometheus_metrics

    metrics = export_prometheus_metrics()
    print(metrics)
"""

from typing import Dict, Any


def export_prometheus_metrics() -> str:
    """Export EventBridge metrics in Prometheus text format.

    Returns:
        Prometheus-formatted metrics string
    """
    from victor.integrations.api.event_bridge import EventBroadcaster

    broadcaster = EventBroadcaster()
    metrics = broadcaster.get_reliability_dashboard()

    lines = []

    # HELP and TYPE metadata
    lines.append("# HELP victor_eventbridge_events_total Total number of events dispatched")
    lines.append("# TYPE victor_eventbridge_events_total counter")
    lines.append(f"victor_eventbridge_events_total {metrics['events_dispatched']}")

    lines.append("")
    lines.append("# HELP victor_eventbridge_send_successes_total Total successful sends to clients")
    lines.append("# TYPE victor_eventbridge_send_successes_total counter")
    lines.append(f"victor_eventbridge_send_successes_total {metrics['send_successes']}")

    lines.append("")
    lines.append("# HELP victor_eventbridge_send_failures_total Total failed sends to clients")
    lines.append("# TYPE victor_eventbridge_send_failures_total counter")
    lines.append(f"victor_eventbridge_send_failures_total {metrics['send_failures']}")

    lines.append("")
    lines.append("# HELP victor_eventbridge_delivery_success_rate Delivery success rate (0-1)")
    lines.append("# TYPE victor_eventbridge_delivery_success_rate gauge")
    lines.append(f"victor_eventbridge_delivery_success_rate {metrics['delivery_success_rate']:.6f}")

    lines.append("")
    lines.append("# HELP victor_eventbridge_dispatch_latency_ms Dispatch latency in milliseconds")
    lines.append("# TYPE victor_eventbridge_dispatch_latency_ms gauge")
    lines.append(
        f"victor_eventbridge_dispatch_latency_p95_ms {metrics['dispatch_latency_p95_ms']:.3f}"
    )

    # SLO status indicators (1 = true, 0 = false)
    lines.append("")
    lines.append(
        "# HELP victor_eventbridge_slo_delivery_success_rate Delivery success rate SLO status"
    )
    lines.append("# TYPE victor_eventbridge_slo_delivery_success_rate gauge")
    slo_status = 1 if metrics["slo_status"]["delivery_success_rate"] else 0
    lines.append(f"victor_eventbridge_slo_delivery_success_rate {slo_status}")

    lines.append("")
    lines.append("# HELP victor_eventbridge_slo_dispatch_latency Dispatch latency SLO status")
    lines.append("# TYPE victor_eventbridge_slo_dispatch_latency gauge")
    slo_status = 1 if metrics["slo_status"]["dispatch_latency_p95_ms"] else 0
    lines.append(f"victor_eventbridge_slo_dispatch_latency {slo_status}")

    # SLO thresholds as constants
    lines.append("")
    lines.append(
        "# HELP victor_eventbridge_slo_threshold_delivery_success_rate Delivery success rate SLO threshold"
    )
    lines.append("# TYPE victor_eventbridge_slo_threshold_delivery_success_rate gauge")
    lines.append(
        f"victor_eventbridge_slo_threshold_delivery_success_rate {metrics['slo_thresholds']['delivery_success_rate_min']:.6f}"
    )

    lines.append("")
    lines.append(
        "# HELP victor_eventbridge_slo_threshold_dispatch_latency_ms Dispatch latency SLO threshold in ms"
    )
    lines.append("# TYPE victor_eventbridge_slo_threshold_dispatch_latency_ms gauge")
    lines.append(
        f"victor_eventbridge_slo_threshold_dispatch_latency_ms {metrics['slo_thresholds']['dispatch_latency_p95_ms_max']:.3f}"
    )

    return "\n".join(lines)


def export_json_metrics() -> Dict[str, Any]:
    """Export EventBridge metrics as JSON.

    Returns:
        Dictionary with all metrics and metadata
    """
    from victor.integrations.api.event_bridge import EventBroadcaster

    broadcaster = EventBroadcaster()
    metrics = broadcaster.get_reliability_dashboard()

    return {
        "metadata": {
            "version": "1.0",
            "exported_at": __import__("time").time(),
            "slo_documentation": "docs/observability/slo-dashboard.md",
        },
        "metrics": metrics,
    }


def print_metrics_table() -> None:
    """Print a formatted table of current metrics."""
    from victor.integrations.api.event_bridge import EventBroadcaster

    broadcaster = EventBroadcaster()
    metrics = broadcaster.get_reliability_dashboard()

    print()
    print("=" * 60)
    print("Event Bridge SLO Dashboard")
    print("=" * 60)
    print()
    print(f"Events Dispatched:     {metrics['events_dispatched']:,}")
    print(f"Send Attempts:         {metrics['total_send_attempts']:,}")
    print(f"Send Successes:        {metrics['send_successes']:,}")
    print(f"Send Failures:         {metrics['send_failures']:,}")
    print()
    print(f"Delivery Success Rate: {metrics['delivery_success_rate']:.4%}")
    print(f"p95 Dispatch Latency:  {metrics['dispatch_latency_p95_ms']:.1f}ms")
    print()
    print("SLO Thresholds:")
    print(f"  Delivery Rate:       {metrics['slo_thresholds']['delivery_success_rate_min']:.1%}")
    print(
        f"  p95 Latency:         {metrics['slo_thresholds']['dispatch_latency_p95_ms_max']:.0f}ms"
    )
    print()
    print("SLO Status:")
    delivery_icon = "✅" if metrics["slo_status"]["delivery_success_rate"] else "❌"
    latency_icon = "✅" if metrics["slo_status"]["dispatch_latency_p95_ms"] else "❌"
    print(
        f"  Delivery Success:    {delivery_icon} {'PASS' if metrics['slo_status']['delivery_success_rate'] else 'FAIL'}"
    )
    print(
        f"  p95 Latency:         {latency_icon} {'PASS' if metrics['slo_status']['dispatch_latency_p95_ms'] else 'FAIL'}"
    )
    print()
    print("=" * 60)
    print()


def main() -> None:
    """CLI entry point for metrics export."""
    import argparse

    parser = argparse.ArgumentParser(description="Export EventBridge metrics")
    parser.add_argument(
        "--format",
        type=str,
        default="table",
        choices=["table", "prometheus", "json"],
        help="Output format",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file (default: stdout)",
    )

    args = parser.parse_args()

    if args.format == "table":
        print_metrics_table()
    elif args.format == "prometheus":
        output = export_prometheus_metrics()
    elif args.format == "json":
        import json

        output = json.dumps(export_json_metrics(), indent=2)

    if args.format != "table":
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Metrics exported to: {args.output}")
        else:
            print(output)


if __name__ == "__main__":
    main()
