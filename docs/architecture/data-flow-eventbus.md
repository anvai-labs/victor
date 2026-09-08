# Event Backend Architecture

The event system lives in `victor.core.events`. Use the [event-bus guide](../guides/observability/event-bus.md)
for the current publishing and subscription interfaces, and the
[observability overview](../guides/observability/index.md) for metrics and telemetry.

`ObservabilityBus` carries telemetry; `AgentMessageBus` carries agent messages.
`create_event_backend()` selects a registered implementation. Its default is in-memory;
an unregistered backend falls back to in-memory with a warning. Register and configure a
backend before relying on persistence or cross-process delivery.

The configuration setting is `event_backend_type` (`VICTOR_EVENT_BACKEND_TYPE`), not the
removed `eventbus_backend` / `VICTOR_EVENTBUS_BACKEND`. JSONL metrics export is a separate
consumer, not a guarantee of distributed message delivery.

The previous factory guide described removed `victor.observability.event_bus_factory`
and `victor.observability.event_bus` modules. Its migration examples and old dashboard
recipes remain available in [git history](https://github.com/anvai-labs/victor/blob/b6b25a634/docs/architecture/data-flow-eventbus.md).

Source: `victor/core/events/__init__.py`, `victor/core/events/backends.py`, and
`victor/config/settings.py`.
